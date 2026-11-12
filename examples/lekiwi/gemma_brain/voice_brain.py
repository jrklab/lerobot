#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Talk to the brain: say a task, hear the plan, say "go", hear how it goes; "stop" halts it.

    cd ~/Work/lerobot/examples/lekiwi/gemma_brain && .venv/bin/python voice_brain.py

Run it in Terminal, which may use the microphone (Claude's sessions can't). Press Enter, then
speak: listening ends after a second of quiet. The plan is read back; answer "go" or "cancel"
(no Enter needed). During the task, say "stop" at any time: the robot halts as with Ctrl-C. The
gamepad's RB and the web page's STOP stay the emergency stops: speech takes a second or two.

Speech to text: whisper.cpp (whisper-cli, the small model: on the owner's voice through the Anker
PowerConf it got every task command exactly in ~0.3 s, Nov 11; Gemma E2B 4 of 9). The words go to
the planner as a typed goal does (plan_brain.py, which works as before without any of this).
Speech out: macOS `say` on the PowerConf. Answers with fixed choices (go/cancel, stop) take the
nearest allowed word: whisper heard "Go." as "Goal." on the same recordings.
"""

from __future__ import annotations

import _thread
import argparse
import contextlib
import datetime
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
from brain.geometry import Geometry
from brain.primitives import Skills
from brain.task_loop import PlanBrain
from common.arm import Arm
from common.gemma import LlamaCpp
from common.planner import Step
from common.robot import OperatorTookOver, Robot, RobotError
from common.runlog import Log
from plan_brain import FAST_URL, PLANNER_URL

RATE = 16000  # whisper's rate, and the PowerConf's own
BLOCK_S = 0.03  # the voice-activity check's step
QUIET_S = 1.0  # an utterance ends after this much quiet once speech began
MAX_UTTERANCE_S = 15.0
SPEECH_OVER_NOISE = 3.0  # a block is speech when this much louder than the room's noise
MIN_SPEECH_RMS = 300.0  # and at least this loud (int16), so a silent room doesn't count its hiss
ANSWER_WAIT_S = 8.0  # how long a go/cancel answer is waited for
WHISPER_MODEL = Path.home() / "models" / "whisper" / "ggml-small.bin"
LEAD_IN_MS = 300  # silence said before each sentence, while the speaker fades in
DEVICE_NAME = "PowerConf"  # the Anker speakerphone, microphone and speaker; else the defaults

logger = logging.getLogger(__name__)


# --- hearing ---


class Mic:
    """The microphone, always open: utterances cut from it by loudness against the room's noise."""

    def __init__(self, device: int | None):
        import sounddevice as sd  # only where there's a microphone (not in the tests)

        self.blocks: queue.Queue = queue.Queue()
        self.noise = MIN_SPEECH_RMS / SPEECH_OVER_NOISE
        self.stream = sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype="int16",
            device=device,
            blocksize=int(RATE * BLOCK_S),
            callback=lambda data, frames, t, status: self.blocks.put(data[:, 0].copy()),
        )
        self.stream.start()

    def drain(self) -> None:
        while not self.blocks.empty():
            self.blocks.get()

    def utterance(
        self, wait_s: float | None = None, stop: threading.Event | None = None
    ) -> np.ndarray | None:
        """The next thing said: from its first loud block to QUIET_S of quiet after it. None if
        nothing was said within wait_s (None: no limit) or `stop` was set."""
        self.drain()
        start = time.monotonic()
        got: list[np.ndarray] = []
        quiet = 0.0
        while True:
            if stop is not None and stop.is_set():
                return None
            try:
                b = self.blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
            loud = rms > max(MIN_SPEECH_RMS, SPEECH_OVER_NOISE * self.noise)
            if not got:
                if loud:
                    got.append(b)
                else:  # the room's noise, followed slowly
                    self.noise = 0.95 * self.noise + 0.05 * rms
                    if wait_s is not None and time.monotonic() - start > wait_s:
                        return None
                continue
            got.append(b)
            quiet = 0.0 if loud else quiet + BLOCK_S
            if quiet >= QUIET_S or len(got) * BLOCK_S >= MAX_UTTERANCE_S:
                return np.concatenate(got)


def transcribe(audio: np.ndarray, lang: str) -> str:
    """whisper.cpp's text of 16 kHz int16 audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(audio.tobytes())
        p = subprocess.run(
            ["whisper-cli", "-m", str(WHISPER_MODEL), "-f", f.name, "-l", lang, "-nt", "-np"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    return " ".join(line.strip() for line in p.stdout.splitlines() if line.strip())


def _edits(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


# The fixed answers, each with the words that say it. Transcribed in the language spoken, not
# translated: whisper's translation turned 积木 into "film" and "ink" (Nov 11, the owner's recordings).
ANSWERS = {"go": ("go", "start", "开始"), "cancel": ("cancel", "取消"), "stop": ("stop", "停")}


def nearest(text: str, options: list[str]) -> str | None:
    """Which of `options` (ANSWERS' keys) a short answer was: the nearest of their words by
    letters, if clearly nearer than the other options' and no further from it than the word's own
    length; None for anything else (a sentence, an unclear word). "Goal" is 2 letters from "go",
    5 from "cancel"."""
    said = re.sub(r"[^a-z一-鿿 ]", "", text.lower()).strip()
    if not said or len(said.split()) > 2:
        return None
    best = {o: min((_edits(said, w), len(w)) for w in ANSWERS[o]) for o in options}
    ranked = sorted((d, o) for o, (d, _) in best.items())
    d, option = ranked[0]
    clear = len(ranked) == 1 or ranked[1][0] - d >= 2
    return option if clear and d <= best[option][1] else None


# --- speaking ---


class Voice:
    """macOS `say`, one sentence after another, on the speakerphone when it's there."""

    def __init__(self):
        devices = subprocess.run(["say", "-a", "?"], capture_output=True, text=True).stdout
        self.device = next((ln.split(None, 1)[0] for ln in devices.splitlines() if DEVICE_NAME in ln), None)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            text = self.q.get()
            # A moment of silence first: the speakerphone fades in when a sentence opens its
            # stream, and the first word came out low (owner, Nov 11).
            subprocess.run(
                ["say"] + (["-a", self.device] if self.device else []) + [f"[[slnc {LEAD_IN_MS}]] {text}"]
            )
            self.q.task_done()

    def say(self, text: str, wait: bool = False) -> None:
        print(f"  robot: {text}")
        self.q.put(text)
        if wait:
            self.q.join()


def plan_words(steps: list[Step]) -> str:
    """The plan as a sentence: "pick the tennis ball, put it in the box, then ..."."""
    rel = {"in": "in", "on": "on", "next_to": "next to"}
    parts = []
    for s in steps:
        if s.verb == "pick":
            parts.append(f"pick the {s.obj}")
        else:
            parts.append(f"put it {rel.get(s.rel, s.rel)} the {s.target}")
    pairs = [", ".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
    return ", then ".join(pairs)


class SpokenLog(Log):
    """The run log, and how the task goes, said as it happens."""

    def __init__(self, out: Path, voice: Voice):
        super().__init__(out)
        self.voice = voice

    def ev(self, kind: str, **kw) -> None:
        super().ev(kind, **kw)
        if kind == "step_result":
            verb, rest = kw["step"].split(" ", 1)
            names = re.findall(r'"([^"]+)"', rest)
            if kw.get("ok"):
                self.voice.say(
                    f"Got the {names[0]}." if verb == "pick" else f"The {names[0]} is in the {names[-1]}."
                )
            else:
                self.voice.say(f"Couldn't {verb} the {names[0]}. Trying again.")
        elif kind == "turn_to_remembered":
            self.voice.say(f"Turning to the {kw['obj']}.")


# --- the task ---


def run_task(robot: Robot, args, voice: Voice, mic: Mic, goal: str) -> None:
    out = Path(__file__).parent / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-voice")
    log = SpokenLog(out, voice)
    log.ev("heard", text=goal)
    with robot:
        arm = Arm(robot)
        sk = Skills(
            robot,
            arm,
            Geometry.load(),
            log,
            main=LlamaCpp(args.vision_url),
            fast=LlamaCpp(args.fast_url, "gemma4-e2b"),
        )
        pb = PlanBrain(sk, args.planner, log, robot)
        voice.say("Planning.")
        steps, why = pb.prepare(goal)
        if steps is None:
            voice.say(f"I can't do that. {why}")
            return
        voice.say(f"The plan: {plan_words(steps)}. Say go, or cancel.", wait=True)
        answer = None
        audio = mic.utterance(wait_s=ANSWER_WAIT_S)
        if audio is not None:
            heard = transcribe(audio, args.lang)
            answer = nearest(heard, ["go", "cancel"])
            print(f"  you: {heard!r} -> {answer}")
        log.ev("answer", answer=answer)
        if answer != "go":
            voice.say("Cancelled." if answer == "cancel" else "I didn't hear go. Cancelled.")
            pb.fold()
            return
        voice.say("Going.")
        stopped = threading.Event()
        done = threading.Event()

        def listen_for_stop() -> None:
            while not done.is_set():
                audio = mic.utterance(stop=done)
                if audio is None:
                    return
                heard = transcribe(audio, args.lang)
                if nearest(heard, ["stop"]) == "stop":
                    print(f"  you: {heard!r} -> stop")
                    stopped.set()
                    with contextlib.suppress(RobotError, OSError):
                        robot.stop()
                    if not done.is_set():  # the task still running: end it as Ctrl-C would
                        _thread.interrupt_main()
                    return

        listener = threading.Thread(target=listen_for_stop, daemon=True)
        listener.start()
        t0 = time.monotonic()
        ok, summary = False, ""
        try:
            ok, summary = pb.run(goal, steps=steps)
        except KeyboardInterrupt:
            summary = "stopped"
        except OperatorTookOver as e:
            summary = f"stopped: {e}"
        except OSError as e:
            summary = f"robot unreachable ({e})"
        finally:
            done.set()
        log.ev("result", success=ok, summary=summary)
        if stopped.is_set():
            voice.say("Stopped.")
            return
        if "stopped" in summary or "unreachable" in summary:
            voice.say(f"Stopped: {summary}.")
            return
        try:
            why = pb.fold()
            if not why and ok:
                why = pb.park()
        except (OSError, RobotError) as e:
            why = str(e)
        took = time.monotonic() - t0
        if ok:
            voice.say(
                f"Done, in {int(took // 60)} minutes {int(took % 60)} seconds."
                if took >= 60
                else f"Done, in {int(took)} seconds."
            )
        else:
            voice.say(f"I couldn't finish. {summary}")
        if why:
            voice.say(f"The arm or base stayed where it was: {why}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=None, help="LeKiwi web service URL (default: the first that answers)")
    ap.add_argument("--planner-url", default=PLANNER_URL)
    ap.add_argument("--vision-url", default=PLANNER_URL)
    ap.add_argument("--fast-url", default=FAST_URL)
    ap.add_argument(
        "--lang", default="auto", help="whisper's language: auto (English or Chinese, as spoken), en, zh"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not WHISPER_MODEL.exists():
        sys.exit(f"no whisper model at {WHISPER_MODEL}")
    import sounddevice as sd
    from common.planner import Planner

    args.planner = Planner(LlamaCpp(args.planner_url, "gemma4-26b"))
    device = next(
        (
            i
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and DEVICE_NAME in d["name"]
        ),
        None,
    )
    mic, voice = Mic(device), Voice()
    print(f"microphone: {sd.query_devices(device if device is not None else sd.default.device[0])['name']}")
    if sys.platform == "darwin":  # no idle sleep mid-task, as plan_brain
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    robot = Robot(args.robot)
    voice.say("Ready.")
    while True:
        try:
            if input("\nEnter, then say a task (q Enter quits): ").strip().lower() == "q":
                return 0
        except (EOFError, KeyboardInterrupt):
            return 0
        print("  listening...")
        audio = mic.utterance(wait_s=ANSWER_WAIT_S)
        if audio is None:
            voice.say("I didn't hear anything.")
            continue
        goal = transcribe(audio, args.lang)
        print(f"  you: {goal!r}")
        if not goal:
            voice.say("I didn't catch that.")
            continue
        try:
            run_task(robot, args, voice, mic, goal)
        except (RobotError, OSError) as e:
            voice.say(f"Robot error: {e}")


if __name__ == "__main__":
    sys.exit(main())
