"""Record spoken commands for the voice interface's speech-to-text comparison (Gemma E2B,
whisper.cpp). Run it in Terminal, which can use the microphone (Claude's sessions can't):

    cd ~/Work/lerobot/examples/lekiwi/gemma_brain && .venv/bin/python tools/record_voice.py
    .venv/bin/python tools/record_voice.py --lang zh     # the same commands in Chinese

For each prompt shown: press Enter, say it, press Enter again. Or type your own command first (it
becomes that recording's reference text), then Enter to start. "s" skips a prompt, "q" quits.
Recordings go to runs/voice/<time>/, 16 kHz mono WAV, with prompts.jsonl beside them (each file
and the words you were asked to say). The microphone is the Anker PowerConf when connected
(--device to pick another by name), else the Mac's default.
"""

import argparse
import datetime
import json
import queue
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 16000  # what the speech models take, and the PowerConf's own rate
OUT = Path(__file__).resolve().parents[1] / "runs" / "voice"
PROMPTS_ZH = [  # --lang zh: the same commands in Chinese
    "把网球放进盒子里。",
    "把红色积木放进碗里。",
    "把网球放进盒子里，然后把红色积木放进碗里。",
    "把绿色积木放进碗里。",
    "停。",
    "开始。",
    "取消。",
    "你在做什么？",
]
PROMPTS = [
    "Pick the tennis ball and put it in the box.",
    "Pick the red brick and put it into the bowl.",
    "Pick the tennis ball and put it in the box, after that pick the red brick and put it into the bowl.",
    "Put the green block in the bowl.",
    "Stop.",
    "Go.",
    "Cancel.",
    "What are you doing?",
]


def pick_device(name: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    return None


def record(device: int | None) -> np.ndarray:
    """From now until Enter: mono int16 at RATE."""
    chunks: queue.Queue = queue.Queue()
    with sd.InputStream(
        samplerate=RATE,
        channels=1,
        dtype="int16",
        device=device,
        callback=lambda data, frames, t, status: chunks.put(data.copy()),
    ):
        input("  recording... Enter to stop ")
    parts = []
    while not chunks.empty():
        parts.append(chunks.get())
    return np.concatenate(parts)[:, 0] if parts else np.zeros(0, np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device", default="PowerConf", help="part of the microphone's name (default: the Anker)"
    )
    ap.add_argument("--lang", default="en", choices=("en", "zh"), help="the prompts' language")
    args = ap.parse_args()
    device = pick_device(args.device)
    name = sd.query_devices(device if device is not None else sd.default.device[0])["name"]
    print(f"microphone: {name}" + ("" if device is not None else f" (no '{args.device}' found: the default)"))
    out = OUT / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ("-zh" if args.lang == "zh" else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"saving to {out}\n")
    n = 0
    for prompt in (PROMPTS_ZH if args.lang == "zh" else PROMPTS) + [None]:
        while True:
            shown = prompt or "(your own command)"
            typed = input(
                f'Say: "{shown}"\n  Enter to start, or type your own words then Enter, s to skip, q to quit: '
            ).strip()
            if typed.lower() == "q":
                print(f"\n{n} recordings in {out}")
                return
            if typed.lower() == "s":
                break
            text = typed or prompt
            if not text:
                print("  type the words you'll say, then Enter")
                continue
            audio = record(device)
            if len(audio) < RATE // 4:
                print("  too short, again")
                continue
            n += 1
            f = out / f"{n:02d}.wav"
            with wave.open(str(f), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(audio.tobytes())
            peak = int(np.abs(audio.astype(np.int32)).max())
            level = (
                "clipping, further away"
                if peak > 32000
                else "very quiet, come closer"
                if peak < 1500
                else "ok"
            )
            print(f"  saved {f.name}: {len(audio) / RATE:.1f} s, level {level}\n")
            with open(out / "prompts.jsonl", "a") as m:
                m.write(
                    json.dumps({"file": f.name, "text": text, "seconds": round(len(audio) / RATE, 2)}) + "\n"
                )
            if prompt is not None:
                break
    print(f"\n{n} recordings in {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
