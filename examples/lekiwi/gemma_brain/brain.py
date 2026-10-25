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

"""Gemma 4 drives LeKiwi toward a goal given in plain words, by calling skills as tools.

    .venv/bin/python brain.py "Pick up the red block and drop it into the cup on your right"

Gemma (Ollama, gemma4:e4b by default) gets the goal and the skill list; it calls one skill
at a time, reads its result (ok, a sentence, numbers), and decides the next call, until it
calls finish(). The skills (skills.py) do all perception and motion: Gemma plans and
recovers, it never moves a motor itself. Every turn, frame and number goes to
runs/<timestamp>-brain/.

Stop at any time: Ctrl-C here (the robot stops; the arm holds where it is), any gamepad
input, or the gamepad's e-stop (RB). Requires the LeKiwi web service on the Pi and Ollama
running on this Mac.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

from arm import Arm, PlanError
from gemma import DEFAULT_MODEL, Gemma
from robot import CANDIDATE_URLS, OperatorTookOver, Robot, RobotError
from skills import Log, Result, Skills

MAX_CALLS = 16

SYSTEM = """You control LeKiwi, a small wheeled robot with one arm and a gripper, in a room. You reach the goal you are given by calling the robot's skills, one call at a time. Each skill reports back whether it worked, what happened, and some numbers; use that to choose the next call.

What the skills can do:
- scan(objects): turn a full circle and look for every listed object on the floor; it reports where each was seen. Call it first, with every object the goal mentions, and again if the robot has lost track of an object.
- approach(object): turn toward the object found by the scan, drive until it is within arm's reach, and check up close with the wrist camera that it really is that object (if not, it tries the next place the scan saw it).
- pick(object): go to a small object on the floor (like approach) and pick it up with the gripper. Only one object can be held at a time.
- place_in(container): drop the held object into an open container on the floor that is within reach (approach the container first).
- reset_arm(): fold the arm back to its rest pose. The skills already do this when they need to; use it only to recover from a failure.
- finish(success, summary): stop. Call it when the goal is done, or when it cannot be done.

Rules:
- Describe objects the way the goal does, with their color, e.g. "red Lego brick", "paper cup".
- If a skill fails, read why. You may retry it once, or call scan again if the object was not found. Do not repeat the same failing call more than twice.
- Call exactly one skill per reply. Do not answer with plain text: always call a skill."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan",
            "description": "Turn a full circle and find every listed object on the floor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "the objects to look for, with their colors, e.g. ['red Lego brick', 'paper cup']",
                    }
                },
                "required": ["objects"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approach",
            "description": "Turn to the object the scan found, drive until it is in reach, and confirm it up close.",
            "parameters": {
                "type": "object",
                "properties": {"object": {"type": "string"}},
                "required": ["object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Drive to a small object on the floor and pick it up.",
            "parameters": {
                "type": "object",
                "properties": {"object": {"type": "string"}},
                "required": ["object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_in",
            "description": "Drop the held object into an open container on the floor within reach.",
            "parameters": {
                "type": "object",
                "properties": {"container": {"type": "string"}},
                "required": ["container"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_arm",
            "description": "Fold the arm back to its rest pose.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Stop: the goal is done, or it cannot be done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "summary": {"type": "string", "description": "one sentence on what happened"},
                },
                "required": ["success", "summary"],
            },
        },
    },
]


def _text(v, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"'{name}' must be a non-empty string")
    return v.strip().strip(".").lower()


class Brain:
    def __init__(self, sk: Skills, gemma: Gemma, log: Log, think: bool = False):
        self.sk, self.gemma, self.log, self.think = sk, gemma, log, think

    def call(self, name: str, args: dict) -> Result:
        """Run one tool call. Arguments are checked here: the tool schema isn't enforced by
        the model server (enums and types can come back wrong)."""
        sk = self.sk
        if name == "scan":
            objs = args.get("objects")
            if isinstance(objs, str):
                objs = [o for o in objs.split(",")]
            if not isinstance(objs, list) or not objs:
                raise ValueError("'objects' must be a non-empty list of object names")
            return sk.scan([_text(o, "objects") for o in objs][:4])
        if name == "approach":
            return sk.go_to(_text(args.get("object"), "object"))
        if name == "pick":
            if getattr(sk, "holding", None):
                return Result(False, f"already holding the {sk.holding}; place it first")
            what = _text(args.get("object"), "object")
            r = sk.fetch(what)
            if not r.ok:
                r.reason += f". The {what} was not picked up and is still on the floor: call pick again to retry."
            return r
        if name == "place_in":
            if not getattr(sk, "holding", None):
                return Result(False, "not holding anything to place: pick the object up first")
            return sk.place_in(_text(args.get("container"), "container"))
        if name == "reset_arm":
            sk.arm.reset()
            return Result(True, "arm at rest")
        raise ValueError(f"unknown skill {name!r}")

    def run(self, goal: str) -> tuple[bool, str]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Goal: {goal}"}]
        self.log.ev("goal", goal=goal, model=self.gemma.model)
        for turn in range(MAX_CALLS):
            t = time.monotonic()
            r = self.gemma.chat(messages, tools=TOOLS, think=self.think, max_tokens=400)
            msg = r["message"]
            calls = msg.get("tool_calls") or []
            self.log.ev(
                "gemma",
                turn=turn,
                ms=round((time.monotonic() - t) * 1000),
                thinking=(msg.get("thinking") or "")[:400],
                content=(msg.get("content") or "")[:200],
                calls=[c["function"] for c in calls],
            )
            messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})
            if not calls:
                messages.append({"role": "user", "content": "Reply by calling exactly one skill."})
                continue
            fn = calls[0]["function"]
            name, args = fn.get("name"), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if name == "finish":
                ok, summary = bool(args.get("success")), str(args.get("summary", ""))
                self.log.ev("finish", success=ok, summary=summary)
                return ok, summary
            print(f"\n>>> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            try:
                res = self.call(name, args)
            except ValueError as e:
                res = Result(False, f"bad call: {e}")
            except (RobotError, PlanError) as e:
                if isinstance(e, OperatorTookOver):
                    raise
                res = Result(False, f"robot error: {e}")
            out = {
                "ok": res.ok,
                "message": res.reason or ("done" if res.ok else "failed"),
                # State every turn: small models plan far better when told what holds now.
                "gripper_holds": getattr(self.sk, "holding", None) or "nothing",
                "data": res.data,
            }
            if len(calls) > 1:
                out["note"] = "only the first skill call was run; call one skill per reply"
            self.log.ev("tool_result", name=name, args=args, **out)
            print(f"<<< {'OK' if res.ok else 'FAILED'}: {out['message']}")
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(out, default=str)})
        return False, f"stopped after {MAX_CALLS} skill calls without finishing"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goal", help="what the robot should do, in plain words")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--robot", default=None, help="LeKiwi web service URL (default: the first that answers of "
                    + ", ".join(CANDIDATE_URLS) + ")")
    ap.add_argument(
        "--no-base",
        action="store_true",
        help="lock the base (every wheel command refused): scan and approach don't drive; pick works on "
        "an object within the arm's reach in front",
    )
    ap.add_argument(
        "--no-think",
        action="store_true",
        help="skip Gemma's reasoning before each call (faster, but it recovered from a failed pick "
        "far less directly in dry runs)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out = Path(__file__).parent / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-brain")
    log = Log(out)
    robot, gemma = Robot(args.robot, base_enabled=not args.no_base), Gemma(args.model)
    ok, summary = False, ""
    with robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, gemma, log)
        # No guess about holding something at start: a closed, empty gripper can read inside
        # the range a held brick does (Oct 24: 3.0-4 empty vs 4.7-6.8 held), and a wrong
        # "already holding" made Gemma refuse to pick.
        try:
            goal = args.goal
            if args.no_base:
                goal += ("\n(The robot's base is locked: it cannot drive or turn. Objects must already be in "
                         "front of it, within the arm's reach. Don't scan; call pick directly.)")
            ok, summary = Brain(sk, gemma, log, think=not args.no_think).run(goal)
        except OperatorTookOver as e:
            summary = f"stopped: {e}"
        except KeyboardInterrupt:
            summary = "stopped with Ctrl-C"
            robot.stop()
        finally:
            if "Ctrl-C" not in summary and "operator" not in summary:
                try:
                    arm.reset()
                except (RobotError, PlanError) as e:
                    log.ev("reset", ok=False, reason=str(e))
    log.ev("result", success=ok, summary=summary)
    print(f"\n{'SUCCESS' if ok else 'NOT DONE'}: {summary}\nlog: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
