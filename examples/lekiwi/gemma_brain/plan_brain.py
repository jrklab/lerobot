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

"""LeKiwi brain, migration step 1: Gemma 4 26B writes the whole goal-level plan once, and
code runs it on the existing skills (skills.py, unchanged).

    .venv/bin/python plan_brain.py "Pick up the red block and drop it into the cup on your right"
    .venv/bin/python plan_brain.py --plan-only "..."     # plan and check it; nothing moves

Differences from brain.py: one planner call per task instead of one per skill; the plan is
grammar-limited and checked before anything moves; scan and approach are added by code, not
chosen by the model; a failed step is retried in code (pick: PICK_TRIES), then replanned (at
most MAX_REPLANS times). Perception inside the skills asks the llama.cpp server instead of
Ollama. Needs llama-server for 26B on --planner-url (and --vision-url).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

from arm import Arm, PlanError
from brain import PICK_TRIES
from gemma import LlamaCpp
from planner import Planner, Step, check_plan, objects_in
from robot import CANDIDATE_URLS, OperatorTookOver, Robot, RobotError
from skills import Log, Result, Skills

PLANNER_URL = "http://127.0.0.1:8081"  # 26B-A4B, 70 image tokens
MAX_REPLANS = 2
# What the existing skills can run: (verb, relation).
SUPPORTED = {("pick", None), ("place", "in"), ("place", "next_to")}


class PlanBrain:
    def __init__(self, sk: Skills | None, planner: Planner, log: Log, robot=None, holding: str | None = None):
        self.sk, self.planner, self.log, self.robot = sk, planner, log, robot
        self.holding = holding
        self.scanned: set[str] = set()

    def frames(self) -> tuple[bytes, bytes]:
        now = time.time()
        front, _ = self.robot.frame("front", after=now)
        wrist, _ = self.robot.frame("wrist", after=now)
        self.log.frame("plan_front", front)
        self.log.frame("plan_wrist", wrist)
        return front, wrist

    def make_plan(self, goal: str, front: bytes, wrist: bytes) -> tuple[list[Step] | None, str]:
        try:
            steps, a = self.planner.plan(goal, front, wrist, self.holding)
        except ValueError as e:
            self.log.ev("plan", ok=False, error=str(e))
            return None, f"the planner's answer was not a plan: {e}"
        self.log.ev("plan", steps=[str(s) for s in steps], ms=round(a.ms))
        print(f"plan ({a.ms / 1000:.1f} s):\n  " + "\n  ".join(map(str, steps)))
        problems = check_plan(steps, SUPPORTED, self.holding)
        if problems:
            self.log.ev("plan_rejected", problems=problems)
            return None, "plan not runnable: " + "; ".join(problems)
        return steps, ""

    # --- running one step on the existing skills ---

    def ensure_scanned(self, objects: list[str]) -> Result:
        new = [o for o in objects if o not in self.scanned and o != self.holding]
        if not new or not self.sk.base_ok:
            return Result(True)
        r = self.sk.scan(new[:4])
        self.log.ev("scan", objects=new, ok=r.ok, reason=r.reason)
        self.scanned.update(new)
        return r

    def pick(self, what: str) -> Result:
        reasons = []
        for attempt in range(1, PICK_TRIES + 1):
            r = self.sk.fetch(what)
            if r.ok:
                return r
            reasons.append(r.reason)
            self.log.ev("pick_retry", attempt=attempt, reason=r.reason)
            print(f"    pick attempt {attempt} failed: {r.reason}")
        return Result(False, f"failed {PICK_TRIES} times ({'; '.join(reasons)})")

    def place(self, rel: str, target: str) -> Result:
        r = self.sk.go_to(target)
        self.log.ev("step", name="go_to", ok=r.ok, reason=r.reason)
        if not r.ok:
            return Result(False, f"could not reach the {target}: {r.reason}")
        return self.sk.place_in(target) if rel == "in" else self.sk.place_next_to(target)

    def run_step(self, s: Step) -> Result:
        if s.verb == "pick":
            return self.pick(s.obj)
        return self.place(s.rel, s.target)

    # --- the task ---

    def run(self, goal: str) -> tuple[bool, str]:
        self.log.ev("goal", goal=goal, planner=self.planner.model.model, holding=self.holding)
        self.sk.holding = self.holding
        front, wrist = self.frames()
        steps, why = self.make_plan(goal, front, wrist)
        if steps is None:
            return False, why
        done: list[Step] = []
        replans = repeats = 0
        while steps:
            s = steps[0]
            r = self.ensure_scanned(objects_in(steps))
            if not r.ok:
                print(f"    scan: {r.reason}")
            print(f"\n>>> {s}")
            try:
                r = self.run_step(s)
            except (RobotError, PlanError) as e:
                if isinstance(e, OperatorTookOver):
                    raise
                r = Result(False, f"robot error: {e}")
            self.holding = getattr(self.sk, "holding", None)
            self.log.ev("step_result", step=str(s), ok=r.ok, reason=r.reason, holding=self.holding)
            print(f"<<< {'OK' if r.ok else 'FAILED'}: {r.reason or 'done'}")
            if r.ok:
                done.append(s)
                steps = steps[1:]
                continue
            if replans >= MAX_REPLANS:
                return False, f"{s} failed after {MAX_REPLANS} replans: {r.reason}"
            replans += 1
            try:
                self.sk.arm.reset()
            except (RobotError, PlanError) as e:
                if isinstance(e, OperatorTookOver):
                    raise
                return False, f"{s} failed ({r.reason}); then the arm couldn't fold back to rest to replan: {e}"
            front, wrist = self.frames()
            try:
                new, a = self.planner.replan(goal, done, s, r.reason, self.holding, front, wrist)
            except ValueError as e:
                return False, f"{s} failed ({r.reason}) and the replan was not a plan: {e}"
            # The same steps again: one more try is a retry; a second time won't go differently.
            same = new == steps
            if same and repeats:
                return False, f"{s} failed twice ({r.reason}) and the replan didn't change it"
            repeats += same
            steps = new
            self.log.ev("replan", n=replans, steps=[str(x) for x in steps], ms=round(a.ms))
            print(f"replan {replans} ({a.ms / 1000:.1f} s):\n  " + "\n  ".join(map(str, steps)))
            problems = check_plan(steps, SUPPORTED, self.holding)
            if problems:
                self.log.ev("plan_rejected", problems=problems)
                return False, f"{s} failed ({r.reason}); replan not runnable: " + "; ".join(problems)
        return True, "done: " + "; ".join(map(str, done))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goal", help="what the robot should do, in plain words")
    ap.add_argument("--planner-url", default=PLANNER_URL, help="llama-server running the planner (26B)")
    ap.add_argument("--vision-url", default=PLANNER_URL, help="llama-server for the skills' boxes and questions")
    ap.add_argument("--robot", default=None, help="LeKiwi web service URL (default: the first that answers of "
                    + ", ".join(CANDIDATE_URLS) + ")")
    ap.add_argument("--holding", default=None, help="what the gripper already holds at start, e.g. 'green block'")
    ap.add_argument("--plan-only", action="store_true",
                    help="read the two camera frames, plan and check the plan, then stop: nothing moves")
    ap.add_argument("--frames", nargs=2, metavar=("FRONT_JPG", "WRIST_JPG"), default=None,
                    help="with --plan-only: plan from these saved frames instead of the robot's cameras")
    ap.add_argument("--scan-sweep", type=float, default=None, metavar="DEG",
                    help="scan sweeps -DEG..+DEG in 30 deg steps instead of a full turn")
    ap.add_argument("--no-turn-scan", action="store_true", help="scan never turns the robot around")
    ap.add_argument("--no-base", action="store_true", help="lock the base: arm-only picks within reach")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out = Path(__file__).parent / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-plan")
    log = Log(out)
    planner = Planner(LlamaCpp(args.planner_url, "gemma4-26b"))

    if args.plan_only:
        # Read-only: camera frames over GET; auto mode is never enabled, so nothing can move.
        pb = PlanBrain(None, planner, log, None if args.frames else Robot(args.robot), args.holding)
        front, wrist = (Path(p).read_bytes() for p in args.frames) if args.frames else pb.frames()
        steps, why = pb.make_plan(args.goal, front, wrist)
        print("\nRUNNABLE" if steps else f"\nNOT RUNNABLE: {why}")
        print(f"log: {out}")
        return 0 if steps else 1

    robot = Robot(args.robot, base_enabled=not args.no_base)
    ok, summary = False, ""
    with robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, LlamaCpp(args.vision_url), log)
        sk.turn_scan = not args.no_turn_scan
        sk.scan_sweep_deg = args.scan_sweep
        try:
            ok, summary = PlanBrain(sk, planner, log, robot, args.holding).run(args.goal)
        except OperatorTookOver as e:
            summary = f"stopped: {e}"
        except OSError as e:
            summary = f"robot unreachable ({e}): the Pi may have rebooted or the link dropped"
        except KeyboardInterrupt:
            summary = "stopped with Ctrl-C"
            robot.stop()
        finally:
            if "Ctrl-C" not in summary and "operator" not in summary and "unreachable" not in summary:
                try:
                    arm.reset()
                except (RobotError, PlanError, OSError) as e:
                    log.ev("reset", ok=False, reason=str(e))
    log.ev("result", success=ok, summary=summary)
    print(f"\n{'SUCCESS' if ok else 'NOT DONE'}: {summary}\nlog: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
