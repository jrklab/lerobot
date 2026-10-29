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

"""LeKiwi brain: say a task, the robot does it. Gemma 4 26B plans the task once from the two
cameras; code runs the steps on the skills (brain/task_loop.py, the task loop).

    .venv/bin/python plan_brain.py "put the red block in the paper cup"
    .venv/bin/python plan_brain.py --plan-only "..."     # plan and check it; nothing moves

Needs llama-server for 26B on --planner-url (also the skills' boxes, --vision-url) and E2B on
--fast-url for repeat looks at an object already found. v1's skills were retired on Oct 28
(tag lekiwi-v1-final).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path

from brain.geometry import Geometry
from brain.primitives import Skills
from brain.task_loop import PlanBrain
from common.arm import Arm
from common.gemma import LlamaCpp
from common.planner import Planner
from common.robot import CANDIDATE_URLS, OperatorTookOver, Robot, RobotError
from common.runlog import Log

PLANNER_URL = "http://127.0.0.1:8081"  # 26B-A4B, 70 image tokens
FAST_URL = "http://127.0.0.1:8082"  # E2B Q4_K_M, 70 image tokens: repeat looks inside the skills


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("goal", help="what the robot should do, in plain words")
    ap.add_argument("--planner-url", default=PLANNER_URL, help="llama-server running the planner (26B)")
    ap.add_argument("--vision-url", default=PLANNER_URL, help="llama-server for the skills' boxes and questions")
    ap.add_argument("--fast-url", default=FAST_URL, help="llama-server for repeat looks at an object already found (E2B)")
    ap.add_argument("--no-fast", action="store_true", help="every look asks --vision-url")
    ap.add_argument("--robot", default=None, help="LeKiwi web service URL (default: the first that answers of "
                    + ", ".join(CANDIDATE_URLS) + ")")
    ap.add_argument("--holding", default=None, help="what the gripper already holds at start, e.g. 'green block'")
    ap.add_argument("--plan-only", action="store_true",
                    help="read the two camera frames, plan and check the plan, then stop: nothing moves")
    ap.add_argument("--frames", nargs=2, metavar=("FRONT_JPG", "WRIST_JPG"), default=None,
                    help="with --plan-only: plan from these saved frames instead of the robot's cameras")
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
        if sys.platform == "darwin":  # no idle sleep mid-task (Oct 27: the Pi stopped on lost heartbeats)
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        sk = Skills(robot, arm, Geometry.load(), log, main=LlamaCpp(args.vision_url),
                     fast=None if args.no_fast else LlamaCpp(args.fast_url, "gemma4-e2b"))
        pb = PlanBrain(sk, planner, log, robot, args.holding)
        try:
            ok, summary = pb.run(args.goal)
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
                    why = pb.fold()
                except (OSError, RobotError) as e:  # incl. the operator taking over meanwhile
                    why = str(e)
                if why:
                    log.ev("reset", ok=False, reason=why)
                    print(f"\nThe arm stayed where it is ({why}): fold it with the gamepad's B button.")
    log.ev("result", success=ok, summary=summary)
    print(f"\n{'SUCCESS' if ok else 'NOT DONE'}: {summary}\nlog: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
