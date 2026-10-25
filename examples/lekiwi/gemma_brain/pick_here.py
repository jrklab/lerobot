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

"""Pick a brick within the arm's reach in front of the robot, with the arm only: the base is
locked (every wheel command is refused in the client), e.g. while the robot is tethered.

    .venv/bin/python pick_here.py ["green Lego brick"]

Prints how long each phase took. The brick should lie about 15-25 cm in front of the arm's
base, roughly centred; the arm ends at rest, holding it.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import time
from pathlib import Path

from arm import Arm, PlanError
from gemma import Gemma
from robot import Robot, RobotError
from skills import ABOVE_XY, Log, Skills


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="green Lego brick")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = Path(__file__).parent / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-pick-here")
    log = Log(out)
    robot = Robot(base_enabled=False)
    marks = []

    # Time each logged step as it ends.
    orig_ev = log.ev

    def ev(kind, **kw):
        if kind == "step":
            marks.append((kw.get("name"), time.time(), kw.get("ok")))
        orig_ev(kind, **kw)

    log.ev = ev
    t0 = time.time()
    with robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, Gemma(), log)
        sk.target = args.target
        try:
            r = sk.grasp(ABOVE_XY)
            log.ev("result", ok=r.ok, reason=r.reason)
        except (RobotError, PlanError, OSError) as e:  # OSError: e.g. Ollama not running
            r = None
            log.ev("result", ok=False, reason=f"error: {e}")
        finally:
            try:
                arm.reset()
                marks.append(("reset", time.time(), True))
            except (RobotError, PlanError) as e:
                log.ev("reset", ok=False, reason=str(e))
    prev = t0
    print("\nphase timings:")
    for name, t, ok in marks:
        print(f"  {name:10s} {t - prev:6.1f} s  {'ok' if ok else 'FAILED'}")
        prev = t
    print(f"  {'total':10s} {time.time() - t0:6.1f} s   ->  {'HOLDING' if r is not None and r.ok else 'not picked'}")
    print("saved", out)


if __name__ == "__main__":
    main()
