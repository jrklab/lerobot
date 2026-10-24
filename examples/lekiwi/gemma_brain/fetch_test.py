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

"""Drive to the brick seen in the front camera and pick it up (skills.Skills.fetch), then
reset. Logs and frames under runs/<timestamp>-fetch/.

    .venv/bin/python fetch_test.py [--approach-only]
"""

import argparse
import datetime
import logging
from pathlib import Path

from arm import Arm, PlanError
from gemma import Gemma
from robot import Robot, RobotError
from skills import Log, Skills


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach-only", action="store_true", help="stop once the brick is in reach")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = Path(__file__).parent / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-fetch")
    log = Log(out)
    robot = Robot()
    with robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, Gemma(), log)
        try:
            r = sk.approach() if args.approach_only else sk.fetch()
            log.ev("result", ok=r.ok, reason=r.reason, data=r.data)
        except (RobotError, PlanError) as e:
            log.ev("result", ok=False, reason=f"error: {e}")
        finally:
            try:
                arm.reset()
                log.ev("reset", ok=True)
            except (RobotError, PlanError) as e:
                log.ev("reset", ok=False, reason=str(e))
    print("saved", out)


if __name__ == "__main__":
    main()
