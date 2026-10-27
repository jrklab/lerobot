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

# ruff: noqa: E402

"""Robot runs of the v2 skills, one at a time (each needs the owner's OK).

    .venv/bin/python run_v2.py land                 hover at the reach point, land, back up, fold
    .venv/bin/python run_v2.py pick ["red block"]   the pick chain (approach first)
    .venv/bin/python run_v2.py pick --no-drive      the brick already near the reach point

Logs and frames under runs/<time>-v2-<what>/. The arm folds at the end, holding or not.
"""

import argparse
import datetime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arm import Arm
from gemma import LlamaCpp
from robot import Robot
from skills import Log
from v2 import actions, chains
from v2.geometry import Geometry
from v2.skills import Skills2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["land", "pick"])
    ap.add_argument("obj", nargs="?", default="red block")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--main-url", default="http://127.0.0.1:8081")
    ap.add_argument("--fast-url", default="http://127.0.0.1:8082")
    args = ap.parse_args()
    log = Log(HERE / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"-v2-{args.what}"))
    robot = Robot()
    with robot:
        arm = Arm(robot)
        sk = Skills2(robot, arm, Geometry.load(), log, main=LlamaCpp(args.main_url),
                     fast=LlamaCpp(args.fast_url, "gemma4-e2b") if args.fast_url else None)
        try:
            if args.what == "land":
                r = chains.run([("hover", lambda: actions.hover(sk)), ("land", lambda: actions.land(sk))])
                print(f"land: {r}; jaw tips {1000 * sk.height():.1f} mm above the predicted floor")
            else:
                r = chains.pick(sk, args.obj, drive=not args.no_drive)
                print(f"pick: {r}")
            log.ev("result", ok=r.ok, reason=r.reason, data=r.data)
        finally:
            arm.reset()
    print("log", log.out)


if __name__ == "__main__":
    main()
