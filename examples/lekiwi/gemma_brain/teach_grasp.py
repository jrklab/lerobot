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

"""The owner's midpoint teach (v2's grasp target, calib.json "wrist_grasp_target"): where the
brick's box sits in the wrist image, just before closing, when it is held by its middle.

    .venv/bin/python teach_grasp.py ["red block"] [--write] [--close]

1. The open, vertical gripper goes over the reach point at 10 cm, then lands at the grasp
   height (jaw tips GRASP_LIFT_M above the floor, without pressing).
2. The owner puts the brick between the open jaws by hand where it should be held: its middle
   on the jaws, its side close to the fixed jaw with a little room. Press Enter.
3. The wrist frame is boxed (main model, snapped to the brick's colour); the box's right edge
   and middle are the target, with the measured height. --close: the jaws close and lift as a
   check that it holds. The arm folds at the end. Frames are saved under runs/.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arm import Arm  # noqa: E402
from gemma import LlamaCpp  # noqa: E402
from robot import Robot  # noqa: E402
from skills import Log  # noqa: E402
from v2 import actions, params as prm  # noqa: E402
from v2.geometry import Geometry  # noqa: E402
from v2.skills import Skills2  # noqa: E402

CALIB = HERE / "calib.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="red block")
    ap.add_argument("--vision-url", default="http://127.0.0.1:8081")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--close", action="store_true", help="close and lift afterwards, to check it holds")
    ap.add_argument("--go-file", type=Path, default=None,
                    help="wait for this file to appear instead of Enter (when run without a terminal)")
    args = ap.parse_args()
    out = HERE / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-teach-grasp")
    log = Log(out)
    robot = Robot()
    with robot:
        arm = Arm(robot)
        sk = Skills2(robot, arm, Geometry.load(), log, main=LlamaCpp(args.vision_url))
        try:
            xy = (prm.REACH_POINT_X_M.value, 0.0)
            for r in (sk.move(xy=xy, height=prm.HOVER_HEIGHT_M.value, roll=0.0), sk.grip("open"), actions.land(sk)):
                if not r.ok:
                    print("stopped:", r.reason)
                    return
            ask = (f"\nPut the {args.what} between the open jaws where it should be held (its middle on the jaws,\n"
                   "its side close to the fixed jaw with a little room), then ")
            if args.go_file is None:
                input(ask + "press Enter... ")
            else:
                print(ask + f"create {args.go_file} ...", flush=True)
                deadline = time.time() + 600
                while not args.go_file.exists():
                    if time.time() > deadline:
                        print("no go after 10 minutes; nothing recorded")
                        return
                    time.sleep(0.5)
            det = sk.find(args.what, "wrist")
            h = sk.height()
            if det is None:
                print(f"no {args.what} boxed in the wrist frame; nothing written")
                return
            target = [det.box[2], float(det.center[1])]
            print(f"box {det.box}  target (right edge u, middle v) {[round(v, 4) for v in target]}  at {1000 * h:.1f} mm")
            log.ev("teach_grasp", box=det.box, target=target, height_m=h)
            if args.close:
                c = sk.grip("close")
                lr = actions.lift(sk) if c.ok else c
                print("close:", c.data.get("pos"), "held after the lift:", lr.ok)
            if args.write:
                calib = json.loads(CALIB.read_text())
                calib["wrist_grasp_target"] = {
                    "value": [round(v, 4) for v in target],
                    "height_m": round(h, 4),
                    "unit": "wrist image fraction (u right, v down) of the brick's box: its right edge and middle",
                    "meaning": "where the brick sits just before closing when held by its middle (owner's teach)",
                    "method": f"teach_grasp.py: owner placed the {args.what} between the open jaws at the grasp height",
                    "box": det.box,
                    "date": datetime.date.today().isoformat(),
                    "log": str(out.relative_to(HERE)),
                }
                CALIB.write_text(json.dumps(calib, indent=2) + "\n")
                print(f"wrote {CALIB}")
        finally:
            arm.reset()


if __name__ == "__main__":
    main()
