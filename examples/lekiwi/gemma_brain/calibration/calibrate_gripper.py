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

"""Measure the gripper's opening in millimetres against its angle (calib.json "gripper_opening"),
by closing it on objects of known width: nothing (0 mm), a 2x4 brick across its short side
(15.8 mm) and its long side (31.8 mm), a 2x6 brick's long side (47.8 mm), twice each.
Interactive: run it in a terminal.

    .venv/bin/python calibration/calibrate_gripper.py [--objects nothing 2x4-short ...] [--add] [--write]

--add keeps the points already in calib.json and adds these (e.g. --objects 2x6-long --add).

The arm goes to the hover over the reach point (gripper vertical, 10 cm up) and stays there.
For each object the gripper opens fully; place the object between the jaws, at the jaw tips
where a pick holds it, held lightly by its ends, and press ENTER: the gripper closes at the
pick's closing speed until it stalls on the object, the same stall a pick ends on, and the
angle it stopped at is recorded. The opening is interpolated straight between the measured
points (each width's mean angle), not fitted with one line: the jaw swings on a pivot and
opens less per degree as it opens wider (Oct 29: ~1.9 mm/deg up to 15.8 mm, ~1.3 to 31.8; a
line was 1.9 mm off). The arm folds at the end.

Why: the gripper's angle depends on where its closed end was recorded at calibration (the
Oct 29 recalibration moved a brick's reading from 8.9 to 9.8-10.0 deg); its opening in
millimetres doesn't, and it compares directly with an object's width.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from brain import actions, params as prm
from brain.geometry import Geometry
from brain.primitives import Skills
from common.arm import Arm
from common.robot import Robot

CALIB = ROOT / "data" / "calib.json"
# Width (mm) of what the jaws close on: nothing, LEGO bricks (8 mm per stud less the 0.2 mm play).
OBJECTS = {
    "nothing": ("nothing between the jaws", 0.0),
    "2x4-short": ("the 2x4 brick across its short side (2 studs)", 15.8),
    "2x4-long": ("the 2x4 brick across its long side (4 studs)", 31.8),
    "2x6-long": ("the 2x6 brick across its long side (6 studs)", 47.8),
}


def curve(points: list[dict]) -> list[list[float]]:
    """[[deg, mm], ...]: each width's mean angle, by width."""
    widths = sorted({p["mm"] for p in points})
    return [[round(float(np.mean([p["deg"] for p in points if p["mm"] == w])), 2), w] for w in widths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--objects", nargs="+", choices=list(OBJECTS), default=list(OBJECTS))
    ap.add_argument("--add", action="store_true", help="keep the points already in calib.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    old = json.loads(CALIB.read_text()).get("gripper_opening", {}).get("points", []) if args.add else []
    points = []
    with Robot() as robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, Geometry.load())
        try:
            r = actions.hover(sk)
            if not r.ok:
                raise SystemExit(f"hover failed: {r.reason}")
            wide = arm.calib.limits("arm_gripper")[1]
            for k in range(args.repeats):
                for what, mm in (OBJECTS[o] for o in args.objects):
                    robot.gripper(wide)
                    input(f"\n[{k + 1}/{args.repeats}] Place {what} ({mm:.1f} mm) at the jaw tips, then press ENTER ")
                    res = robot.gripper(0.0, speed=prm.CLOSE_SPEED.value).get("result", {})
                    deg = float(robot.status()["joints"]["arm_gripper"])
                    points.append({"mm": mm, "deg": round(deg, 2), "stalled": bool(res.get("stalled")), "what": what})
                    print(f"    closed at {deg:.2f} deg (stalled: {res.get('stalled')})")
            robot.gripper(wide)
            input("\nTake the brick out, then press ENTER to fold the arm ")
        finally:
            arm.reset()

    points = old + points
    c = curve(points)
    print("\npoints:", [(p["mm"], p["deg"]) for p in points])
    spread = {w: round(max(p["deg"] for p in points if p["mm"] == w) - min(p["deg"] for p in points if p["mm"] == w), 2)
              for _, w in c}
    print("curve (deg -> mm):", c, "| repeat spread (deg):", spread)
    for (d0, w0), (d1, w1) in zip(c, c[1:], strict=False):
        print(f"  {w0:4.1f} -> {w1:4.1f} mm over {d1 - d0:5.2f} deg: {(w1 - w0) / (d1 - d0):.2f} mm/deg")
    if any(b[0] <= a[0] for a, b in zip(c, c[1:], strict=False)):
        raise SystemExit("the angle doesn't grow with the width: a slipped or misplaced object? nothing written")
    if args.write:
        calib = json.loads(CALIB.read_text())
        calib["gripper_opening"] = {
            "value": {"curve": c},
            "meaning": "jaw opening (mm, at the tips) against the gripper angle (deg open from its recorded closed end): "
                       "straight between these [deg, mm] points; beyond the last, its last segment's slope",
            "method": "calibrate_gripper.py: closed at the pick's speed until the stall on nothing and LEGO bricks of known width",
            "date": datetime.date.today().isoformat(),
            "points": points,
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")


if __name__ == "__main__":
    main()
