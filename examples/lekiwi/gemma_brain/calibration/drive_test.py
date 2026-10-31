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

"""Does the base drive where the arm's frame says it does? The planner treats the arm's x axis
as the base's forward; the base's moves, measured by the front camera (placed in the arm's frame
by the camera calibration), say how far off that is, and why.

    .venv/bin/python calibration/drive_test.py [--leg 0.2] [--repeats 2]

From the calibration scene's start (the front camera seeing the boards; ~25 cm clear behind and
to the left): legs of LEG_M back, forward (back to the start), left, right (back again), each
measured before and after. The floor board ahead is never driven onto. Per leg: the actual
direction against the commanded one (the turn about base_turn_centre during it taken off) and
the distance driven against the commanded. Oct 31 (offline): 22 return drives of the calibration
runs, all one way, put it at 3.2 +/- 0.8 deg, which can't tell its two causes apart:

  the same angle in every direction    the arm's frame is turned on the base: the shoulder pan's
                                       zero (or the arm mounted turned); correcting the pan zero
                                       by it aligns the arm with the base either way
  a different angle per direction      the wheels (radius, geometry): the base itself

It applies nothing: it prints the pan correction for self_calibrate.py correct. The pan zero is
in the cameras' calibration (the front camera was placed through it): run calibrate_all.py again
after applying one.

Not here: the slowest speed the base still moves at (it would replace the hand-set
MIN_BASE_XY_SPEED, MIN_BASE_THETA_SPEED): the Pi clamps every base command to those, so a slower
one can't be asked for without a raw-speed mode on the Pi.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "web_server")]

LEG_M = 0.20  # convention: long enough that the camera's ~2 mm reads a direction to under 1 deg
# (dx, dy) in the base's frame: each pair goes out and back to the start, the floor board ahead
# never driven onto.
LEGS = (("back", (-1.0, 0.0)), ("forward", (1.0, 0.0)), ("left", (0.0, 1.0)), ("right", (0.0, -1.0)))
# The same angle every way when the directions' means agree within this (deg): the 22 return
# drives scattered +/-3.6 deg one by one; averaged over a direction's repeats, less.
SAME_ANGLE_DEG = 2.0


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def leg_result(before, after, cmd_base, centre) -> dict:
    """One leg from the front camera's measures before and after ((x, y, heading deg) of the
    arm's frame against the reference, in its axes): the drive alone (the turn about `centre`
    taken off), its direction against the command (turned by the heading at the start), and the
    distance ratio."""
    (x0, y0, t0), (x1, y1, t1) = before, after
    c = np.asarray(centre, float)
    drive = np.array([x1 - x0, y1 - y0]) - (np.eye(2) - _rot(t1 - t0)) @ c
    cmd = _rot(t0) @ np.asarray(cmd_base, float)
    angle = math.degrees(math.atan2(drive[1], drive[0]) - math.atan2(cmd[1], cmd[0]))
    angle = (angle + 180.0) % 360.0 - 180.0
    return {
        "angle_deg": round(angle, 2),
        "ratio": round(float(np.linalg.norm(drive) / np.linalg.norm(cmd)), 3),
        "turned_deg": round(t1 - t0, 2),
    }


def verdict(legs: list[dict]) -> dict:
    """The per-direction means, and what they say: one angle every way (the arm's frame turned
    on the base: a pan correction of minus it) or not (the wheels)."""
    by = {}
    for leg in legs:
        by.setdefault(leg["dir"], []).append(leg["angle_deg"])
    means = {d: round(float(np.mean(v)), 2) for d, v in by.items()}
    spread = max(means.values()) - min(means.values())
    mean = float(np.mean([leg["angle_deg"] for leg in legs]))
    same = spread <= SAME_ANGLE_DEG
    # The front camera was placed through the arm's zeros: a pan read d short of true (offset +d,
    # self_calibrate.py correct) turns everything it placed by -d, the base's moves included.
    return {
        "by_direction_deg": means,
        "spread_deg": round(spread, 2),
        "mean_deg": round(mean, 2),
        "same_every_way": same,
        "pan_offset_deg": round(-mean, 2) if same else None,
    }


def drive_legs(robot, measure, leg_m: float, repeats: int, centre) -> list[dict]:
    out = []
    for k in range(repeats):
        for name, (ux, uy) in LEGS:
            before = measure()
            robot.base(dx=ux * leg_m, dy=uy * leg_m)
            after = measure()
            r = {"dir": name, "repeat": k, **leg_result(before, after, (ux * leg_m, uy * leg_m), centre)}
            out.append(r)
            print(
                f"  {name:8s} {100 * leg_m:.0f} cm: drove {r['angle_deg']:+.1f} deg off its command, "
                f"{100 * r['ratio']:.0f}% of the distance, turned {r['turned_deg']:+.1f} deg"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", type=float, default=LEG_M, help="each leg's length (m)")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    import time

    from common.robot import Robot

    from calibration import boards
    from calibration.capture_cameras import _cameras, front_measure, return_to_start

    cams = _cameras()
    if not cams:
        raise SystemExit("no cameras in calib.json: run calibrate_all.py first")
    with Robot() as robot:
        ref = boards.detect(robot.frame("front", after=time.time())[0])
        if not ref:
            raise SystemExit(
                "the front camera sees no board: place the robot at the calibration scene's start"
            )
        measure = front_measure(robot, ref, cams)
        print(f"{args.repeats} x {len(LEGS)} legs of {100 * args.leg:.0f} cm:")
        try:
            legs = drive_legs(robot, measure, args.leg, args.repeats, cams["base_turn_centre"])
        finally:
            print("back to the start:")
            return_to_start(robot, measure)
    v = verdict(legs)
    print(f"\nby direction (deg off the command): {v['by_direction_deg']}; spread {v['spread_deg']} deg")
    if v["same_every_way"]:
        print(
            f"the same every way, {v['mean_deg']:+.2f} deg: the arm's frame is turned on the base.\n"
            f"  to align it: self_calibrate.py correct shoulder_pan={v['pan_offset_deg']:+.2f}, then calibrate_all.py again"
        )
    else:
        print("different per direction: the wheels, not the arm (no pan correction)")
    out = ROOT / "runs" / f"{datetime.datetime.now():%Y%m%dT%H%M%S}-drive_test.json"
    out.write_text(json.dumps({"legs": legs, "verdict": v, "leg_m": args.leg}, indent=1) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
