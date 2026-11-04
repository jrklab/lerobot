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

    .venv/bin/python calibration/drive_test.py [--leg 0.2] [--repeats 3]

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
# The same angle every way when the directions' medians agree within this (deg), or within what
# the legs' own scatter makes of four medians when that is more (same_angle_tolerance): the 22
# return drives scattered +/-3.6 deg one by one; Nov 4, after the pan correction, the legs scattered
# ~1.3 deg and their medians, all near zero, spread 2.7.
SAME_ANGLE_DEG = 2.0
# Statistics, not tuning: the range of 4 normal samples (one median per direction) stays under this
# many standard deviations 95% of the time; a median of n legs varies by MEDIAN_SE_FACTOR / sqrt(n)
# of their standard deviation; a normal sample's median absolute deviation is 1 / MAD_TO_SD of it.
RANGE_95_OF_4 = 3.63
MEDIAN_SE_FACTOR = 1.2533
MAD_TO_SD = 1.4826


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


def same_angle_tolerance(by: dict[str, list[float]], medians: dict[str, float]) -> float:
    """How far the directions' medians can spread with one angle every way: SAME_ANGLE_DEG, or
    the 95% range of four medians given the legs' scatter about their direction's median if that
    is more. The scatter is robust (the median deviation, the zero of each direction's own middle
    leg left out), so one bad leg doesn't widen it."""
    devs = [abs(a - medians[d]) for d, v in by.items() for a in v]
    devs = [x for x in devs if x > 1e-9]
    if not devs:
        return SAME_ANGLE_DEG
    sd = MAD_TO_SD * float(np.median(devs))
    n = min(len(v) for v in by.values())
    return max(SAME_ANGLE_DEG, RANGE_95_OF_4 * MEDIAN_SE_FACTOR * sd / math.sqrt(n))


def verdict(legs: list[dict]) -> dict:
    """The per-direction medians, and what they say: one angle every way (the arm's frame turned
    on the base: a pan correction of it) or not (the wheels). Medians, not means: Nov 3's
    first leg back read +8.3 deg against +1.7 to +3.2 for the other seven, and its mean alone made
    "back" differ; with three or more repeats a direction's median ignores one such leg. The legs
    off their direction's median by more than SAME_ANGLE_DEG are listed as outliers."""
    by = {}
    for leg in legs:
        by.setdefault(leg["dir"], []).append(leg["angle_deg"])
    medians = {d: round(float(np.median(v)), 2) for d, v in by.items()}
    spread = max(medians.values()) - min(medians.values())
    mean = float(np.mean(list(medians.values())))
    tolerance = same_angle_tolerance(by, medians)
    same = spread <= tolerance
    outliers = [
        {"dir": leg["dir"], "repeat": leg["repeat"], "angle_deg": leg["angle_deg"]}
        for leg in legs
        if abs(leg["angle_deg"] - medians[leg["dir"]]) > SAME_ANGLE_DEG
    ]
    # The front camera was placed through the arm's zeros, so a pan zero off turns the base's
    # moves as it sees them. Measured, not derived (Nov 3): correcting the pan by -2.43 deg took
    # the drives from +2.4 to +5.3 deg, so the correction is plus the angle.
    return {
        "by_direction_deg": medians,
        "spread_deg": round(spread, 2),
        "tolerance_deg": round(tolerance, 2),
        "mean_deg": round(mean, 2),
        "same_every_way": same,
        "outliers": outliers,
        "repeats": min(len(v) for v in by.values()),
        "pan_offset_deg": round(mean, 2) if same else None,
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


REPEATS = 3  # 3 or more: a direction's median ignores one bad leg


def run(robot, cams: dict, leg_m: float = LEG_M, repeats: int = REPEATS) -> dict:
    """The legs from where the base stands (the front camera seeing a board), back to it after;
    the legs and their verdict. Applies nothing."""
    import time

    from calibration import boards
    from calibration.capture_cameras import front_measure, return_to_start

    ref = boards.detect(robot.frame("front", after=time.time())[0])
    if not ref:
        raise RuntimeError("the front camera sees no board: place the robot at the calibration scene's start")
    measure = front_measure(robot, ref, cams)
    print(f"{repeats} x {len(LEGS)} legs of {100 * leg_m:.0f} cm:")
    try:
        legs = drive_legs(robot, measure, leg_m, repeats, cams["base_turn_centre"])
    finally:
        print("back to the start:")
        return_to_start(robot, measure)
    return {"legs": legs, "verdict": verdict(legs), "leg_m": leg_m}


def report(v: dict) -> None:
    print(
        f"\nby direction (median deg off the command): {v['by_direction_deg']}; spread {v['spread_deg']} deg"
        f" (the same within {v['tolerance_deg']})"
    )
    for o in v["outliers"]:
        print(
            f"  outlier: {o['dir']} (repeat {o['repeat']}) {o['angle_deg']:+.1f} deg, off its direction's median"
        )
    if v["repeats"] < 3:
        print("  (under 3 repeats: a direction's median is its mean, one bad leg can sway it)")
    if v["same_every_way"]:
        print(
            f"the same every way, {v['mean_deg']:+.2f} deg: the arm's frame is turned on the base.\n"
            f"  to align it: self_calibrate.py correct shoulder_pan={v['pan_offset_deg']:+.2f}, then calibrate_all.py again"
        )
    else:
        print("different per direction: the wheels, not the arm (no pan correction)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", type=float, default=LEG_M, help="each leg's length (m)")
    ap.add_argument("--repeats", type=int, default=REPEATS, help="3 or more: a direction's median ignores one bad leg")
    args = ap.parse_args()

    from common.robot import Robot

    from calibration.capture_cameras import _cameras

    cams = _cameras()
    if not cams:
        raise SystemExit("no cameras in calib.json: run calibrate_all.py first")
    with Robot() as robot:
        try:
            res = run(robot, cams, args.leg, args.repeats)
        except RuntimeError as e:
            raise SystemExit(str(e)) from None
    report(res["verdict"])
    out = ROOT / "runs" / f"{datetime.datetime.now():%Y%m%dT%H%M%S}-drive_test.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
