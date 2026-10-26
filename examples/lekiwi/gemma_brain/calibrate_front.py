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

"""Calibrate the front camera's geometry after it is moved (calib.json "front_camera"), with
the robot itself: one brick on the floor 0.6-1 m in front, roughly straight ahead, ~1 m of
free floor up to it.

    .venv/bin/python calibrate_front.py ["red Lego brick"]

1. Turn test: the brick's image column at heading 0, +8 and -8 deg -> FRONT_U_PER_DEG.
2. Drive test: straight at it in 8 cm steps (at most 1 m), logging where its box meets the
   floor, until that is at STOP_BOTTOM of the image.
3. Wrist line-up: the arm 10 cm over the reach point (ABOVE_XY), the base lines the brick up
   under it (skills.line_up_base) and the wrist camera measures what is left. Every drive
   step then has a known distance beyond the reach point (the base moves after it, by
   odometry, plus that residual), and the fit gives: the floor row filter and range model, the bottom rows at which
   approach() stops (CLOSE_IN_M beyond the reach point) and slows down, and the column of a
   brick straight ahead of the arm.

Moves the base (~1 m forward, small turns) and the arm (rest -> 10 cm over the floor -> rest).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

import numpy as np

import skills as S
from arm import Arm
from gemma import Gemma
from robot import Robot

TURN_DEG = 8.0
STEP_M = 0.08
MAX_DRIVE_M = 1.0
MID_BEYOND_M = 0.16  # approach slows to its short steps this far beyond the reach point
# Stop the drive once the brick's box reaches this row: closer and the base drives past the
# reach point (Oct 25 first run: the wrist camera then saw it only in its corner).
STOP_BOTTOM = 0.75


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="red Lego brick")
    args = ap.parse_args()
    out = Path(__file__).parent / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-front-calib")
    log = S.Log(out)
    S.FRONT_FLOOR_V = 0.0  # accept any box while calibrating; the fit sets the real filter
    robot = Robot()
    with robot:
        arm = Arm(robot)
        sk = S.Skills(robot, arm, Gemma(), log)
        sk.target = args.target
        arm.reset()

        def look(tag):
            got = sk.see_front(tag, args.target)
            if got is None:
                raise SystemExit(f"no {args.target} in the front camera ({tag})")
            return float(got[0][0]), float(got[0][1])

        # 1. Turn test.
        u0, _ = look("turn_0")
        robot.base(dtheta_deg=TURN_DEG)
        u_plus, _ = look("turn_plus")
        robot.base(dtheta_deg=-2 * TURN_DEG)
        u_minus, _ = look("turn_minus")
        robot.base(dtheta_deg=TURN_DEG)
        u_per_deg = (u_plus - u_minus) / (2 * TURN_DEG)  # left turn = +theta moves it right
        log.ev("turn_test", u0=u0, u_plus=u_plus, u_minus=u_minus, u_per_deg=u_per_deg)
        print(f"turn test: {u_per_deg:.4f} of the width per degree (was {S.FRONT_U_PER_DEG})")
        if not 0.005 < u_per_deg < 0.05:
            raise SystemExit("implausible turn response: check the brick is the only one boxed")

        # Face it (column 0.5: the target column isn't known yet; the drive must be straight at it).
        u, _ = look("face")
        robot.base(dtheta_deg=float(np.clip(-(u - 0.5) / u_per_deg, -20, 20)))

        # 2. Drive test.
        def odom_x():
            return float(robot.status()["odom"]["x"])  # forward travel; the drive is straight

        rows, x0 = [], odom_x()
        while odom_x() - x0 < MAX_DRIVE_M:
            got = sk.see_front(f"drive_{len(rows)}", args.target)
            if got is None:
                break
            rows.append({"x": odom_x(), "u": float(got[0][0]), "bottom": float(got[0][1])})
            log.ev("drive_row", **rows[-1])
            if rows[-1]["bottom"] >= STOP_BOTTOM:
                break
            robot.base(dx=STEP_M)
        if len(rows) < 4:
            raise SystemExit(f"only {len(rows)} drive rows: start further away")

        # 3. Wrist line-up at the reach point: the base puts the brick under it.
        sk.go_above(S.ABOVE_XY)
        lu = sk.line_up_base()
        x_end = odom_x()
        c = sk.see_block("wrist_check", check_present=False)
        arm.reset()
        if not lu.ok or c is None:
            raise SystemExit(f"the wrist line-up failed ({lu.reason or 'brick not seen'}): nothing saved")
        h = sk.height()
        d = np.linalg.solve(S.jacobian(h, sk.roll), S.grasp_pixel(h) - c)  # brick = reach point + d
        log.ev("wrist_check", center=c, dxy=d, height=h, x_end=x_end)

    # Fit: distance beyond the reach point for each row; range model d_cam = K / (bottom - vh)
    # with d_cam = beyond + offset (camera to reach point), by least squares over vh, offset.
    beyond = np.array([d[0] + x_end - r["x"] for r in rows])
    bottom = np.array([r["bottom"] for r in rows])

    def fit(idx):
        best = None
        for vh in np.arange(0.0, min(bottom[idx]) - 0.01, 0.005):
            x = 1.0 / (bottom[idx] - vh)
            for off in np.arange(-0.10, 0.60, 0.005):
                k = float(np.dot(x, beyond[idx] + off) / np.dot(x, x))
                err = float(np.sum((k * x - (beyond[idx] + off)) ** 2))
                if best is None or err < best[0]:
                    best = (err, vh, off, k)
        return best

    idx = np.arange(len(rows))
    best = fit(idx)
    # One wrong box (Oct 25: Gemma boxed a neighbour of the far, tiny brick) wrecks the fit:
    # drop the worst row once if it is far off the rest.
    _, vh, off, k = best
    res = np.abs(k / (bottom - vh) - (beyond + off))
    worst = int(np.argmax(res))
    if len(rows) > 5 and res[worst] > 3 * np.median(res):
        idx = np.delete(idx, worst)
        log.ev("fit_drop", row=worst, residual=float(res[worst]))
        print(f"dropped row {worst} (off by {1000 * res[worst]:.0f} mm)")
        best = fit(idx)
    _, vh, off, k = best

    def bottom_at(beyond_m):
        return float(vh + k / (beyond_m + off))

    near, mid = bottom_at(S.CLOSE_IN_M), bottom_at(MID_BEYOND_M)
    # Straight ahead of the arm: the last rows' column, corrected by the brick's measured side
    # offset seen from the arm base (left = +y = smaller u).
    bearing_deg = math.degrees(math.atan2(d[1], S.ABOVE_XY[0] + d[0]))
    u_target = float(np.mean([r["u"] for r in rows[-3:]]) + u_per_deg * bearing_deg)
    value = {
        "u_per_deg": round(u_per_deg, 4),
        "u_target": round(u_target, 3),
        "floor_v": round(vh + 0.02, 3),
        "range_k": round(k, 4),
        "range_horizon_v": round(vh, 3),
        "range_offset_m": round(off, 3),
        "near_bottom": round(near, 3),
        "mid_bottom": round(mid, 3),
    }
    rms = math.sqrt(best[0] / len(idx))
    if rms > 0.03 or not 0.0 < u_target < 1.0:
        raise SystemExit(f"fit too poor to save (rms {1000 * rms:.0f} mm, u_target {u_target:.2f}): {value}")
    print("rows (beyond reach m, bottom):", [(round(b, 3), round(v, 3)) for b, v in zip(beyond, bottom)])
    print(f"fit rms {1000 * rms:.0f} mm:", value)
    if near > 0.99:
        print("WARNING: the stop row is at the very bottom of the view: approach can't see that close")
    calib_path = Path(__file__).with_name("calib.json")
    calib = json.loads(calib_path.read_text())
    calib["front_camera"] = {
        "value": value,
        "meaning": "front-camera geometry for scan/approach (skills.FRONT_*, RANGE_*)",
        "method": "calibrate_front.py: turn test, straight drive at a brick, wrist check at the reach point",
        "date": datetime.date.today().isoformat(),
        "log": str(out.relative_to(Path(__file__).parent)),
        "fit_rms_m": round(rms, 4),
    }
    calib_path.write_text(json.dumps(calib, indent=2) + "\n")
    print("saved to calib.json; log", out)


if __name__ == "__main__":
    main()
