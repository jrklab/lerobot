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

"""Calibrate the front camera's geometry after it is moved (calib.json "front_camera"), with
the robot itself: one brick on the floor 0.6-1 m in front, roughly straight ahead, ~1 m of
free floor up to it.

    .venv/bin/python calibration/calibrate_front.py ["red Lego brick"] [--vision-url http://127.0.0.1:8081]

1. Turn test: the brick's image column at heading 0, +8 and -8 deg -> FRONT_U_PER_DEG.
2. Drive test: straight at it in 8 cm steps (at most 1 m), logging where its box meets the
   floor, until that is at STOP_BOTTOM of the image.
3. Wrist line-up: the arm hovering over the reach point (v2 actions.hover), the base lines the
   brick up under it (actions.line_up) and the wrist camera locates what is left
   (geometry.locate_from_one_view). Every drive step then has a known distance beyond the
   reach point (the base moves after it, by odometry, plus that residual), and the fit gives
   the range model, the bottom row of an object at the reach point (near_bottom: where v2's
   FrontApproachGoal parks it) and the column of a brick straight ahead of the arm.

v1 wrote near_bottom 6 cm beyond the reach point (its CLOSE_IN_M) and v2 reads it as at the
reach point, so the calibration in calib.json (Oct 27) parks objects ~6 cm short and the
line-up drives the rest; a new run moves the approach's stop to the reach point itself.
The front boxes are v2 find()'s, which refuses boxes ending above the current calibration's
horizon: recalibrate from a brick below it.

Moves the base (~1 m forward, small turns) and the arm (rest -> 10 cm over the floor -> rest).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from brain import actions, params as prm
from brain.geometry import Geometry, locate_from_one_view
from brain.primitives import Skills
from common.arm import Arm
from common.gemma import LlamaCpp
from common.robot import Robot
from common.runlog import Log

TURN_DEG = 8.0
STEP_M = 0.08
MAX_DRIVE_M = 1.0
# Stop the drive once the brick's box reaches this row: closer and the base drives past the
# reach point (Oct 25 first run: the wrist camera then saw it only in its corner).
STOP_BOTTOM = 0.75


def main() -> None:
    raise SystemExit(
        "superseded (Oct 30): the camera calibration (capture_cameras.py + solve_cameras.py) gives the front "
        "camera's geometry; only this script's near_bottom (a brick at the reach point) is still read, from calib.json"
    )
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="red Lego brick")
    ap.add_argument("--vision-url", default="http://127.0.0.1:8081", help="llama-server for the boxes (26B)")
    args = ap.parse_args()
    out = ROOT / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-front-calib")
    log = Log(out)
    robot = Robot()
    with robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, Geometry.load(), log, main=LlamaCpp(args.vision_url))
        was = sk.geom.front["u_per_deg"]
        arm.reset()

        def see(tag):
            det = sk.find(args.target, "front")
            log.ev("front_look", tag=tag, box=None if det is None else det.box)
            return det

        def look(tag):
            det = see(tag)
            if det is None:
                raise SystemExit(f"no {args.target} in the front camera ({tag})")
            return float(det.center[0]), float(det.box[3])

        # 1. Turn test.
        u0, _ = look("turn_0")
        robot.base(dtheta_deg=TURN_DEG)
        u_plus, _ = look("turn_plus")
        robot.base(dtheta_deg=-2 * TURN_DEG)
        u_minus, _ = look("turn_minus")
        robot.base(dtheta_deg=TURN_DEG)
        u_per_deg = (u_plus - u_minus) / (2 * TURN_DEG)  # left turn = +theta moves it right
        log.ev("turn_test", u0=u0, u_plus=u_plus, u_minus=u_minus, u_per_deg=u_per_deg)
        print(f"turn test: {u_per_deg:.4f} of the width per degree (was {was})")
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
            det = see(f"drive_{len(rows)}")
            if det is None:
                break
            rows.append({"x": odom_x(), "u": float(det.center[0]), "bottom": float(det.box[3])})
            log.ev("drive_row", **rows[-1])
            if rows[-1]["bottom"] >= STOP_BOTTOM:
                break
            robot.base(dx=STEP_M)
        if len(rows) < 4:
            raise SystemExit(f"only {len(rows)} drive rows: start further away")

        # 3. Wrist line-up at the reach point: the base puts the brick under it.
        lu = actions.hover(sk)
        if lu.ok:
            lu = actions.line_up(sk, args.target)
        x_end = odom_x()
        det = sk.find(args.target, "wrist", fast_ok=False) if lu.ok else None
        h = sk.height()
        arm.reset()
        if det is None:
            raise SystemExit(f"the wrist line-up failed ({lu.reason or 'brick not seen'}): nothing saved")
        # brick = the point under the TCP (the reach point) + d
        d = locate_from_one_view(sk.geom, det.center, h, prm.BRICK_HEIGHT_M.value, sk.roll)
        log.ev("wrist_check", center=det.center, dxy=d, height=h, x_end=x_end)

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

    near = bottom_at(0.0)  # at the reach point
    # Straight ahead of the arm: the last rows' column, corrected by the brick's measured side
    # offset seen from the arm base (left = +y = smaller u).
    bearing_deg = math.degrees(math.atan2(d[1], prm.REACH_POINT_X_M.value + d[0]))
    u_target = float(np.mean([r["u"] for r in rows[-3:]]) + u_per_deg * bearing_deg)
    value = {
        "u_per_deg": round(u_per_deg, 4),
        "u_target": round(u_target, 3),
        "range_k": round(k, 4),
        "range_horizon_v": round(vh, 3),
        "range_offset_m": round(off, 3),
        "near_bottom": round(near, 3),
    }
    rms = math.sqrt(best[0] / len(idx))
    if rms > 0.03 or not 0.0 < u_target < 1.0:
        raise SystemExit(f"fit too poor to save (rms {1000 * rms:.0f} mm, u_target {u_target:.2f}): {value}")
    print(
        "rows (beyond reach m, bottom):",
        [(round(b, 3), round(v, 3)) for b, v in zip(beyond, bottom, strict=True)],
    )
    print(f"fit rms {1000 * rms:.0f} mm:", value)
    if near > 0.99:
        print("WARNING: the stop row is at the very bottom of the view: approach can't see that close")
    calib_path = ROOT / "data" / "calib.json"
    calib = json.loads(calib_path.read_text())
    calib["front_camera"] = {
        "value": value,
        "meaning": "front-camera geometry for the approach and object heights (brain/geometry.py, goals.FrontApproachGoal)",
        "method": "calibrate_front.py: turn test, straight drive at a brick, wrist check at the reach point",
        "date": datetime.date.today().isoformat(),
        "log": str(out.relative_to(ROOT)),
        "fit_rms_m": round(rms, 4),
    }
    calib_path.write_text(json.dumps(calib, indent=2) + "\n")
    print("saved to calib.json; log", out)


if __name__ == "__main__":
    main()
