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

"""Wrist camera: how a TCP move shows up in the image (the image Jacobian), measured by
moving the gripper over a block on the floor and asking Gemma where the block is.

    .venv/bin/python calibration/wrist_calib.py [--x 0.22] [--z 0.0] [--step 0.02] [--write]

The gripper points straight down at (x, 0, z); then it moves +/-step along X and Y (arm base
frame), and each time a fresh wrist frame (taken after the move settled) goes to Gemma's box
call. Least squares over the moves gives d(pixel)/d(TCP xy) at this height. Frames, boxes
and poses are saved under runs/<timestamp>-wrist-calib/. No move goes lower than z.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from common.arm import Arm
from common.gemma import Gemma, LlamaCpp
from common.robot import Robot

WHAT = "the Lego brick on the floor"
CALIB = ROOT / "data" / "calib.json"


def look(robot: Robot, gemma: Gemma, out: Path, tag: str, arm: Arm) -> dict:
    time.sleep(0.4)
    jpeg, t = robot.frame("wrist", after=time.time())
    (out / f"{tag}.jpg").write_bytes(jpeg)
    a = gemma.box(jpeg, WHAT)
    tcp = arm.tcp()
    rec = {
        "tag": tag,
        "tcp_xyz": tcp[:3, 3].round(4).tolist(),
        "box": a.value,
        "center": None if a.value is None else [(a.value[0] + a.value[2]) / 2, (a.value[1] + a.value[3]) / 2],
        "gemma_ms": round(a.ms),
        "answer": a.text,
    }
    print(f"{tag:8s} tcp {np.round(tcp[:3, 3], 3)}  box {None if a.value is None else np.round(a.value, 3)}  {a.ms:.0f} ms")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=float, default=0.22)
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--vision-url", default="http://127.0.0.1:8081", help="llama-server for the boxes ('' = Ollama Gemma)")
    ap.add_argument("--hold", action="store_true", help="only go to the centre pose and stay there (place the brick under it)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out = ROOT / "runs" / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-wrist-calib")
    out.mkdir(parents=True)
    robot, gemma = Robot(), (LlamaCpp(args.vision_url) if args.vision_url else Gemma())
    recs = []
    with robot:
        arm = Arm(robot)
        try:
            arm.pose("ready", speed=20)
            center = np.array([args.x, 0.0, args.z])
            arm.move_to(center, pitch_deg=90, roll_deg=0, speed=0.03)
            if args.hold:
                print(f"holding at {center}: place the brick under the gripper")
                return
            recs.append(look(robot, gemma, out, "center", arm))
            s = args.step
            for tag, d in (("+x", (s, 0)), ("-x", (-s, 0)), ("+y", (0, s)), ("-y", (0, -s))):
                arm.move_to(center + [d[0], d[1], 0.0], pitch_deg=90, roll_deg=0, speed=0.02)
                recs.append(look(robot, gemma, out, tag, arm))
            arm.move_to(center, pitch_deg=90, roll_deg=0, speed=0.02)
            recs.append(look(robot, gemma, out, "center2", arm))
        finally:
            (out / "records.json").write_text(json.dumps(recs, indent=2))
            if not args.hold:
                arm.reset()

    ok = [r for r in recs if r["center"] is not None]
    if len(ok) < 3:
        print("not enough detections to fit")
        return
    # center (u, v) = c0 + J @ (x - x0, y - y0), least squares over every detection.
    p0 = np.array(ok[0]["tcp_xyz"][:2])
    a_rows = [[1, 0, *(np.array(r["tcp_xyz"][:2]) - p0), 0, 0] for r in ok] + [
        [0, 1, 0, 0, *(np.array(r["tcp_xyz"][:2]) - p0)] for r in ok
    ]
    b = [r["center"][0] for r in ok] + [r["center"][1] for r in ok]
    sol, res, *_ = np.linalg.lstsq(np.array(a_rows, dtype=float), np.array(b), rcond=None)
    j = np.array([[sol[2], sol[3]], [sol[4], sol[5]]])
    pred = np.array(a_rows, dtype=float) @ sol
    rms_px = float(np.sqrt(np.mean((pred - b) ** 2))) * 640
    print(f"\nimage Jacobian d(u,v)/d(x,y) [image fraction per m]:\n{np.round(j, 2)}")
    print(f"fit rms {rms_px:.1f} px; center repeat: {np.round(ok[0]['center'], 3)} vs {np.round(recs[-1]['center'], 3) if recs[-1]['center'] else None}")
    print(f"saved {out}")
    if args.write:
        calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
        calib["wrist_image_jacobian"] = {
            "value": np.round(j, 3).tolist(),
            "unit": "image fraction (u right, v down) per metre of TCP motion (arm base x, y)",
            "pose": f"TCP pointing straight down, roll 0, at ({args.x}, 0, {args.z})",
            "method": f"wrist_calib.py: +/-{args.step} m moves, Gemma box centres, least squares (rms {rms_px:.1f} px)",
            "date": datetime.date.today().isoformat(),
            "log": str(out.relative_to(ROOT)),
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")


if __name__ == "__main__":
    main()
