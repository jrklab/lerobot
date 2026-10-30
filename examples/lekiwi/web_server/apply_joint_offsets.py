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

"""Correct arm joint zeros in the robot's own calibration, from the offsets the camera calibration
measured (gemma_brain/calibration/solve_cameras.py --fit-offsets: "to apply"). Run on the Pi,
with the web service stopped:

    sudo systemctl stop lekiwi-web
    .venv/bin/python examples/lekiwi/web_server/apply_joint_offsets.py \\
        --port-name /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 --offsets arm_elbow_flex=-9.0 [--write]
    sudo systemctl start lekiwi-web

Without --write it only prints what would change. With --write: the calibration file is backed
up next to itself, the arm's torque is switched off (the arm goes limp: fold it to rest or hold
it), each corrected motor gets its new Homing_Offset and position limits, the file is saved, and
the motors are checked to match it. Both must change together: at the next start LeKiwi.connect()
compares them and, on a mismatch, asks at the keyboard for a recalibration, which the service
can't answer. --restore <backup> writes a backup back the same way.

Teleop, the Pi's auto interface and the Mac planner all read the corrected zeros from here:
there is no second offset table to keep in step. After a correction, run the camera capture
again: its offsets should come out within their spread.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import shutil
from pathlib import Path

from joint_angles import GRIPPER, shifted_calibration


def _parse_offsets(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        joint, _, deg = item.partition("=")
        out[joint if joint.startswith("arm_") else f"arm_{joint}"] = float(deg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-id", default="kiwi_sn_0", help="as server.py: the calibration file's name")
    ap.add_argument("--port-name", default="/dev/ttyUSB0", help="as server.py: the motor bus's serial port")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--offsets", nargs="+", metavar="JOINT=DEG", help="e.g. arm_elbow_flex=-9.0")
    group.add_argument("--restore", type=Path, help="a backup this script made")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from lerobot.motors import MotorCalibration
    from lerobot.robots.lekiwi import LeKiwi
    from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

    robot = LeKiwi(LeKiwiConfig(port=args.port_name, id=args.robot_id))
    current = {m: dataclasses.asdict(c) for m, c in robot.calibration.items()}
    if args.restore:
        new = json.loads(args.restore.read_text())
    else:
        new = {
            **current,
            **shifted_calibration(
                {j: current[j] for j in current if j.startswith("arm_")}, _parse_offsets(args.offsets)
            ),
        }
    changed = [m for m in new if new[m] != current.get(m)]
    for m in changed:
        print(
            f"{m}: "
            + ", ".join(
                f"{k} {current[m][k]} -> {new[m][k]}"
                for k in ("homing_offset", "range_min", "range_max")
                if current[m][k] != new[m][k]
            )
        )
    if not changed:
        print("nothing to change")
        return
    if GRIPPER in changed:
        raise SystemExit("refusing to change the gripper's calibration")
    if not args.write:
        print("(dry run: --write to apply)")
        return

    backup = robot.calibration_fpath.with_name(
        f"{robot.calibration_fpath.name}.before-zero-fix-{datetime.datetime.now():%Y%m%dT%H%M%S}"
    )
    shutil.copy2(robot.calibration_fpath, backup)
    print(f"backed up {robot.calibration_fpath} to {backup}")
    robot.bus.connect()
    try:
        robot.bus.disable_torque(robot.arm_motors)  # writing a motor's offset and limits needs it off
        robot.bus.write_calibration({m: MotorCalibration(**new[m]) for m in changed}, cache=False)
        robot.calibration = {m: MotorCalibration(**c) for m, c in new.items()}
        robot.bus.calibration = robot.calibration
        robot._save_calibration()
        if not robot.bus.is_calibrated:
            raise SystemExit(
                f"the motors don't match the saved file: restore with --restore {backup} --write"
            )
        print(f"wrote {', '.join(changed)} to the motors and {robot.calibration_fpath}; the motors match it")
    finally:
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
