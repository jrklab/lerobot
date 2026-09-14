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

"""
calibrate_joint.py -- Recalibrate a single joint on LeKiwi or a standalone SO101/SO100
follower arm (e.g. after replacing one motor) without redoing the whole robot's
calibration.

`lerobot-calibrate` always recalibrates every motor in one pass, even if only one was
touched. The underlying bus methods (`set_half_turn_homings`, `record_ranges_of_motion`,
`write_calibration`) already support operating on a single motor -- this script just
does that, leaving every other motor's calibration in the file and on-device completely
untouched.

Two-step workflow for a freshly replaced motor:

  1. (Only if the replacement motor still has its factory-default ID) Assign it the
     correct ID for the joint it replaces:

       uv run python examples/lekiwi/calibrate_joint.py --joint arm_wrist_flex --setup-id-only

     Follow the same "connect the controller board to this motor only" procedure as
     LeKiwi.setup_motors() -- this just does it for the one motor instead of all of them.

  2. Calibrate that joint (homing + range of motion, same procedure as the full
     `lerobot-calibrate` flow, just scoped to this motor):

       uv run python examples/lekiwi/calibrate_joint.py --joint arm_wrist_flex

  Base wheels (base_left_wheel/base_back_wheel/base_right_wheel, LeKiwi only) need no
  manual motion during calibration -- they're continuous-rotation motors with a fixed
  homing_offset=0 and a hardcoded full-turn range, matching how the full-robot
  calibration already treats them.

Usage:
  # LeKiwi (default robot type):
  uv run python examples/lekiwi/calibrate_joint.py --joint arm_wrist_flex \
      --robot-id kiwi_sn_0 --port /dev/ttyUSB0

  # Standalone SO101/SO100 follower arm:
  uv run python examples/lekiwi/calibrate_joint.py --robot-type so101_follower \
      --joint wrist_flex --robot-id so101_test --port /dev/ttyACM0
"""

import argparse

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.robots.lekiwi.lekiwi import LeKiwi
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ROBOT_CLASSES = {
    "lekiwi": (LeKiwiConfig, LeKiwi),
    # so100_follower uses the exact same config/robot classes as so101_follower.
    "so101_follower": (SOFollowerRobotConfig, SOFollower),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot-type",
        default="lekiwi",
        choices=sorted(ROBOT_CLASSES),
        help="Which robot this joint belongs to (default: lekiwi).",
    )
    parser.add_argument(
        "--joint", required=True, help="Motor name to (re)calibrate, e.g. arm_wrist_flex, base_left_wheel"
    )
    parser.add_argument("--robot-id", required=True, help="Robot id (calibration file name).")
    parser.add_argument("--port", required=True, help="Serial port for the motor bus.")
    parser.add_argument(
        "--setup-id-only",
        action="store_true",
        help="Only assign this motor's ID (for a fresh replacement motor still at its factory-default "
        "ID/baudrate); does not run calibration. Run again without this flag afterward.",
    )
    args = parser.parse_args()

    config_cls, robot_cls = ROBOT_CLASSES[args.robot_type]
    robot = robot_cls(config_cls(port=args.port, id=args.robot_id))

    # LeKiwi exposes .arm_motors/.base_motors; a standalone follower arm has neither
    # attribute (there's no base) -- every one of its motors counts as an "arm motor".
    arm_motors = getattr(robot, "arm_motors", list(robot.bus.motors))
    base_motors = getattr(robot, "base_motors", [])

    if args.joint not in robot.bus.motors:
        raise SystemExit(f"Unknown joint {args.joint!r}. Valid motors: {sorted(robot.bus.motors)}")

    if args.setup_id_only:
        # setup_motor() connects itself (without the all-motors handshake, since a fresh
        # motor won't pass that check yet) -- do not call robot.bus.connect() first.
        input(
            f"Disconnect every other motor from the bus and connect only '{args.joint}', then press ENTER..."
        )
        robot.bus.setup_motor(args.joint)
        print(f"'{args.joint}' motor ID set to {robot.bus.motors[args.joint].id}. Reconnect the full "
              f"daisy chain, then re-run this script without --setup-id-only to calibrate it.")
        return

    if not robot.calibration:
        raise SystemExit(
            f"No existing calibration file found at {robot.calibration_fpath} -- run the full "
            "`lerobot-calibrate` once first; this script only replaces one motor's entry in an "
            "existing file."
        )

    print(f"Loaded existing calibration for {len(robot.calibration)} motors from {robot.calibration_fpath}")
    print(f"Recalibrating ONLY '{args.joint}' -- every other motor's calibration is left untouched.")

    robot.bus.connect()  # bus only: skip cameras and LeKiwi.connect()'s auto-full-calibrate path
    try:
        is_wheel = args.joint in base_motors

        if is_wheel:
            robot.bus.disable_torque([args.joint])
            # Wheels are continuous-rotation motors -- no meaningful "home" position or
            # mechanical range, so the full-robot calibration hardcodes these too.
            print(f"'{args.joint}' is a base wheel: using a fixed full-turn calibration, no motion needed.")
            homing_offset = 0
            range_min, range_max = 0, 4095
        else:
            # Disable torque on the WHOLE arm, not just the target joint -- matches how the
            # full LeKiwi.calibrate() flow does it. Leaving the other arm joints
            # torque-enabled means you're moving this joint against a rigid linked
            # structure, which can spike current draw on the still-torqued motors and,
            # on a shared bus, corrupt reads for any motor (this caused an intermittent
            # "Incorrect status packet" error here in practice).
            robot.bus.disable_torque(arm_motors)
            robot.bus.write("Operating_Mode", args.joint, OperatingMode.POSITION.value)
            input(f"Move '{args.joint}' to the middle of its range of motion and press ENTER...")
            homing_offsets = robot.bus.set_half_turn_homings([args.joint])
            homing_offset = homing_offsets[args.joint]

            print(f"Move '{args.joint}' through its entire range of motion. Press ENTER to stop...")
            range_mins, range_maxes = robot.bus.record_ranges_of_motion([args.joint])
            range_min, range_max = range_mins[args.joint], range_maxes[args.joint]

        robot.calibration[args.joint] = MotorCalibration(
            id=robot.bus.motors[args.joint].id,
            drive_mode=0,
            homing_offset=homing_offset,
            range_min=range_min,
            range_max=range_max,
        )

        # write_calibration() only writes EEPROM for motors present in the dict passed in, so
        # passing the full (8 unchanged + 1 new) dict just re-writes the same values to the
        # unchanged motors -- harmless, and keeps bus.calibration's in-memory cache consistent
        # (it replaces the cache wholesale, not merges, so a partial dict here would be wrong).
        robot.bus.write_calibration(robot.calibration)
        robot._save_calibration()
        print(f"Saved updated calibration to {robot.calibration_fpath}")

        robot.configure()  # restore normal operating mode/torque for all motors
    finally:
        robot.bus.disconnect(robot.config.disable_torque_on_disconnect)


if __name__ == "__main__":
    main()
