# !/usr/bin/env python

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

import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.teleoperators.torque_feedback import TorqueFeedbackConfig, map_load_to_torque_limit
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip="raspberrypi.local", id="my_lekiwi")
    teleop_arm_config = SO101LeaderConfig(port="/dev/ttyUSB0", id="leader_arm_1")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO101Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Init rerun viewer
    init_rerun(session_name="lekiwi_teleop")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    print("Press 'b' to toggle torque feedback ON/OFF (starts disabled).")

    # Torque feedback config: per-motor scales and thresholds for SO101 arm
    torque_feedback_config = TorqueFeedbackConfig(
        enabled=False,  # starts disabled; toggle with 'b'
        global_scale_factor=1.0,
        per_motor_scales={
            "shoulder_pan": 0.5,
            "shoulder_lift": 0.5,
            "elbow_flex": 0.5,
            "wrist_flex": 0.5,
            "wrist_roll": 0.5,
            "gripper": 0.3,
        },
        per_motor_thresholds={
            "shoulder_pan": 0.3,
            "shoulder_lift": 0.5,
            "elbow_flex": 0.5,
            "wrist_flex": 0.5,
            "wrist_roll": 0.3,
            "gripper": 0.2,
        },
        speed_threshold=50,  # suppress feedback if raw speed > 50 (motor is moving, not stalled)
    )
    _prev_b_pressed = False

    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        # Arm
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        # print arm action for debugging with timestamp
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Arm action: {arm_action}")
        # Keyboard
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        # Toggle torque feedback with 'b' (edge-detect: trigger only on key-down)
        b_pressed = "b" in keyboard_keys
        if b_pressed and not _prev_b_pressed:
            torque_feedback_config.enabled = not torque_feedback_config.enabled
            if torque_feedback_config.enabled:
                print("\n[Torque feedback: ENABLED]")
            else:
                print("\n[Torque feedback: DISABLED]")
                # Send all-zero feedback to disable torque on leader arm
                zero_feedback = {f"{m}.torque": 0 for m in [
                    "shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll", "gripper"
                ]}
                leader_arm.send_feedback(zero_feedback)
        _prev_b_pressed = b_pressed

        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # Send action to robot
        _ = robot.send_action(action)

        # Send torque feedback to leader arm if enabled
        if torque_feedback_config.enabled:
            # Extract load values: strip "arm_" prefix and ".load" suffix to get motor names
            load_dict = {
                k.removeprefix("arm_").removesuffix(".load"): v
                for k, v in observation.items()
                if k.startswith("arm_") and k.endswith(".load")
            }
            # Extract speed values for stall detection
            speed_dict = {
                k.removeprefix("arm_").removesuffix(".speed"): v
                for k, v in observation.items()
                if k.startswith("arm_") and k.endswith(".speed")
            }
            # print load dict for debugging with timestamp
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Load dict: {load_dict}")
            if load_dict:
                torque_limits = map_load_to_torque_limit(
                    load_dict, torque_feedback_config, list(load_dict.keys()), speed_dict=speed_dict
                )
                feedback_dict = {f"{motor}.torque": val for motor, val in torque_limits.items()}
                leader_arm.send_feedback(feedback_dict)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
