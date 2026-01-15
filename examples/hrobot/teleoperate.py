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

import argparse
import time

from lerobot.robots.hrobot.hrobot_client import HRobotClient, HRobotClientConfig
from lerobot.teleoperators.bi_so101_leader.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.voice.teleop_voice import VoiceTeleop
from lerobot.teleoperators.voice.configuration_voice import VoiceTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="HRobot Teleoperation Script")
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["keyboard", "voice", "both"],
        default="keyboard",
        help="The control mode for teleoperation.",
    )
    args = parser.parse_args()

    # --- Robot and Teleop Configuration ---
    # IMPORTANT: Replace with your actual IP address and port.
    # robot_config = HRobotClientConfig(remote_ip="localhost", id="hrobot_follower") # localhost for debugging
    robot_config = HRobotClientConfig(remote_ip="raspberrypi.local", id="hrobot_follower") # with raspberry pi
    teleop_config = BiSO101LeaderConfig(port="/dev/ttyUSB0", id="hrobot_leader")
    keyboard_config = KeyboardTeleopConfig(id="keyboard_base_control")
    voice_config = VoiceTeleopConfig(id="voice_control")

    # --- Initialization ---
    robot = HRobotClient(robot_config)
    leader_arms = BiSO101Leader(teleop_config)
    
    keyboard = None
    if args.control_mode in ["keyboard", "both"]:
        keyboard = KeyboardTeleop(keyboard_config)

    voice = None
    if args.control_mode in ["voice", "both"]:
        voice = VoiceTeleop(voice_config)

    # --- Connection ---
    # Make sure the host script is running on the robot:
    # python -m lerobot.robots.hrobot.hrobot_host --robot.id=hrobot_follower
    robot.connect()
    leader_arms.connect()
    if keyboard:
        keyboard.connect()
    if voice:
        voice.connect()

    init_rerun(session_name="hrobot_teleop")

    connected_devices = [robot.is_connected, leader_arms.is_connected]
    if keyboard:
        connected_devices.append(keyboard.is_connected)
    # VoiceTeleop connect is async and might take a moment
    # if voice:
    #     connected_devices.append(voice.is_connected)
    
    if not all(connected_devices):
        raise ConnectionError("Failed to connect to one or more devices.")

    print("Starting teleoperation loop...")
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()
        arm_action = leader_arms.get_action()

        base_action = {}
        head_action = {}
        
        keyboard_keys = {}
        use_keyboard_for_base = False
        use_keyboard_for_head = False

        if args.control_mode in ["keyboard", "both"]:
            keyboard_keys = keyboard.get_action()
            
            base_movement_keys = ["forward", "backward", "left", "right", "rotate_left", "rotate_right", "speed_up", "speed_down"]
            head_movement_keys = ["head_pan_left", "head_pan_right", "head_lift_up", "head_lift_down", "head_reset"]
            
            # Check if any key related to base movement is pressed
            if any(robot.teleop_keys.get(k) in keyboard_keys for k in base_movement_keys):
                use_keyboard_for_base = True
            
            # Check if any key related to head movement is pressed
            if any(robot.teleop_keys.get(k) in keyboard_keys for k in head_movement_keys):
                use_keyboard_for_head = True

        voice_command_dict = None
        voice_command = None
        if args.control_mode in ["voice", "both"]:
            voice_command_dict = voice.get_action()
            voice_command = voice_command_dict.get("voice_command")

        if use_keyboard_for_base:
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)
            robot.voice_base_velocity = {"x": 0.0, "y": 0.0, "theta": 0.0}
        elif args.control_mode in ["voice", "both"]:
            base_action = robot._from_voice_to_base_action(voice_command)
            if voice_command and voice_command[0] == 'base':
                print(f"Executing voice command: {voice_command}")

        if use_keyboard_for_head:
            head_action = robot._from_keyboard_to_head_action(keyboard_keys, observation)
        elif args.control_mode in ["voice", "both"]:
            head_action = robot._from_voice_to_head_action(voice_command, observation)
            if voice_command and voice_command[0] == 'head':
                print(f"Executing voice command: {voice_command}")

        action = {**arm_action, **base_action, **head_action}
        _ = robot.send_action(action)

        log_rerun_data(observation=observation, action=action)

        busy_wait(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))

if __name__ == "__main__":
    main()
