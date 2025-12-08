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

import time

from lerobot.robots.hrobot.hrobot_client import HRobotClient, HRobotClientConfig
from lerobot.teleoperators.bi_so101_leader.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# --- Robot and Teleop Configuration ---
# IMPORTANT: Replace with your actual IP address and port.
robot_config = HRobotClientConfig(remote_ip="192.168.1.176", id="hrobot_follower")
teleop_config = BiSO101LeaderConfig(port="/dev/ttyUSB1", id="hrobot_leader")
keyboard_config = KeyboardTeleopConfig(id="keyboard_base_control")

# --- Initialization ---
robot = HRobotClient(robot_config)
leader_arms = BiSO101Leader(teleop_config)
keyboard = KeyboardTeleop(keyboard_config)

# --- Connection ---
# Make sure the host script is running on the robot:
# python -m lerobot.robots.hrobot.hrobot_host --robot.id=hrobot_follower
robot.connect()
leader_arms.connect()
keyboard.connect()

init_rerun(session_name="hrobot_teleop")

if not all((robot.is_connected, leader_arms.is_connected, keyboard.is_connected)):
    raise ConnectionError("Failed to connect to one or more devices.")

print("Starting teleoperation loop...")
while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    arm_action = leader_arms.get_action()
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    head_action = robot._from_keyboard_to_head_action(keyboard_keys, observation)

    action = {**arm_action, **base_action, **head_action}
    _ = robot.send_action(action)

    log_rerun_data(observation=observation, action=action)

    busy_wait(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))