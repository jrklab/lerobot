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

import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from lerobot.robots.hrobot.config_hrobot import HRobotConfig

logger = logging.getLogger(__name__)


class HRobot(Robot):
    """
    The hrobot includes a three omniwheel mobile base, two SO101 arms, and a pan-tilt head.
    All motors are connected to a single USB serial port.
    """

    config_class = HRobotConfig
    name = "hrobot"

    def __init__(self, config: HRobotConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # left arm
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # base
                "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                # right arm
                "right_arm_shoulder_pan": Motor(11, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(12, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(13, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(14, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(15, "sts3215", norm_mode_body),
                "right_arm_gripper": Motor(16, "sts3215", MotorNormMode.RANGE_0_100),
                # head
                "head_pan": Motor(17, "sts3215", norm_mode_body),
                "head_lift": Motor(18, "sts3215", norm_mode_body),
            },
            calibration=self.calibration,
        )
        self.arm_motors = [motor for motor in self.bus.motors if "arm" in motor]
        self.base_motors = [motor for motor in self.bus.motors if motor.startswith("base")]
        self.head_motors = [motor for motor in self.bus.motors if motor.startswith("head")]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "head_pan.pos",
                "head_lift.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")

        position_mode_motors = self.arm_motors + self.head_motors

        self.bus.disable_torque(position_mode_motors)
        for name in position_mode_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move all position-controlled motors to the middle of their range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings(position_mode_motors)
        homing_offsets.update(dict.fromkeys(self.base_motors, 0))

        unknown_range_motors = position_mode_motors

        print(
            "Move all arm and head joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for name in self.base_motors:
            range_mins[name] = 0
            range_maxes[name] = 4095

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        self.bus.disable_torque()
        self.bus.configure_motors()
        position_mode_motors = self.arm_motors + self.head_motors
        for name in position_mode_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", name, 16)
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.bus.enable_torque()

    def setup_motors(self) -> None:
        all_motors = self.arm_motors + self.head_motors + self.base_motors
        for motor in reversed(all_motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        # Copied from LeKiwi
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        elif speed_int < -0x8000:
            speed_int = -0x8000
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        # Copied from LeKiwi
        steps_per_deg = 4096.0 / 360.0
        return raw_speed / steps_per_deg

    def _body_to_wheel_raw(
        self, x: float, y: float, theta: float, wheel_radius: float = 0.05, base_radius: float = 0.125, max_raw: int = 3000
    ) -> dict:
        # Copied from LeKiwi
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats) if raw_floats else 0
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps *= scale
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        return {"base_left_wheel": wheel_raw[0], "base_back_wheel": wheel_raw[1], "base_right_wheel": wheel_raw[2]}

    def _wheel_raw_to_body(
        self, left_wheel_speed, back_wheel_speed, right_wheel_speed, wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> dict[str, Any]:
        # Copied from LeKiwi
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        m_inv = np.linalg.inv(m)
        x, y, theta_rad = m_inv.dot(wheel_linear_speeds)
        theta = theta_rad * (180.0 / np.pi)
        return {"x.vel": x, "y.vel": y, "theta.vel": theta}

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        pos_motors = self.arm_motors + self.head_motors
        present_pos = self.bus.sync_read("Present_Position", pos_motors)
        base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"], base_wheel_vel["base_back_wheel"], base_wheel_vel["base_right_wheel"]
        )

        pos_state = {f"{k}.pos": v for k, v in present_pos.items()}
        obs_dict = {**pos_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        pos_goal = {k: v for k, v in action.items() if k.endswith(".pos")}
        vel_goal = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            vel_goal.get("x.vel", 0.0), vel_goal.get("y.vel", 0.0), vel_goal.get("theta.vel", 0.0)
        )

        if self.config.max_relative_target is not None:
            pos_motors = self.arm_motors + self.head_motors
            present_pos = self.bus.sync_read("Present_Position", pos_motors)
            goal_present_pos = {key.removesuffix(".pos"): (g_pos, present_pos[key.removesuffix(".pos")]) for key, g_pos in pos_goal.items()}
            safe_pos_goal_raw = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            pos_goal = {f"{k}.pos": v for k, v in safe_pos_goal_raw.items()}

        # Clamp head positions to calibration limits
        for motor_name in self.head_motors:
            pos_key = f"{motor_name}.pos"
            if pos_key in pos_goal:
                # Assuming normalized values are in the same range as what the bus expects after normalization.
                # The bus normalizer will handle the conversion from normalized value to raw value.
                # We are clamping the normalized value here.
                pos_goal[pos_key] = np.clip(pos_goal[pos_key], -100, 100)

        pos_goal_raw = {k.removesuffix(".pos"): v for k, v in pos_goal.items()}
        self.bus.sync_write("Goal_Position", pos_goal_raw)
        self.bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {**pos_goal, **vel_goal}

    def stop_base(self):
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")