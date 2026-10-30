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

"""Joint positions in degrees, for the auto interface (auto_control.py) and the planner on the
Mac (gemma_brain/common/arm.py), from one robot calibration.

LeRobot's bus works in normalized units: -100..100 over each arm joint's recorded range,
0..100 over the gripper's. A unit's size depends on that range (0.88-1.52 deg on this arm,
Oct 29), so every value stored in units changes meaning when a range is recalibrated. Degrees
don't:

    arm joints   degrees from the homing tick (half a turn, 2047 of 4096): LeRobot's
                 calibration writes each motor's Homing_Offset so the pose held at "the middle
                 of its range" reads that tick, and that pose is so101_new_calib.urdf's zero.
                 Not LeRobot's own degrees mode, which counts from the recorded range's
                 midpoint: off by however far the range isn't centred on the homing pose (15.7
                 deg on this arm's elbow, the Oct 27 joint-zero bug).
    gripper      degrees of opening from its recorded closed end (0 = closed); the gripper
                 isn't in the URDF's chain.

limits(): the recorded range clipped to the URDF's limits, at each end: past the URDF the arm
model isn't valid; short of it, the recorded range is where the joint was physically swept.

Uncalibrated (a mock robot, the offline simulation): -100..100 units = -180..180 deg, the
gripper's 0..100 = 0..180, as RobotBridge's fallback always was.
"""

from __future__ import annotations

import math

import so101_kinematics as kin

GRIPPER = "arm_gripper"
IK_ARM_JOINTS = [f"arm_{name}" for name in kin.JOINT_NAMES]
URDF_LIMITS_DEG = {f"arm_{n}": (math.degrees(lo), math.degrees(hi)) for n, (lo, hi) in kin.JOINT_LIMITS.items()}
UNCALIBRATED_DEG_PER_UNIT = 1.8  # +/-100 units = +/-180 deg


class JointAngles:
    def __init__(self, joints: dict[str, dict] | None = None, resolution: int = 4096):
        """joints: {joint: {range_min, range_max, drive_mode}} in motor ticks, as the Pi's
        /auto/calibration returns it."""
        self.joints = joints or {}
        self.resolution = resolution
        self.homing_tick = (resolution - 1) // 2

    def _ticks_per_deg(self) -> float:
        return self.resolution / 360.0

    def to_deg(self, joint: str, units: float) -> float:
        c = self.joints.get(joint)
        if not c:
            return units * UNCALIBRATED_DEG_PER_UNIT
        span = c["range_max"] - c["range_min"]
        if joint == GRIPPER:
            return units / 100.0 * span / self._ticks_per_deg()
        ticks = c["range_min"] + (units + 100.0) / 200.0 * span
        deg = (ticks - self.homing_tick) / self._ticks_per_deg()
        return -deg if c["drive_mode"] else deg

    def to_units(self, joint: str, deg: float) -> float:
        c = self.joints.get(joint)
        if not c:
            return deg / UNCALIBRATED_DEG_PER_UNIT
        span = c["range_max"] - c["range_min"]
        if joint == GRIPPER:
            return deg * self._ticks_per_deg() / span * 100.0
        deg = -deg if c["drive_mode"] else deg
        ticks = deg * self._ticks_per_deg() + self.homing_tick
        return (ticks - c["range_min"]) / span * 200.0 - 100.0

    def limits(self, joint: str) -> tuple[float, float]:
        """(lo, hi) degrees a command may reach: the recorded range, clipped to the URDF's."""
        if joint == GRIPPER:
            return 0.0, self.to_deg(GRIPPER, 100.0)
        lo, hi = sorted((self.to_deg(joint, -100.0), self.to_deg(joint, 100.0)))
        ulo, uhi = URDF_LIMITS_DEG.get(joint, (-math.inf, math.inf))
        return max(lo, ulo), min(hi, uhi)
