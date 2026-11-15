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

"""SO-101 arm constants that the executor enforces and the brain plans with."""

MAX_JOINT_SPEED = 60.0  # deg/s, on any joint in any segment (was 60 units/s: 53-91 deg/s)
# The jaw tips extend this far past the TCP along its approach axis: the SO-101 URDF's own geometry
# (so101_new_calib.urdf's jaw meshes placed as the model places them: the fixed jaw's tip 6.3 mm
# past gripper_frame_link, on its axis; the closed moving jaw's 4.5 mm), the owner's stock parts.
# Was 0.0097 (Oct 30): fitted so the floor touches and the cameras agreed under the hand
# calibration, absorbing its 3-4 deg zero errors; the self-calibrated zeros left that fit 5 mm off.
# Before that 0.015, an estimate. The executor's floor guard and the planner both use it.
JAW_TIP_OFFSET_M = 0.0063

# Stall detection, per examples/lekiwi/torque_feedback.md: a motor is "stalled" when it's
# under significant load (Present_Load, raw 0-1000) while barely moving (Present_Velocity,
# raw magnitude) -- high load alone can just mean fast acceleration, so the speed gate
# guards against false positives on an unloaded but quickly-moving joint.
# Per-motor load thresholds match torque_feedback.md's recommended `per_motor_thresholds`.
# The executor (web_server/robot_bridge.py) judges stalls with them; calibration proposes new ones.
STALL_LOAD_THRESHOLDS = {
    "arm_shoulder_pan": 0.3,
    "arm_shoulder_lift": 0.5,
    "arm_elbow_flex": 0.5,
    "arm_wrist_flex": 0.5,
    "arm_wrist_roll": 0.3,
    "arm_gripper": 0.2,
}
STALL_SPEED_THRESHOLD = 50  # raw Present_Velocity magnitude, matches torque_feedback.md's example
