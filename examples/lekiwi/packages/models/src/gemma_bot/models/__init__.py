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

"""Robot models shared by the brain and the executors: what both sides must agree on about a
robot's body (its kinematics, its joint units, its limits), so neither copies the other's numbers.

    so101_kinematics  the SO-101 arm's forward and inverse kinematics
    joint_angles      joint positions in degrees from a robot calibration
    so101             SO-101 arm constants: jaw tip offset, joint speed limit, stall thresholds
    lekiwi_base       LeKiwi base constants: where the executor ends a drive or a turn
"""
