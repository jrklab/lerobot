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

"""What the robot has, from its description (/v1/describe, common/robot.Robot.description): skills
ask here, not which robot it is. Without a description (the offline tests' simulated robot) the
answers are LeKiwi's: a base, a front and a wrist camera."""

from __future__ import annotations

from gemma_bot import protocol

from brain import params as prm

OFFLINE_CAMERAS = ("front", "wrist")


def _description(robot) -> protocol.Description | None:
    return getattr(robot, "description", None)


def has_base(robot) -> bool:
    d = _description(robot)
    return d is None or d.base is not None


def has_camera(robot, role: str) -> bool:
    d = _description(robot)
    return role in OFFLINE_CAMERAS if d is None else d.camera(role) is not None


def base_limits(robot) -> protocol.Base:
    """The base's limits as the robot reports them (its one-command limits, where it ends a drive
    or a turn), or params' values without a description (the offline tests)."""
    d = _description(robot)
    if d is not None and d.base is not None:
        return d.base
    return protocol.Base(
        kind="omni",
        max_speed_mps=prm.BASE_SPEED_MPS.value,
        max_turn_dps=prm.BASE_TURN_DPS.value,
        max_move_m=prm.BASE_MOVE_MAX_M.value,
        max_turn_deg=prm.BASE_TURN_MAX_DEG.value,
        pos_tol_m=prm.BASE_STOP_SHORT_M.value,
        turn_tol_deg=prm.BASE_TURN_STOP_SHORT_DEG.value,
    )
