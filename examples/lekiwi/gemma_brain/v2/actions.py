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

"""v2 layer 3, actions: readable names for common uses of the primitives (skills.py). They hold
no numbers of their own (test_v2.test_actions_and_chains_hold_no_numbers): every value is a
params.py or geometry.py one, so a new action adds a name, not a parameter to calibrate."""

from __future__ import annotations

from v2 import params as prm
from v2.goals import AngleGoal, FrontApproachGoal, ReachGoal
from v2.skills import Result, Skills2


def land(sk: Skills2) -> Result:
    """Down to GRASP_LIFT_M above the predicted floor without pressing into it (v1's touch
    pushed into the floor and slid 8-10 mm, Oct 27); the contact stop stays as the safety. A
    contact more than half a brick above the floor may be the brick's edge: refused."""
    r = sk.move("arm", height=prm.GRASP_LIFT_M.value, until_contact=True)
    if not r.ok or not r.data.get("contact"):
        return r
    above = r.data["contact_z"] - r.data["floor"]
    if above > prm.BRICK_HEIGHT_M.value / 2:
        return Result(False, f"met something {1000 * above:.0f} mm above the floor", r.data)
    # The floor, a little higher than predicted: back up to the grasp lift above it.
    return sk.move("arm", delta=(0.0, 0.0, r.data["contact_z"] + prm.GRASP_LIFT_M.value - sk.tcp()[2]))


def lift(sk: Skills2) -> Result:
    """Straight up by the demo's lift, then whether the object is still held."""
    r = sk.move("arm", delta=(0.0, 0.0, prm.LIFT_M.value))
    if not r.ok:
        return r
    held = sk.held()
    return Result(held, "" if held else "nothing held after the lift", {**r.data, "held": held})


def approach(sk: Skills2, obj: str) -> Result:
    """Base: turn and drive until the front camera sees the object at the reach point."""
    return sk.servo(obj, FrontApproachGoal())


def hover(sk: Skills2) -> Result:
    """The open, vertical gripper over the reach point at the hover height, roll 0."""
    r = sk.move("arm", xy=(prm.REACH_POINT_X_M.value, 0.0), height=prm.HOVER_HEIGHT_M.value, roll=0.0)
    return sk.grip("open") if r.ok else r


def line_up(sk: Skills2, obj: str) -> Result:
    """Base, with the gripper hovering: the object within the arm's reach of the grasp spot."""
    return sk.servo(obj, ReachGoal())


def align(sk: Skills2, obj: str) -> Result:
    """Wrist roll: the object's long side across the jaws."""
    return sk.servo(obj, AngleGoal())
