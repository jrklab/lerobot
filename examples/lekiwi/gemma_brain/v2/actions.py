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
from v2.goals import AboveGoal, AngleGoal, BesideGoal, FrontApproachGoal, ReachGoal
from v2.params import Param
from v2.skills import Result, Skills2


def land(sk: Skills2) -> Result:
    """Down to GRASP_LIFT_M above the predicted floor without pressing into it (v1's touch
    pushed into the floor and slid 8-10 mm, Oct 27); the contact stop stays as the safety. A
    contact more than half a brick above the floor may be the brick's edge: refused. From higher
    up it first goes freely to FINAL_LOOK_HEIGHT_M: the Pi's contact watch learns the arm's
    droop at the start, and over a 10 cm descent the droop changes by more than its 1.5 mm
    threshold (Oct 27: "held up" 83 mm above the floor, runs 20261027T181003 and later -v2-land)."""
    if sk.height() > prm.FINAL_LOOK_HEIGHT_M.value:
        r = sk.move("arm", height=prm.FINAL_LOOK_HEIGHT_M.value)
        if not r.ok:
            return r
    r = sk.move("arm", height=prm.GRASP_LIFT_M.value, until_contact=True)
    if not r.ok or not r.data.get("contact"):
        return r
    above = r.data["contact_z"] - r.data["floor"]
    if above > prm.BRICK_HEIGHT_M.value / 2:
        return Result(False, f"met something {above:.3f} m above the floor", r.data)
    # The floor, a little higher than predicted: back up to the grasp lift above it.
    return sk.move("arm", delta=(0.0, 0.0, r.data["contact_z"] + prm.GRASP_LIFT_M.value - sk.tcp()[2]))


def lift(sk: Skills2) -> Result:
    """Straight up by the demo's lift, or, where the vertical gripper can't reach that high (at
    27 cm out it goes ~11 cm up; Oct 27 v2 pick), by half as much again until it can, but not
    less than the final look height; then whether the object is still held. The planner refuses
    before anything moves, so a try costs nothing."""
    up = prm.LIFT_M.value
    r = sk.move("arm", delta=(0.0, 0.0, up))
    while not r.ok and up / 2 >= prm.FINAL_LOOK_HEIGHT_M.value:
        up /= 2
        r = sk.move("arm", delta=(0.0, 0.0, up))
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


def hover_at(sk: Skills2, height: Param) -> Result:
    """The vertical gripper over the reach point at `height`, roll 0 (jaws as they are)."""
    return sk.move("arm", xy=(prm.REACH_POINT_X_M.value, 0.0), height=height.value, roll=0.0)


def over_container(sk: Skills2, container: str) -> Result:
    """Base, then arm: the container where the owner's demo had it before lowering in."""
    r = sk.servo(container, AboveGoal(tol_m=prm.REACH_TOL_M), part="base")
    return sk.servo(container, AboveGoal()) if r.ok else r


def beside(sk: Skills2, ref: str) -> Result:
    """Base, then arm: the reference object just beside where the held one will land."""
    r = sk.servo(ref, BesideGoal(tol_m=prm.REACH_TOL_M), part="base")
    return sk.servo(ref, BesideGoal()) if r.ok else r


def lower_to_release(sk: Skills2) -> Result:
    """Straight down to the demo's release height; stops early if the held object meets
    something (the container's bottom)."""
    return sk.move("arm", height=prm.RELEASE_ABOVE_FLOOR_M.value, until_contact=True)


def release(sk: Skills2) -> Result:
    """Open the jaws and check they hold nothing any more."""
    r = sk.grip("open")
    if not r.ok:
        return r
    held = sk.held()
    return Result(not held, "still holding after opening" if held else "", {**r.data, "held": held})
