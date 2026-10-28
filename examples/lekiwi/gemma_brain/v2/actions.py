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

import numpy as np

from v2 import params as prm
from v2.geometry import grasp_spot, locate_from_two_heights, look_offsets
from v2.goals import AngleGoal, BesideGoal, FrontApproachGoal, ReachGoal, round_centre
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
    sk.frame("wrist", "contact")  # what the jaws met, for the record (Oct 27 trial 5: 11 mm up)
    sk.ev("land_contact", above_floor_m=round(above, 4), detector=r.data.get("detector"))
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


def look_around(sk: Skills2, obj: str) -> Result:
    """Turn the base until the front camera finds `obj`: alternately left and right in widening
    steps of (1 - LOOK_OVERLAP) of the view's width, all the way round; then back up
    LOOK_BACK_M once (it may be too close to see) and turn round again. The base ends facing
    the view it was found in. Oct 27 trial: the brick wasn't in the single forward view."""
    if sk.find(obj, "front") is not None:
        return Result(True, data={"turned": 0.0})
    step = (1 - prm.LOOK_OVERLAP.value) / sk.geom.front["u_per_deg"]  # share of the view's width, in deg
    for backed in (False, True):
        if backed:
            r = sk.move("base", delta=(-prm.LOOK_BACK_M.value, 0.0))
            if not r.ok:
                return r
            if sk.find(obj, "front") is not None:
                return Result(True, data={"turned": 0.0, "backed": True})
        at = 0.0
        for off in look_offsets(step):
            r = sk.move("base", roll=off - at)
            if not r.ok:
                return r
            at = off
            if sk.find(obj, "front") is not None:
                return Result(True, data={"turned": at, "backed": backed})
        r = sk.move("base", roll=-at)  # face the start again before backing up or giving up
        if not r.ok:
            return r
    return Result(False, f"no {obj} in a full look around, nor after backing up")


def approach(sk: Skills2, obj: str) -> Result:
    """Base: find the object (looking around if it isn't in the forward view), then turn and
    drive until the front camera sees it at the reach point."""
    r = look_around(sk, obj)
    return sk.servo(obj, FrontApproachGoal()) if r.ok else r


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


def bring_into_view(sk: Skills2, container: str) -> Result:
    """Base moves until the whole container is inside the wrist frame, aiming its visible box's
    centre at the spot under the gripper (where the taught brick's middle was). Coarse: the
    floor's Jacobian and SERVO_GAIN; the two-height location after it is the precise part. The
    main model only (Oct 28: with the cup out of view the fast one boxed the held brick as the
    cup). Stops if a move leaves the box further out of the frame than before. The approach
    parks a container by its near edge, so a cup 8 cm across sat ~4 cm past the reach point
    and ran off the frame at the place hover (Oct 28)."""
    worst = None
    for _ in range(int(prm.SERVO_MAX_LOOKS.value)):
        det = sk.find(container, "wrist", fast_ok=False)
        if det is None:
            return Result(False, f"no {container} in the wrist camera")
        x0, y0, x1, y1 = det.box
        out = max(-x0, 0) + max(-y0, 0) + max(x1 - 1, 0) + (x0 <= 0) + (y0 <= 0) + (x1 >= 1)
        if not out:  # the near side is hidden under the gripper anyway
            return Result(True)
        if worst is not None and out > worst:
            return Result(False, f"the {container} went further out of view", {"box": det.box})
        worst = out
        j = sk.geom.jacobian(sk.height(), sk.roll)
        # Moving the TCP by d moves a floor point's image by j @ d: bring the centre to the spot.
        d = prm.SERVO_GAIN.value * np.linalg.solve(j, grasp_spot(sk.geom) - np.array(det.center))
        # The base, not the arm: a container ~4 cm past the reach point is past the vertical
        # gripper's reach at the place hover (Oct 28: refused at x 0.28). It shares the arm's
        # x (forward) and y (left) axes, so the same move vector drives it.
        r = sk.move("base", delta=(float(d[0]), float(d[1])))
        if not r.ok:
            return r
    return Result(False, f"the {container} stays partly out of the wrist frame")


def over_container(sk: Skills2, container: str) -> Result:
    """Put the gripper over a stationary container (owner, Oct 27: it won't move): look from the
    place hover and from PLACE_LOOK_RISE_M higher, locate its opening centre and rim height
    from the two views (geometry.locate_from_two_heights), move over it once, without looking
    again. Looking while moving over it failed: from above, the cup overflows the frame."""
    r = bring_into_view(sk, container)
    if not r.ok:
        return r
    views = []
    for k in range(len(("low", "high"))):
        if k:
            r = sk.move("arm", delta=(0.0, 0.0, prm.PLACE_LOOK_RISE_M.value))
            if not r.ok:
                return r
        det = sk.find(container, "wrist", fast_ok=False)
        if det is None:
            return Result(False, f"no {container} in the wrist camera", {"look": k})
        x0, y0, x1, y1 = det.box
        if y0 <= 0 or x0 <= 0 or x1 >= 1:
            return Result(False, f"the {container} runs off the frame", {"box": det.box})
        views.append((round_centre(det.box), x1 - x0, sk.height()))
    (c1, w1, h1), (c2, w2, h2) = views
    delta, rim = locate_from_two_heights(sk.geom, c1, w1, h1, c2, w2, h2, sk.roll)
    sk.ev("located", obj=container, delta=[round(float(v), 4) for v in delta], rim_height=round(rim, 4))
    if not 0.0 <= rim < h1 or np.linalg.norm(delta) > prm.PLACE_MAX_OFFSET_M.value:
        # Not a container under the gripper: the boxes weren't of it (Oct 28: the held brick
        # boxed as the cup gave a rim 0.44 m below the floor and a 27 cm offset).
        return Result(False, f"the {container}'s measurement makes no sense (rim {rim:.3f} m, offset "
                             f"{np.linalg.norm(delta):.3f} m)", {"rim_height": rim, "delta": delta.tolist()})
    r = sk.move("arm", delta=(float(delta[0]), float(delta[1]), -prm.PLACE_LOOK_RISE_M.value))
    return Result(r.ok, r.reason, {**r.data, "rim_height": rim, "moved": delta.tolist()})


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


def put_down(sk: Skills2) -> Result:
    """Set the held object back on the floor over the reach point: hover there, land, release,
    back up to the hover height."""
    for step in (lambda: hover_at(sk, prm.HOVER_HEIGHT_M), lambda: land(sk), lambda: release(sk),
                 lambda: hover_at(sk, prm.HOVER_HEIGHT_M)):
        r = step()
        if not r.ok:
            return r
    return Result(True)
