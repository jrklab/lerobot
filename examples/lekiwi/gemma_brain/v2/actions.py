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
from v2.geometry import (
    front_object_height,
    grasp_spot,
    locate_from_one_view,
    look_offsets,
    view_half_width,
    wrap_deg,
)
from v2.goals import (
    AngleGoal,
    BesideGoal,
    FrontApproachGoal,
    ReachGoal,
    placed_beside,
    round_centre,
)
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
            # The short way round: from 163 deg right to facing backwards is 17 deg, not 343
            # (Oct 28: asked for 343 and refused).
            r = sk.move("base", roll=wrap_deg(off - at))
            if not r.ok:
                return r
            at = off
            if sk.find(obj, "front") is not None:
                return Result(True, data={"turned": at, "backed": backed})
        r = sk.move("base", roll=-at)  # face the start again before backing up or giving up
        if not r.ok:
            return r
    return Result(False, f"no {obj} in a full look around, nor after backing up")


def approach(sk: Skills2, obj: str, tol=None) -> Result:
    """Base: find the object (looking around if it isn't in the forward view), then turn and
    drive until the front camera sees it at the reach point."""
    r = look_around(sk, obj)
    if r.ok:
        r = sk.servo(obj, FrontApproachGoal(tol))
    det = sk.seen.get(f"front:{obj}")
    if r.ok and det is not None:  # side-on, for placing into it later (front_object_height)
        sk.heights[obj] = front_object_height(sk.geom, det.box)
        sk.ev("object_height", obj=obj, height_m=round(sk.heights[obj], 4))
    return r


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


def carry(sk: Skills2) -> Result:
    """Hold the object over the reach point at the hover height before the base drives: held low
    after the pick's lift (~6 cm at 27 cm out) the brick caught a cup's rim and pushed the cup
    ahead of the robot (owner, Oct 28). Folding instead cost ~10 s a task (fold, then unfold for
    the place); at the hover the arm is out of the front camera's floor view, its jaw tips just
    in the frame's top above the horizon (Oct 28 check), where find() takes no box."""
    return hover_at(sk, prm.HOVER_HEIGHT_M)


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
        # One more look before giving up: the main model sometimes returns no box for a cup
        # close up that is plainly there (Oct 28, first full task: failed the step).
        det = sk.find(container, "wrist", fast_ok=False, rim=True) or sk.find(container, "wrist", fast_ok=False, rim=True)
        if det is None:
            return Result(False, f"no {container} in the wrist camera")
        x0, y0, x1, y1 = det.box
        out = max(-x0, 0) + max(-y0, 0) + max(x1 - 1, 0) + (x0 <= 0) + (y0 <= 0) + (x1 >= 1)
        if not out:  # the near side is hidden under the gripper anyway
            return Result(True, data={"det": det})
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
    """Put the gripper over a stationary container (owner, Oct 27: it won't move): bring it
    wholly into the wrist view with the base, locate its opening's centre from one look with
    its rim height as the front camera measured it during the approach, and move over it once
    without looking again. Two looks 2.5 cm apart gave rim heights from -1.5 to 5.2 cm for the
    same cup (box widths vary by a few % between looks); the front camera gave 4.4 cm +/-0.25
    in 18 views (Oct 28)."""
    rim = sk.heights.get(container)
    if rim is None:
        return Result(False, f"no side-on measurement of the {container}'s height (the approach makes it)")
    for _ in range(int(prm.SERVO_MAX_LOOKS.value)):
        r = bring_into_view(sk, container)
        if not r.ok:
            return r
        # Its look, not another: a second look a second later got no box for a cup filling the
        # view, twice in one task (Oct 28, run 20261028T234620).
        det = r.data["det"]
        h = sk.height()
        delta = locate_from_one_view(sk.geom, round_centre(det.box), h, rim, sk.roll)
        sk.ev("located", obj=container, delta=[round(float(v), 4) for v in delta], rim_height=round(rim, 4))
        if np.linalg.norm(delta) > view_half_width(sk.geom, h, rim):
            # Not a container in view: the box wasn't of it (Oct 28: the held brick boxed as the cup).
            return Result(False, f"the {container} measures {np.linalg.norm(delta):.3f} m off, outside the view",
                          {"delta": delta.tolist()})
        if np.linalg.norm(delta) <= prm.REACH_TOL_M.value:
            r = sk.move("arm", delta=(float(delta[0]), float(delta[1]), 0.0))
            return Result(r.ok, r.reason, {**r.data, "rim_height": rim, "moved": delta.tolist()})
        # Further than the arm should reach vertically from here: the base takes it, look again.
        r = sk.move("base", delta=(float(delta[0]), float(delta[1])))
        if not r.ok:
            return r
    return Result(False, f"couldn't get over the {container} in {int(prm.SERVO_MAX_LOOKS.value)} looks")


def beside(sk: Skills2, ref: str) -> Result:
    """Base, then arm: the reference object just beside where the held one will land."""
    r = sk.servo(ref, BesideGoal(tol_m=prm.REACH_TOL_M), part="base")
    return sk.servo(ref, BesideGoal(tol_m=prm.REACH_TOL_M)) if r.ok else r  # coarse here; precise at 3 cm


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


def check_in(sk: Skills2, obj: str, container: str) -> Result:
    """After the release: is the object on the floor in the front camera? Then it missed the
    container. The front camera looks across the floor from ~4 cm up, so an object inside a
    container taller than that is hidden by its wall, and a miss lands around the reach point,
    which it sees; the arm at the place height is out of its floor view (Oct 28 carry check).
    A miss hidden behind the container passes. The same find() as the approach, nothing else
    (owner, Oct 29: the wrist pixel counts had grown too hand-crafted)."""
    det = sk.find(obj, "front", fast_ok=False)
    sk.ev("place_check", obj=obj, target=container, on_floor=det is not None, box=None if det is None else det.box)
    if det is not None:
        return Result(False, f"the {obj} is on the floor, not in the {container}", {"box": det.box})
    return Result(True)


def check_beside(sk: Skills2, obj: str, ref: str) -> Result:
    """After the release, from the hover: the object beside the reference with about the
    intended gap (goals.placed_beside). Can't see both: passes, unchecked."""
    a, b = sk.find(obj, "wrist", fast_ok=False), sk.find(ref, "wrist", fast_ok=False)
    if a is None or b is None:
        sk.ev("place_check", obj=obj, target=ref, checked=False)
        return Result(True, f"couldn't see both the {obj} and the {ref} to check", {"checked": False})
    ok = placed_beside(a.box, b.box)
    sk.ev("place_check", obj=obj, target=ref, checked=True, ok=ok, box=a.box, ref_box=b.box)
    return Result(ok, "" if ok else f"the {obj} is not beside the {ref}", {"checked": True})


def put_down(sk: Skills2) -> Result:
    """Set the held object back on the floor over the reach point: hover there, land, release,
    back up to the hover height."""
    for step in (lambda: hover_at(sk, prm.HOVER_HEIGHT_M), lambda: land(sk), lambda: release(sk),
                 lambda: hover_at(sk, prm.HOVER_HEIGHT_M)):
        r = step()
        if not r.ok:
            return r
    return Result(True)
