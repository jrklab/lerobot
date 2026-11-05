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

"""Layer 3, actions: readable names for common uses of the primitives (primitives.py). They hold
no numbers of their own (tests/test_geometry.py, test_brain_code_holds_no_stray_numbers): every value is a
params.py or geometry.py one, so a new action adds a name, not a parameter to calibrate."""

from __future__ import annotations

import numpy as np
from common.vision import brick_angle, snap_top

from brain import params as prm
from brain.geometry import (
    MM_PER_M,
    front_object_height,
    front_object_width,
    grasp_spot,
    locate_from_one_view,
    look_offsets,
    lowest_view_height,
    view_half_width,
    wrap_deg,
)
from brain.goals import (
    AngleGoal,
    BesideGoal,
    FrontApproachGoal,
    ReachGoal,
    placed_beside,
    round_centre,
)
from brain.params import Param
from brain.primitives import Result, Skills


def land(sk: Skills, obj: str | None = None) -> Result:
    """Down to GRASP_LIFT_M above the predicted floor without pressing into it (v1's touch
    pushed into the floor and slid 8-10 mm, Oct 27); the contact stop stays as the safety. A
    contact more than half the object's height above the floor may be its edge: refused (its
    measured height, Skills.object_height). From higher up it first goes freely to the last
    look height: the Pi's contact watch learns the arm's
    droop at the start, and over a 10 cm descent the droop changes by more than its 1.5 mm
    threshold (Oct 27: "held up" 83 mm above the floor, runs 20261027T181003 and later -v2-land)."""
    look = sk.final_look_height(obj)
    if sk.height() > look:
        r = sk.move("arm", height=look)
        if not r.ok:
            return r
    r = sk.move("arm", height=prm.GRASP_LIFT_M.value, until_contact=True)
    if not r.ok or not r.data.get("contact"):
        return r
    above = r.data["contact_z"] - r.data["floor"]
    sk.frame("wrist", "contact")  # what the jaws met, for the record (Oct 27 trial 5: 11 mm up)
    sk.ev("land_contact", above_floor_m=round(above, 4), detector=r.data.get("detector"))
    if above > sk.object_height(obj) / 2:
        return Result(False, f"met something {above:.3f} m above the floor", r.data)
    # The floor, a little higher than predicted: back up to the grasp lift above it.
    return sk.move("arm", delta=(0.0, 0.0, r.data["contact_z"] + prm.GRASP_LIFT_M.value - sk.tcp()[2]))


def measure_width(sk: Skills, obj: str) -> dict | None:
    """The object's width from the last wrist look (the final look, the jaws over it): its
    extent along the jaws' closing direction (the image's horizontal, any residual angle
    included), its short side (what squeezing jaws close to) and its long side, all on its
    measured top (m).
    None when the look has no brick-shaped blob."""
    det = sk.seen.get(f"wrist:{obj}")
    m = None if det is None else brick_angle(det.jpeg, det.box, look=sk.looks.get(f"wrist:{obj}"))
    if m is None:
        return None
    h, top = sk.height(), sk.object_height(obj)
    return {
        "extent_m": sk.geom.wrist_distance(*m["extent_u"], h, top, sk.roll),
        "short_m": sk.geom.wrist_distance(*m["across"], h, top, sk.roll),
        "long_m": sk.geom.wrist_distance(*m["along"], h, top, sk.roll),
    }


def open_to_fit(sk: Skills, obj: str) -> Result:
    """Open the jaws to the object before landing on it (Nov 4; the demo's one opening before):
    its measured extent across the jaws, the taught gap on the fixed jaw's side, and on the
    moving jaw's the servo's tolerance and the width's error, each either way (the owner: room
    to spare, Nov 4). Refused when that is wider than the gripper opens; unmeasured, the jaws
    stay as they are, noted."""
    w = measure_width(sk, obj)
    sk.grasp_width = None if w is None else w["short_m"]
    if w is None:
        sk.ev("grasp_width", obj=obj, measured=False)
        return Result(True, "width not measured: the jaws stay as they are")
    top = sk.object_height(obj)
    room = 2 * (prm.SERVO_TOL_M.value + prm.WIDTH_ERROR_M.value)
    mm = MM_PER_M * (w["extent_m"] + sk.geom.fixed_jaw_gap(top) + room)
    sk.ev("grasp_width", obj=obj, measured=True, extent_mm=round(MM_PER_M * w["extent_m"], 1),
          short_mm=round(MM_PER_M * w["short_m"], 1), open_mm=round(mm, 1))
    if mm > sk.geom.gripper_max_mm():
        return Result(False, f"the {obj} needs the jaws {mm:.0f} mm open: wider than they open")
    return sk.grip("open", width=mm)


def close(sk: Skills) -> Result:
    """Close the jaws on the object, noting how far it reaches below them: the jaw tips' height
    above the floor it stands on (what the release and the carry clear by). Missed when they
    close well past its measured width (open_to_fit), more than WIDTH_ERROR_M twice over:
    on an edge, or on something else."""
    sk.held_below = max(sk.height(), 0.0)
    r = sk.grip("close")
    if not r.ok or sk.grasp_width is None or r.data.get("mm") is None:
        return r
    least = MM_PER_M * (sk.grasp_width - 2 * prm.WIDTH_ERROR_M.value)
    if r.data["mm"] < least:
        return Result(False, f"closed to {r.data['mm']} mm, past the measured width "
                             f"({MM_PER_M * sk.grasp_width:.1f} mm): missed it", r.data)
    return r


def hang(sk: Skills) -> float:
    """How far the held object reaches below the jaw tips (Skills.held_below), the grasp's own
    height if no close noted it."""
    return sk.held_below if sk.held_below is not None else prm.GRASP_LIFT_M.value


def carry_height(sk: Skills) -> float:
    """The jaw tips' height for carrying: the held object's bottom PLACE_CLEARANCE_M over the
    tallest object measured so far (the place's target among them), at least the hover height
    (the arm out of the front camera's floor view, carry())."""
    tallest = max(sk.heights.values(), default=0.0)
    return max(prm.HOVER_HEIGHT_M.value, tallest + hang(sk) + prm.PLACE_CLEARANCE_M.value)


def lift(sk: Skills) -> Result:
    """Straight up to the carry height (Nov 4: was the demo's fixed lift), or, where the vertical
    gripper can't reach that high (at 27 cm out it goes ~11 cm up; Oct 27 v2 pick), by half as
    much again until it can, but not less than the final look height; then whether the object
    is still held. The planner refuses before anything moves, so a try costs nothing."""
    up = carry_height(sk) - sk.height()
    r = sk.move("arm", delta=(0.0, 0.0, up))
    while not r.ok and up / 2 >= prm.FINAL_LOOK_HEIGHT_M.value:
        up /= 2
        r = sk.move("arm", delta=(0.0, 0.0, up))
    if not r.ok:
        return r
    held = sk.held()
    return Result(held, "" if held else "nothing held after the lift", {**r.data, "held": held})


def look_around(sk: Skills, obj: str) -> Result:
    """Turn the base until the front camera finds `obj`: alternately left and right in widening
    steps of (1 - LOOK_OVERLAP) of the view's width, all the way round; then back up
    LOOK_BACK_M once (it may be too close to see) and turn round again. The base ends facing
    the view it was found in. Oct 27 trial: the brick wasn't in the single forward view."""
    det = sk.find(obj, "front")
    if det is not None:
        return Result(True, data={"turned": 0.0, "det": det})
    step = (1 - prm.LOOK_OVERLAP.value) * sk.geom.front_view_deg()  # share of the view's width, in deg
    for backed in (False, True):
        if backed:
            r = sk.move("base", delta=(-prm.LOOK_BACK_M.value, 0.0))
            if not r.ok:
                return r
            det = sk.find(obj, "front")
            if det is not None:
                return Result(True, data={"turned": 0.0, "backed": True, "det": det})
        at = 0.0
        for off in look_offsets(step):
            # The short way round: from 163 deg right to facing backwards is 17 deg, not 343
            # (Oct 28: asked for 343 and refused).
            r = sk.move("base", roll=wrap_deg(off - at))
            if not r.ok:
                return r
            at = off
            det = sk.find(obj, "front")
            if det is not None:
                return Result(True, data={"turned": at, "backed": backed, "det": det})
        r = sk.move("base", roll=-at)  # face the start again before backing up or giving up
        if not r.ok:
            return r
    return Result(False, f"no {obj} in a full look around, nor after backing up")


def approach(sk: Skills, obj: str, tol=None) -> Result:
    """Base: find the object (looking around if it isn't in the forward view), then turn and
    drive until the front camera sees it at the reach point. The first drive is measured on the
    main model's box, the look that found it, not a second look a second later: from 0.5 m away
    one pixel of the box's bottom is ~1.5 cm, and the fast model's box sat 7 px higher than the
    main model's of the same view, the drive went 9 cm too far and lost the brick (Oct 29,
    run 20261029T071804)."""
    r = look_around(sk, obj)
    if r.ok:
        first = r.data["det"] if r.data["det"].by == "main" else sk.find(obj, "front", fast_ok=False)
        r = sk.servo(obj, FrontApproachGoal(tol), first=first)
    det = sk.seen.get(f"front:{obj}")
    if r.ok and det is not None:  # side-on, for placing into it later (front_object_height)
        box, how = snap_top(det.jpeg, det.box)  # the box's top up to a rim it left out (Nov 4)
        sk.heights[obj] = front_object_height(sk.geom, box)
        sk.widths[obj] = front_object_width(sk.geom, box)
        sk.ev("object_height", obj=obj, height_m=round(sk.heights[obj], 4), width_m=round(sk.widths[obj], 4),
              top=how, unsnapped_m=round(front_object_height(sk.geom, det.box), 4))
    return r


def hover(sk: Skills) -> Result:
    """The open, vertical gripper over the reach point at the hover height, roll 0."""
    r = sk.move("arm", xy=(prm.REACH_POINT_X_M.value, 0.0), height=prm.HOVER_HEIGHT_M.value, roll=0.0)
    return sk.grip("open") if r.ok else r


def line_up(sk: Skills, obj: str) -> Result:
    """Base, with the gripper hovering: the object within the arm's reach of the grasp spot."""
    return sk.servo(obj, ReachGoal())


def align(sk: Skills, obj: str) -> Result:
    """Wrist roll: the object's long side across the jaws."""
    return sk.servo(obj, AngleGoal())


def carry(sk: Skills) -> Result:
    """Hold the object over the reach point at the hover height before the base drives: held low
    after the pick's lift (~6 cm at 27 cm out) the brick caught a cup's rim and pushed the cup
    ahead of the robot (owner, Oct 28). Folding instead cost ~10 s a task (fold, then unfold for
    the place); at the hover the arm is out of the front camera's floor view, its jaw tips just
    in the frame's top above the horizon (Oct 28 check), where find() takes no box. Higher if
    a taller object was measured (carry_height)."""
    return hover_at(sk, carry_height(sk))


def hover_at(sk: Skills, height: Param | float) -> Result:
    """The vertical gripper over the reach point at `height`, roll 0 (jaws as they are)."""
    h = height.value if isinstance(height, Param) else float(height)
    return sk.move("arm", xy=(prm.REACH_POINT_X_M.value, 0.0), height=h, roll=0.0)


def place_height(sk: Skills, target: str, which: str) -> float:
    """The jaw tips' height for the place's "release" or "hover" over `target`, from what was
    measured (Nov 4; the demo's offsets before):
      release  the target's top (the approach's front camera) + how far the held object reaches
               below the jaws (the close) + PLACE_CLEARANCE_M: its bottom just over the top
      hover    the lowest height the target's whole width fits in the wrist view (the approach's
               front camera), at least LOOK_CLEARANCE_M over the release
    The demo's heights for the same target are logged beside them, to compare. Without a
    measured height, the demo's absolute heights (made for its one container), noted."""
    h = sk.heights.get(target)
    if h is None or h <= 0.0:
        absolute = prm.RELEASE_ABOVE_FLOOR_M if which == "release" else prm.PLACE_HOVER_M
        sk.ev("place_height_absolute", target=target, which=which, height_m=absolute.value,
              why="the target's height wasn't measured")
        return absolute.value
    out = h + hang(sk) + prm.PLACE_CLEARANCE_M.value
    if which == "hover":
        out += prm.LOOK_CLEARANCE_M.value
        width = sk.widths.get(target, 0.0)
        if width > 0.0:
            out = lowest_view_height(sk.geom, width, h, out)
    offsets = prm.demo_place_offsets()
    sk.ev("place_height", target=target, which=which, height_m=round(out, 4),
          demo_m=None if not offsets else round(h + offsets[which], 4))
    return out


def bring_into_view(sk: Skills, container: str) -> Result:
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


def over_container(sk: Skills, container: str) -> Result:
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


def beside(sk: Skills, ref: str) -> Result:
    """Base, then arm: the reference object just beside where the held one will land."""
    r = sk.servo(ref, BesideGoal(tol_m=prm.REACH_TOL_M), part="base")
    return sk.servo(ref, BesideGoal(tol_m=prm.REACH_TOL_M)) if r.ok else r  # coarse here; precise at 3 cm


def lower_to_release(sk: Skills, target: str | None = None) -> Result:
    """Straight down to the release height over `target` (place_height); stops early if the held
    object meets something (the container's bottom)."""
    h = place_height(sk, target, "release") if target else prm.RELEASE_ABOVE_FLOOR_M.value
    return sk.move("arm", height=h, until_contact=True)


def release(sk: Skills) -> Result:
    """Open the jaws and check they hold nothing any more."""
    r = sk.grip("open")
    if not r.ok:
        return r
    held = sk.held()
    return Result(not held, "still holding after opening" if held else "", {**r.data, "held": held})


def check_in(sk: Skills, obj: str, container: str) -> Result:
    """After the release: is the object on the floor in the front camera? Then it missed the
    container. The front camera looks across the floor from ~4 cm up, so an object inside a
    container taller than that is hidden by its wall, and a miss lands around the reach point,
    which it sees; the arm at the place height is out of its floor view (Oct 28 carry check).
    A miss hidden behind the container passes. The same find() as the approach, nothing else
    (owner, Oct 29: the wrist pixel counts had grown too hand-crafted)."""
    det = sk.find(obj, "front", fast_ok=False)
    if det is not None and sk.widths.get(container, 0.0) > 0.0:
        # A miss lands by the container: one let go just over its rim falls within its width of
        # where it was let go (the reach point). Further off it's something else (Nov 4: a
        # basketball 2 m away boxed as the red block, and the retry drove for it).
        p = sk.geom.front_point(((det.box[0] + det.box[2]) / 2, det.box[3]))
        off = np.inf if p is None else float(np.hypot(p[0] - prm.REACH_POINT_X_M.value, p[1]))
        if off > sk.widths[container]:
            sk.ev("place_check", obj=obj, target=container, on_floor=False, box=det.box,
                  far_m=None if p is None else round(off, 3), why="seen too far from the release to be it")
            return Result(True)
    if det is not None and on_container(sk, det, container):
        return Result(True)
    sk.ev("place_check", obj=obj, target=container, on_floor=det is not None, box=None if det is None else det.box)
    if det is not None:
        return Result(False, f"the {obj} is on the floor, not in the {container}", {"box": det.box})
    return Result(True)


def on_container(sk: Skills, det, container: str) -> bool:
    """Whether the object's box in the check's frame is the container's own face: mostly inside
    the container's box (ON_CONTAINER_SHARE) and not reaching below its bottom, where an object
    fallen in front of it would (nearer the camera, lower in the image). The container asked of
    the main model in the same frame (Nov 4: the cup's green band boxed as the light green
    block)."""
    main = getattr(sk, "main", None)
    if main is None or not det.jpeg:
        return False
    c = main.box(det.jpeg, f"the {container} on the floor").value
    if c is None:
        return False
    x0, y0, x1, y1 = det.box
    area = (x1 - x0) * (y1 - y0)
    if area <= 0.0:
        return False
    share = max(0.0, min(x1, c[2]) - max(x0, c[0])) * max(0.0, min(y1, c[3]) - max(y0, c[1])) / area
    ok = share >= prm.ON_CONTAINER_SHARE.value and y1 <= c[3]
    if ok:
        sk.ev("place_check", obj=container, target=container, on_floor=False, box=det.box, container_box=c,
              share=round(share, 2), why="the box is on the container's face")
    return ok


def check_beside(sk: Skills, obj: str, ref: str) -> Result:
    """After the release, from the hover: the object beside the reference with about the
    intended gap (goals.placed_beside). Can't see both: passes, unchecked."""
    a, b = sk.find(obj, "wrist", fast_ok=False), sk.find(ref, "wrist", fast_ok=False)
    if a is None or b is None:
        sk.ev("place_check", obj=obj, target=ref, checked=False)
        return Result(True, f"couldn't see both the {obj} and the {ref} to check", {"checked": False})
    ok = placed_beside(a.box, b.box)
    sk.ev("place_check", obj=obj, target=ref, checked=True, ok=ok, box=a.box, ref_box=b.box)
    return Result(ok, "" if ok else f"the {obj} is not beside the {ref}", {"checked": True})


def put_down(sk: Skills) -> Result:
    """Set the held object back on the floor over the reach point: hover there, land, release,
    back up to the hover height."""
    for step in (lambda: hover_at(sk, prm.HOVER_HEIGHT_M), lambda: land(sk), lambda: release(sk),
                 lambda: hover_at(sk, prm.HOVER_HEIGHT_M)):
        r = step()
        if not r.ok:
            return r
    return Result(True)
