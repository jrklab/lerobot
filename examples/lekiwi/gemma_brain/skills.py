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

"""Composite skills: Gemma says what and where, code does the geometry, every step closes on
a sensor (the wrist camera, the measured joints, the gripper's stall).

Grasp, from a top-down pose above the block:
  1. look: Gemma yes/no "is there a brick?", then its box (a box alone is not trusted: with
     no brick in view Gemma still returns one);
  2. centre: move in x/y until the block sits on the taught grasp pixel for this height
     (calib.json wrist_grasp_pixel), at 10, 6 and 3 cm;
     (heights above the floor predicted for that spot: floor_at, no probe);
  3. touch: step down until the floor holds the jaws up (measured TCP above its free-air
     sag), refusing if the touch comes much higher than the predicted floor (landed on the
     block); the touch is added to the floor fit;
  4. lift GRASP_LIFT_M off the floor, close, read the stall;
  5. lift LIFT_M; held = the jaws stalled on it and didn't close to empty, and the Pi's
     hold watch saw no drop (no camera: see lift_and_check).
Each skill returns a Result; every frame and number goes to the run's log.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import overhead
from arm import JAW_TIP_OFFSET_M, NAMED_POSES, Arm, PlanError, plan_ee_line
from gemma import Gemma
from robot import Robot
from vision import brick_angle, colour_matches, crop_around, plausible_brick, snap_box

CALIB = json.loads(Path(__file__).with_name("calib.json").read_text())
FLOOR_TCP_Z = CALIB["floor_tcp_z_m"]["value"]
J_WRIST = np.array(CALIB["wrist_image_jacobian"]["value"])  # d(u, v) / d(x, y), at J_HEIGHT
J_HEIGHT = 0.099
# Camera distance above the jaw tips along the descent: pixel motion per metre scales as
# 1 / (height + this). Fitted from how much the brick's box grows on logged descents
# (calibrate_descent.py: 0.065 m; the earlier guess was 0.06). A floor point's image offset
# from DESCENT_P0, the point a straight descent heads for, scales the same way.
CAM_ABOVE_TIPS_M = CALIB["wrist_descent"]["value"]["depth_m"]
DESCENT_P0 = np.array(CALIB["wrist_descent"]["value"]["p0"])
# The fixed jaw in the wrist image, the same in every frame (calibrate_jaws.py).
FIXED_JAW_INNER_U = CALIB["wrist_jaws"]["value"]["fixed_inner_u"]
FIXED_JAW_TIP_V = CALIB["wrist_jaws"]["value"]["fixed_tip_v"]
# Room left between the fixed jaw and the brick's side when aiming the grasp: the median
# sideways scatter of the last descent (3 cm look -> just before closing) against the descent
# model, from the logged picks (calibrate_descent.py). A brick landing within it of the aim
# still leaves the jaw beside it; the larger misses are for the check before closing.
GRASP_MARGIN_M = float(np.median(np.abs(CALIB["wrist_descent"]["last_descent_right_edge_mm"]))) / 1000
# The check before closing (step 5.3): the fixed jaw's gap to the brick's side, measured in
# the wrist frame with the jaws at the floor, must be 0..GRASP_GAP_MAX_M, else lift to the
# last servo height, re-aim by the measured landing error and land again (GRASP_REDOS times).
# From the 33 logged closes (Oct 26-27, gap corrected for the brick's height): square grips
# at 0.1-5.6 mm, tilted grips at 4.9-11.0 mm, empty closes spread over 0.1-11.7 mm.
GRASP_GAP_MAX_M = 0.005
# 0 (Oct 27): the redo's second touch stopped on a false contact ~24 mm up five times in a
# row (run 20261027T030658); the check stays, logged only, until the redo lands on the
# measured floor without a second touch.
GRASP_REDOS = 0
BRICK_HEIGHT_M = 0.0114  # 2x4 brick body, spec; its box in the wrist image is the top face
GRASP_PIXEL = [(p["height_m"], np.array(p["center"])) for p in CALIB["wrist_grasp_pixel"]["value"]]

PITCH = 90.0
# Wrist roll: the camera rides on the wrist, so rolling by r turns the image by -r (roll test
# 2026-10-23, runs/20261023T222148-roll-test). The grasp was taught with the brick's long
# side vertical in the image (jaws closing across the 16 mm side), so roll by the measured
# angle to get back there. The angle measurement is good to ~+/-15 deg; squeezing jaws square
# a brick up, so that's enough.
ROLL_TOL_DEG = 12.0
ROLL_TRIES = 2
# The roll range, owner's rule (2026-10-27): the wrist-roll joint within its rest position
# +/-ROLL_FROM_REST_DEG, and ROLL_OVERSHOOT_DEG inside the motor's calibrated range. Rest
# (+79 deg) is not the motor's midpoint, so this is not roll -90..+90 (the Oct 24 range,
# which at +90 turned the joint to -84 deg, 163 deg from rest, and the 3 cm look after the
# descent found the brick far from where the model put it: run 20261027T014954). Roll r
# turns the joint by about -r (roll 0 ~ joint +3 deg, -90 ~ +90), so the range is about
# roll -144..+14; roll_range() works it out from the calibration. Corrections after the first
# roll may go ROLL_OVERSHOOT_DEG past the ends rather than flip the wrist 180 deg. A need
# beyond that (the ~22 deg the range misses of the brick's 180 deg symmetry) rolls to the
# nearest end and grips up to ~11 deg off square; the base doesn't turn for the roll (owner,
# Oct 27: orbiting the brick to fix its angle chased the angle without end).
ROLL_FROM_REST_DEG = 90.0
ROLL_OVERSHOOT_DEG = 10.0
IMAGE_PX = np.array([640.0, 480.0])
# The open width before a grasp: 40 until the owner's Oct 27 demo opened to 24.6.
OPEN = 25.0
# The last stage, 1.5 cm (tips just above the 13 mm brick's studs), was added after the
# owner saw the jaws closing on the brick's end (runs 005149: after a blind descent from 3 cm
# the brick sat ~0.05 of the frame lower than taught, along its length).
# 2 cm, not 1.5: with the tips 2 mm over the studs, a few mm of floor error at the brick
# had the fixed jaw clip it on the way down and drag it (Oct 24 run 203916, green brick).
# 2 cm stage dropped Oct 27 with the floor probe: the floor is now predicted (floor_at), worst
# error 8.4 mm, and at 3 cm that still leaves the jaws >2 cm up, clear of the 11.4 mm brick.
SERVO_HEIGHTS = (0.10, 0.06, 0.03)
# Per-height tolerance (pixels): 10 cm is the coarse line-up, 3 cm sets the grasp.
# 1.5 cm: 20 px, not 10: that close the jaws partly hide the brick and Gemma's box jitters by
# 15-30 px frame to frame (Oct 24 run 170013 oscillated 15-29 px at 10); the along-length
# check before closing does the fine placement.
SERVO_TOL_PX = (20.0, 14.0, 12.0)
# Below this the centring corrects across the jaws only (see centre).
ACROSS_ONLY_BELOW_M = 0.025
SERVO_GAIN_LAST = 0.5  # gentler at the last stage, so noisy boxes don't make it overshoot
# At 6 cm and below, Gemma's box jumps more between looks (Oct 24: the 6 cm stage chased
# 0.54 <-> 0.68 swings and diverged): gentler, smaller steps there, and a box that lands more
# than SERVO_JUMP_PX from where the last move should have put it is asked for once more.
SERVO_LOW_HEIGHT_M = 0.065
SERVO_GAIN_LOW = 0.5
SERVO_MAX_STEP_LOW_M = 0.01
SERVO_JUMP_PX = 35.0
# Smaller steps than this don't move the arm (servo deadband: Oct 24 run 210758 sent six
# 1.4 mm steps at 2 cm and the image didn't change), so low-stage steps are at least this.
SERVO_MIN_STEP_M = 0.0025
# At the low stages a brick that a jaw is pushing moves with the gripper: two moves in a row
# that shift it less than SERVO_DRAG_RATIO of the predicted image motion (when that is over
# SERVO_DRAG_MIN_PX) stop the centring, lifted clear, so the pick is retried.
SERVO_DRAG_RATIO = 0.4
SERVO_DRAG_MIN_PX = 15.0
# A stage that may be skipped when the one above already ended within its tolerance.
SKIPPABLE_HEIGHTS = (0.06,)
# Speeds (m/s): moves well clear of the floor and the brick (the descents between centring
# stages, the lifts), and the small centring corrections. The floor touch and the moves right
# before closing stay at 0.008. Were 0.015-0.03.
FREE_EE_SPEED = 0.06
SERVO_EE_SPEED = 0.03
# Pause before a camera look: frames are already fetched as taken after the move finished
# (and the Pi only reports a move done once the arm has settled), so this is just margin.
LOOK_SETTLE_S = 0.15
SERVO_GAIN = 0.7
SERVO_MAX_STEP_M = 0.02
SERVO_ITERS = 6
# The floor touch: one contact-stop descent (see touch_floor), this fast, searching this far.
TOUCH_SPEED = 0.012
TOUCH_SEARCH_M = 0.06
# The floor under the brick is predicted, not probed (Oct 27; the probe beside the brick took
# ~10 s a pick): arm sag makes the touched floor depend on reach (-0.119 near to -0.147 m far
# out), so z = a + b x + c y + d r^2 is fitted to every floor touch so far (floor_touches.json;
# each grasp's touch is added). On 33 touches each predicted from the others: median 1.1 mm,
# worst 8.4 mm off. Below FLOOR_FIT_MIN_POINTS touches, the last touch's floor is used.
FLOOR_TOUCHES = Path(__file__).with_name("floor_touches.json")
FLOOR_FIT_MIN_POINTS = 8
# The grasp touch stopping this far above the predicted floor came down on the brick: between
# the prediction's worst error (8.4 mm) and the brick's height (11.4 mm).
ON_BRICK_M = 0.010
GRASP_LIFT_M = 0.004  # jaw tips this far above the floor when closing (block body is 11.4 mm)
PRE_CLOSE_MAX_FIX_M = 0.003
# Just before closing, the share of the brick's length beyond the jaw tips: aim for
# OUTSIDE_AIM, accept OUTSIDE_MIN..OUTSIDE_MAX, else slide the gripper along the brick's
# length by at most PRE_CLOSE_ALONG_MAX_M per try. Half outside = the jaws close on the
# brick's middle (owner, Oct 27: grab the middle; the demo's deeper 0.26 is not a target).
# The brick's box is often cut off by the image bottom just before closing, so the outside
# share is measured against its length seen at the last servo look, scaled down to grasp
# height; BRICK_LEN_V (teach frame) when that look didn't show all of it.
BRICK_LEN_V = 0.30
OUTSIDE_AIM, OUTSIDE_MIN, OUTSIDE_MAX = 0.5, 0.35, 0.65
PRE_CLOSE_ALONG_MAX_M = 0.006
PRE_CLOSE_ALONG_TRIES = 2
# 6 mm until Oct 26, when the owner saw an ~8 mm gap at the close (and picks missed with the
# moving jaw pushing the brick that far): 2 mm aims for ~4 mm real gap. Still open (Oct 27):
# 0 nearly landed the fixed jaw on the brick; landing with room and sliding over at the floor
# lifted the jaw off the floor (the slide used the commanded height) and missed.
GRASP_CLEAR_FIXED_JAW_M = 0.002
LIFT_M = 0.09  # was two lifts of 5 + 4 cm for the old image test; the same height in one
GRIPPER_EMPTY_MAX = 3.5  # closed to empty reads ~1.9; on the 2x4 brick's short side, 6.7
DEFAULT_TARGET = "red Lego brick"

# Approach on the front camera (2026-10-23 base check, runs/20261023T221427-base-check): a
# left turn (+theta) moves the brick right in the image by ~0.019 of the width per degree.
FRONT_U_PER_DEG = 0.019
# Image column of a brick straight ahead of the arm: 0.43 when one sat in arm reach at arm
# y ~0 (Oct 23 first frames), 0.49 for one ~1 m out (the camera is off the arm's axis); the
# wrist-camera stage fixes what's left.
FRONT_U_TARGET = 0.46
# A turn smaller than the Pi's BASE_THETA_TOL_DEG (2 deg) finishes without moving (run
# 224141: twelve 1.6 deg turns in a row, none moved), so turns are at least MIN_TURN_DEG and
# the column tolerance is about what such a turn can fix.
FRONT_U_TOL = 0.04
MIN_TURN_DEG = 3.0
# Bottom edge of the brick's box (image fraction) when the brick sat in arm reach (~0.96).
FRONT_NEAR_BOTTOM = 0.95
FRONT_MID_BOTTOM = 0.90
APPROACH_FAR_STEP_M = 0.10
APPROACH_NEAR_STEP_M = 0.04
APPROACH_MAX_STEPS = 20
# Wrist stage: where the arm lines up from, and how much of the residual the arm (rather
# than the base) takes.
# 0.17 m (was 0.22): the owner saw the arm reaching far for the brick; the base now lines
# the brick up under this nearer point (pointing straight down from 10 cm works at
# x 0.17-0.23), and the arm only takes up ARM_REACH_M of what's left.
ABOVE_XY = (0.17, 0.0)
ARM_REACH_M = 0.015
# After the front-camera approach (it can't stop much closer than ~24 cm: the brick drops out
# of the bottom of its view), drive this much further before looking down with the wrist
# camera, which lines up the rest.
CLOSE_IN_M = 0.06
# Scan: one full turn in SCAN_STEP_DEG stops (owner's choice, 2026-10-23: 30 deg, may change).
SCAN_STEP_DEG = 30.0
# Sightings of one object closer than this in bearing are the same candidate.
SCAN_MERGE_DEG = 12.0
# Front camera: a box whose bottom edge is above this row isn't something on the floor (0.74
# rejected a real cup ~1.5 m out at 0.725-0.74; boxes invented on a hat or coats sat <= 0.46).
FRONT_FLOOR_V = 0.70
# Rough range from where the box meets the floor: d ~ RANGE_K / (bottom - RANGE_HORIZON_V),
# fitted to the red brick's 10 cm approach steps (Oct 23). For the log and Gemma, not control.
RANGE_K = 0.043
RANGE_HORIZON_V = 0.70
RANGE_OFFSET_M = 0.0  # calibrate_front.py: K / (bottom - horizon) - this = distance beyond ABOVE_XY
# The values above are for the first camera mount (Oct 23-24). After the camera is moved,
# calibrate_front.py measures them again (calib.json "front_camera"), and those win.
if "front_camera" in CALIB:
    _fc = CALIB["front_camera"]["value"]
    FRONT_U_PER_DEG, FRONT_U_TARGET = _fc["u_per_deg"], _fc["u_target"]
    FRONT_FLOOR_V, RANGE_K, RANGE_HORIZON_V = _fc["floor_v"], _fc["range_k"], _fc["range_horizon_v"]
    FRONT_NEAR_BOTTOM, FRONT_MID_BOTTOM = _fc["near_bottom"], _fc["mid_bottom"]
    RANGE_OFFSET_M = _fc["range_offset_m"]


def front_range(bottom: float) -> float:
    """Rough distance (m) from the arm's base to where a box with this bottom row meets the
    floor: the old fit read it directly; calibrate_front.py's is beyond the reach point."""
    r = RANGE_K / max(0.005, bottom - RANGE_HORIZON_V)
    return r - RANGE_OFFSET_M + ABOVE_XY[0] if "front_camera" in CALIB else r
# Scan candidates are checked by asking E4B to classify a zoomed crop of the box among fixed
# options (16/17 right on Oct 23 crops: real cups, a backpack, a power strip, cases, chair
# legs; its yes/no and free descriptions were not usable). The object's category must win.
CLASS_OPTIONS = ["cup", "box", "bag", "cable", "furniture", "toy brick", "appliance", "nothing"]
CATEGORY_WORDS = {"cup": ("cup", "mug", "bowl", "container"), "toy brick": ("brick", "block", "lego")}
# Wrist-camera confirmation from the rest pose (it looks at the floor just ahead): E4B's
# yes/no was 12/12 there (3 phrasings x 4 frames, 1 with the cup), unlike on the far front view.
CONFIRM_TRIES = 3
# Overhead scan: the probe move that measures heading and scale (backward: what the scan
# looks for is usually ahead, and a forward probe nearly pushed the brick on Oct 23), the
# least image shift that counts as having seen it, and how close two kept boxes must be on
# the floor to be one object.
OVERHEAD_PROBE_M = -0.10
OVERHEAD_MERGE_M = 0.08
OVERHEAD_MIN_SHIFT_PX = 15.0
# Place: gripper tips this high above the floor over the container (a short paper cup, rim
# ~10-11 cm up on the play mat, 2026-10-23); the drop line-up treats the rim plane as the
# surface the camera looks at.
DROP_TIP_HEIGHT_M = 0.135
DROP_RIM_HEIGHT_M = 0.105
DROP_TOL_PX = 20.0
# Wrist roll for the drop, owner's rule (2026-10-24): not the pick's roll (a drop at roll 0
# missed the cup, run 010904) but the rest pose's wrist angle: skill roll -80 = joint +79 deg.
DROP_ROLL_DEG = -80.0
DROP_ITERS = 6
DROP_MAX_BOX_AREA = 0.8  # a container box bigger than this share of the wrist frame = not found
DROP_ARM_STEP_M = 0.05  # largest single arm correction while lining up over a container
FRONT_GUESS_DEPTH_M = 0.04  # from where an object meets the floor (its near edge) to its middle
# Release by opening only this much past the held position: the moving jaw swings wide when
# opened fully, and on Oct 23 (run 225755) it carried the brick out over the cup's rim.
RELEASE_OPEN_BY = 12.0
# place_next_to: line up over the reference object from NEXT_HOVER_M (jaw tips above the
# floor), then set the held object down NEXT_OFFSET_M to its side (centre to centre: a 2x4
# brick is 16 x 32 mm, the closed jaws ~32 mm across), lowering until the held object meets
# the floor (contact stop), searching at most NEXT_HOVER_M + 1 cm.
NEXT_HOVER_M = 0.08
NEXT_OFFSET_M = 0.05
NEXT_REF_HEIGHT_M = 0.011  # top of a brick on the floor: the plane the camera lines up on
NEXT_MAX_REF_SHIFT = 0.04  # image fraction: the reference moved less than this = undisturbed


@dataclass
class Result:
    ok: bool
    reason: str = ""
    data: dict = field(default_factory=dict)


class Log:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.f = (out / "log.jsonl").open("a")
        self.n = 0

    def ev(self, kind: str, **kw) -> None:
        rec = {"t": round(time.time(), 3), "ev": kind, **kw}
        self.f.write(json.dumps(rec, default=_jsonable) + "\n")
        self.f.flush()
        print(f"[{kind}] " + " ".join(f"{k}={_short(v)}" for k, v in kw.items()))

    def frame(self, tag: str, jpeg: bytes) -> str:
        self.n += 1
        name = f"{self.n:03d}_{tag}.jpg"
        (self.out / name).write_bytes(jpeg)
        return name


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.round(4).tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return str(v)


def _short(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) <= 4:
        return "[" + ", ".join(f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x) for x in v) + "]"
    return str(v)


def category(what: str) -> str:
    """The CLASS_OPTIONS entry an object name belongs to ("red Lego brick" -> "toy brick"),
    else its last word (then added to the options)."""
    words = what.lower().split()
    for cat, keys in CATEGORY_WORDS.items():
        if any(k in words for k in keys):
            return cat
    return words[-1] if words else what


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def grasp_pixel(height: float, extra_clear_m: float = 0.0) -> np.ndarray:
    """Where the brick's box centre should be in the wrist image at this gripper height: the
    taught position (brick flush against the fixed jaw), moved GRASP_CLEAR_FIXED_JAW_M away
    from the fixed jaw so that jaw comes down on the floor beside the brick, not on its edge
    (owner's observation, 2026-10-23 run 223236). The jaws close along image u at any roll
    (the camera turns with them), the fixed jaw on the right: the offset is a shift to the
    left by the image motion of that distance along the gripper's x at roll 0."""
    hs = [h for h, _ in GRASP_PIXEL]
    us = [c[0] for _, c in GRASP_PIXEL]
    vs = [c[1] for _, c in GRASP_PIXEL]
    taught = np.array([np.interp(height, hs, us), np.interp(height, hs, vs)])
    return taught - jacobian(height, 0.0)[:, 0] * (GRASP_CLEAR_FIXED_JAW_M + extra_clear_m)


def descend(point, h_from: float, h_to: float) -> np.ndarray:
    """Where a floor point seen at image `point` with the jaw tips h_from above the floor
    appears after a straight descent (or rise) to h_to."""
    s = (CAM_ABOVE_TIPS_M + h_from) / (CAM_ABOVE_TIPS_M + h_to)
    return DESCENT_P0 + (np.asarray(point, float) - DESCENT_P0) * s


def grasp_goal(height: float, shift_m: float = 0.0) -> np.ndarray:
    """(u of the brick's right edge, v of its centre) in the wrist image at this gripper
    height, such that a straight descent to the grasp leaves its right side GRASP_MARGIN_M
    from the fixed jaw's inner edge and its middle on the jaw tips' row (half the brick
    beyond the tips: the owner's "grab the middle"). Goals come from the measured jaw and
    the descent model, not from a taught brick, so they hold for any brick size. shift_m
    moves the aim that much closer to the fixed jaw (a re-aim after a measured landing)."""
    u_per_m = float(np.linalg.norm(jacobian(GRASP_LIFT_M)[0]))
    at_grasp = np.array([FIXED_JAW_INNER_U - (GRASP_MARGIN_M - shift_m) * u_per_m, FIXED_JAW_TIP_V])
    return descend(at_grasp, GRASP_LIFT_M, height)


def roll_range(calib) -> tuple[float, float]:
    """(min, max) skills roll keeping the wrist-roll joint within ROLL_FROM_REST_DEG of its
    rest position and ROLL_OVERSHOOT_DEG inside its calibrated range, from where the joint
    lands for rolls 0 and -90 over a brick straight ahead."""
    joint = "arm_wrist_roll"

    def deg(norm: float) -> float:
        return math.degrees(calib.to_rad(joint, norm))

    def joint_at(roll: float) -> float:
        wps, _ = plan_ee_line(calib, dict(NAMED_POSES["ready"]), [0.2, 0.0, -0.10], PITCH, roll, 0.04, -1.0)
        return deg(wps[-1][joint])

    j0 = joint_at(0.0)
    slope = (joint_at(-90.0) - j0) / -90.0  # joint degrees per degree of roll (about -1)
    rest = deg(NAMED_POSES["rest"][joint])
    lo, hi = sorted((deg(-100.0), deg(100.0)))
    jmin = max(rest - ROLL_FROM_REST_DEG, lo + ROLL_OVERSHOOT_DEG)
    jmax = min(rest + ROLL_FROM_REST_DEG, hi - ROLL_OVERSHOOT_DEG)
    return tuple(sorted(((jmin - j0) / slope, (jmax - j0) / slope)))


def grasp_gap_m(box) -> float:
    """The fixed jaw's gap to the brick's right side, in metres, from a wrist frame taken with
    the jaws at grasp height: the box is the brick's top face, BRICK_HEIGHT_M above the floor
    and so nearer the camera than the jaw tips; its edge is brought to the tips' depth before
    comparing with the jaw's inner edge. Negative: the brick reaches under the jaw."""
    tips = CAM_ABOVE_TIPS_M
    top = tips - (BRICK_HEIGHT_M - GRASP_LIFT_M)
    u_edge = DESCENT_P0[0] + (box[2] - DESCENT_P0[0]) * top / tips
    return float((FIXED_JAW_INNER_U - u_edge) / np.linalg.norm(jacobian(GRASP_LIFT_M)[0]))


def jacobian(height: float, roll_deg: float = 0.0) -> np.ndarray:
    """d(image u, v)/d(TCP x, y) at this height and wrist roll: the roll-0 measurement,
    scaled for height and rotated (in pixel units: the image isn't square) by the image's
    turn of -roll."""
    j = J_WRIST * (J_HEIGHT + CAM_ABOVE_TIPS_M) / (max(height, 0.0) + CAM_ABOVE_TIPS_M)
    a = math.radians(-roll_deg)
    # Clockwise-positive rotation in a v-down image.
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (rot @ (j * IMAGE_PX[:, None])) / IMAGE_PX[:, None]


def _floor_terms(x: float, y: float) -> list[float]:
    return [1.0, x, y, x * x + y * y]


def floor_at(x: float, y: float, fallback: float) -> float:
    """The floor (TCP z where the jaws meet it) predicted at TCP x, y from floor_touches.json;
    `fallback` until there are FLOOR_FIT_MIN_POINTS touches."""
    pts = json.loads(FLOOR_TOUCHES.read_text())["points"] if FLOOR_TOUCHES.exists() else []
    if len(pts) < FLOOR_FIT_MIN_POINTS:
        return fallback
    a = np.array([_floor_terms(p["x"], p["y"]) for p in pts])
    coef, *_ = np.linalg.lstsq(a, np.array([p["z"] for p in pts]), rcond=None)
    return float(np.array(_floor_terms(x, y)) @ coef)


def add_floor_touch(x: float, y: float, z: float, run: str) -> None:
    data = json.loads(FLOOR_TOUCHES.read_text()) if FLOOR_TOUCHES.exists() else {"points": []}
    data["points"].append({"x": round(x, 4), "y": round(y, 4), "z": round(z, 4), "how": "touch", "run": run})
    FLOOR_TOUCHES.write_text(json.dumps(data, indent=1) + "\n")


def measured_angle(jpeg: bytes, box, what: str = "") -> tuple[float, str]:
    """The brick's long-axis angle from image vertical, and how it was measured: the
    OpenCV fit when its blob is plausible, else the box's shape (vertical vs horizontal only)."""
    m = brick_angle(jpeg, box, what=what)
    if m is not None and 0.4 <= m["fill"] <= 1.15 and m["aspect"] >= 1.4:
        return m["angle_deg"], "fit"
    w_px = (box[2] - box[0]) * IMAGE_PX[0]
    h_px = (box[3] - box[1]) * IMAGE_PX[1]
    return (90.0 if w_px > 1.15 * h_px else 0.0), "box shape"


class Skills:
    def __init__(self, robot: Robot, arm: Arm, gemma: Gemma, log: Log, fast=None):
        self.robot, self.arm, self.gemma, self.log = robot, arm, gemma, log
        # A faster model for repeat looks at an object already found (plan_brain: E2B on
        # llama.cpp); None: every look asks `gemma`.
        self.fast = fast
        self.roll = 0.0  # current wrist roll target for top-down moves
        self.roll_min, self.roll_max = roll_range(arm.calib)
        self.target = DEFAULT_TARGET  # what pick() is after, as Gemma will be asked for it
        # The floor under the gripper (TCP z when the closed jaws touch it): predicted where the
        # gripper is about to work (floor_at), and set from each floor touch.
        self.floor_tcp = FLOOR_TCP_Z
        self.predicted_floor = None  # floor_at() for this grasp, the on-brick test's reference
        self.brick_len_v = None  # brick length in the image at grasp height, from the last servo look
        self.aim_shift_m = 0.0  # grasp_goal's re-aim after a measured landing (step 5.3)
        # The base can be locked (robot.base_enabled False): then the arm alone lines up.
        self.base_ok = getattr(robot, "base_enabled", True)
        # False: scan never turns the robot around; without the overhead camera it takes one
        # look straight ahead.
        self.turn_scan = True
        # Set (degrees): scan sweeps from -this (right) to +this (left) instead of a full turn.
        self.scan_sweep_deg = None

    # --- sensing ---

    def dropped(self) -> bool:
        """The Pi's hold watch saw the object slip out (auto_control: jaws closed past where
        they stalled on it). False when there's no hold to watch."""
        hold = self.robot.status().get("gripper_hold") or {}
        return hold.get("state") == "dropped"

    def look(self, jpeg: bytes, what: str, check=None):
        """Box of `what`, an object already found, on this frame: the fast model first, then
        `gemma` when the fast one finds nothing or its box fails `check(box) -> bool`. E2B
        boxes something even when the object isn't there (0 of 9 absent cases said so, Oct 26
        benchmark), so `check` is what catches a wrong fast box."""
        if self.fast is not None:
            a = self.fast.box(jpeg, what)
            if a.value is not None and (check is None or check(a.value)):
                a.by = "fast"
                return a
            self.log.ev("fast_miss", what=what, box=a.value, ms=round(a.ms))
        a = self.gemma.box(jpeg, what)
        a.by = "main"
        return a

    def tcp(self) -> np.ndarray:
        return self.arm.tcp()[:3, 3]

    def height(self) -> float:
        return float(self.tcp()[2] - self.floor_tcp)

    def wrist(self, tag: str, settle: float = LOOK_SETTLE_S) -> bytes:
        time.sleep(settle)
        jpeg, _ = self.robot.frame("wrist", after=time.time())
        self.log.frame(tag, jpeg)
        return jpeg

    def see_block(self, tag: str, check_present: bool = True, want_box: bool = False, allow_border: bool = False):
        """Centre (u, v) of the brick in a fresh wrist frame, or None if it isn't there.
        want_box: return (centre, box, jpeg) instead."""
        jpeg = self.wrist(tag)
        check = (lambda b: plausible_brick(jpeg, snap_box(jpeg, b, what=self.target)[0], self.target, allow_border)[0]) \
            if check_present else None
        a = self.look(jpeg, f"the {self.target} on the floor", check)
        self.log.ev("box", tag=tag, box=a.value, ms=round(a.ms), by=a.by)
        if a.value is None:
            return None
        box, how = snap_box(jpeg, a.value, what=self.target)
        if how == "snapped":
            self.log.ev("snap", tag=tag, box=[round(v, 4) for v in box])
        if check_present:
            # Not Gemma's yes/no: unreliable both ways on this brick (see vision.plausible_brick).
            ok, why = plausible_brick(jpeg, box, self.target, allow_border)
            self.log.ev("present", tag=tag, ok=ok, why=why)
            if not ok:
                return None
        x0, y0, x1, y1 = box
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        return (c, box, jpeg) if want_box else c

    # --- motion helpers ---

    def move_tcp(self, xyz, speed: float = 0.02, z_min: float | None = None) -> None:
        # Default floor guard for moves near the floor: below the session floor by the touch's
        # overshoot margin and the Pi's jaw-tip margin.
        z_min = self.floor_tcp - 0.015 - 0.017 if z_min is None else z_min
        self.arm.move_to(xyz, pitch_deg=PITCH, roll_deg=self.roll, speed=speed, z_min=z_min)

    def cmd_tcp(self) -> np.ndarray:
        return self.arm.calib.fk(self.arm.targets())[:3, 3]

    # --- skills ---

    def go_above(self, xy, height: float = SERVO_HEIGHTS[0]) -> Result:
        self.roll = 0.0
        # One clearance-checked joint move from wherever the arm is (usually rest); it used to
        # go rest -> "ready" -> a straight line down, ~18 s.
        self.brick_len_v = None
        self.aim_shift_m = 0.0
        self.floor_tcp = self.predicted_floor = floor_at(xy[0], xy[1], self.floor_tcp)
        self.arm.move_direct([xy[0], xy[1], self.floor_tcp + height], pitch_deg=PITCH, roll_deg=0.0)
        g = self.robot.gripper(OPEN)
        self.log.ev("above", tcp=self.tcp(), gripper=g["result"])
        return Result(True)

    def align_roll(self) -> Result:
        """Roll the wrist so the jaws close on the brick's two long faces (squeezing across its
        16 mm width: the most pad contact; the owner's rule, and the taught grasp), i.e. its
        long side vertical in the image. Image angle changes by -roll (roll test, Oct 23), and
        the brick looks the same every 180 deg, so the roll needed is the measured angle, or
        that minus 180. Rolls stay in roll_min..roll_max (roll_range); when neither fits, the
        nearest end of the range.
        Gripping across the 32 mm length (to roll less) was tried and missed (run 002847)."""
        for k in range(ROLL_TRIES + 1):
            got = self.see_block(f"roll_{k}", check_present=(k == 0), want_box=True)
            if got is None:
                return Result(False, "no brick in view to measure its angle")
            _, box, jpeg = got
            ang, how = measured_angle(jpeg, box, self.target)  # long side from image vertical, (-90, 90]
            self.log.ev("brick_angle", k=k, angle=ang, how=how, roll=self.roll)
            if abs(ang) <= ROLL_TOL_DEG or k == ROLL_TRIES:
                return Result(True, data={"roll": self.roll, "residual_deg": ang})
            options = [self.roll + ang, self.roll + ang - 180.0, self.roll + ang + 180.0]
            slack = ROLL_OVERSHOOT_DEG if k > 0 else 0.0
            lo, hi = self.roll_min, self.roll_max
            fits = [r for r in options if lo - slack <= r <= hi + slack]
            if fits:
                new = min(fits, key=lambda r: abs(r - self.roll))
            else:
                near = min(options, key=lambda r: min(abs(r - lo), abs(r - hi)))
                new = float(np.clip(near, lo, hi))
                self.log.ev("roll_clamped", needed=round(near, 1), roll=round(new, 1))
            p = self.cmd_tcp()
            self.roll = new
            self.move_tcp(p, speed=0.02)
        return Result(True)

    def centre(self, start: int = 0) -> Result:
        """Servo the brick onto grasp_goal at each SERVO_HEIGHTS, going down between: its
        right edge (the fixed jaw's side) and the centre row, not its box centre. start > 0:
        from that stage on, the roll kept (a re-aim after a landing)."""
        if start == 0:
            r = self.align_roll()
            self.log.ev("step", name="roll", ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return r
        last_err = None
        for i, h in enumerate(SERVO_HEIGHTS):
            if i < start:
                continue
            if i == 1:
                # The floor where the centring at 10 cm left the gripper, before going lower.
                p = self.cmd_tcp()
                self.floor_tcp = self.predicted_floor = floor_at(p[0], p[1], self.floor_tcp)
                self.log.ev("floor_predicted", xy=[round(float(p[0]), 4), round(float(p[1]), 4)], z=round(self.floor_tcp, 4))
            if h in SKIPPABLE_HEIGHTS and last_err is not None and last_err < SERVO_TOL_PX[i]:
                self.log.ev("servo_skip", h=h, last_err_px=round(last_err, 1))
                continue
            if i > 0 or start > 0:
                p = self.cmd_tcp()
                self.move_tcp([p[0], p[1], self.floor_tcp + h], speed=FREE_EE_SPEED)
            expect = None  # where the last correction should have put the brick in the image
            prev_c, stuck = None, 0
            for it in range(SERVO_ITERS):
                h_meas = self.height()
                got = self.see_block(f"servo_h{int(h * 100)}_{it}", check_present=False, want_box=True)
                if got is None:
                    return Result(False, f"no brick in view at {100 * h:.0f} cm")
                c = self._goal_point(got[1], h_meas)
                if expect is not None and np.linalg.norm((c - expect) * IMAGE_PX) > SERVO_JUMP_PX:
                    # Gemma's box jumped far from where the move should have put it (Oct 24:
                    # 0.54 <-> 0.68 of the frame between looks, the arm a few mm): ask once more
                    # and keep the box closer to the expectation.
                    got2 = self.see_block(f"servo_h{int(h * 100)}_{it}_recheck", check_present=False, want_box=True)
                    c2 = self._goal_point(got2[1], h_meas) if got2 is not None else None
                    self.log.ev("servo_recheck", first=c, second=c2, expected=expect)
                    if c2 is not None and np.linalg.norm(c2 - expect) < np.linalg.norm(c - expect):
                        c = c2
                if prev_c is not None and h <= SERVO_LOW_HEIGHT_M:
                    predicted = float(np.linalg.norm((expect - prev_c) * IMAGE_PX))
                    moved = float(np.linalg.norm((c - prev_c) * IMAGE_PX))
                    stuck = stuck + 1 if predicted > SERVO_DRAG_MIN_PX and moved < SERVO_DRAG_RATIO * predicted else 0
                    if stuck >= 2:
                        self.log.ev("servo_drag", h=h, predicted_px=round(predicted, 1), moved_px=round(moved, 1))
                        p = self.cmd_tcp()
                        self.move_tcp([p[0], p[1], self.floor_tcp + SERVO_HEIGHTS[0]], speed=FREE_EE_SPEED)
                        return Result(False, f"the brick moved with the gripper at {100 * h:.0f} cm: a jaw was pushing it")
                prev_c = c
                target = grasp_goal(h_meas, self.aim_shift_m)
                e = target - c
                last = h < ACROSS_ONLY_BELOW_M
                if last:
                    # Across the jaws (image u) only: the along-length placement is the slide
                    # before closing (center_along_length). Correcting v here slid the fixed
                    # jaw along the brick's side and failed picks that were already in place
                    # (Oct 24 runs 205433: u 0-9 px, v 15-31 px off; 210137).
                    e = np.array([e[0], 0.0])
                err_px = float(np.linalg.norm(e) * 640)
                self.log.ev("servo", h=h_meas, center=c, target=target, err_px=round(err_px, 1), goal="edge")
                last_err = err_px
                if err_px < SERVO_TOL_PX[i]:
                    break
                low = h <= SERVO_LOW_HEIGHT_M
                gain = SERVO_GAIN_LAST if last else SERVO_GAIN_LOW if low else SERVO_GAIN
                j = jacobian(h_meas, self.roll)
                d = gain * np.linalg.solve(j, e)
                n = float(np.linalg.norm(d))
                cap = SERVO_MAX_STEP_LOW_M if low else SERVO_MAX_STEP_M
                if n > cap:
                    d *= cap / n
                elif low and 0 < n < SERVO_MIN_STEP_M:
                    d *= SERVO_MIN_STEP_M / n
                expect = c + j @ d
                p = self.cmd_tcp()
                self.log.ev("servo_step", dxy=d)
                self.move_tcp([p[0] + d[0], p[1] + d[1], p[2]], speed=SERVO_EE_SPEED)
            else:
                return Result(False, f"did not converge at {100 * h:.0f} cm ({err_px:.0f} px off)")
        return Result(True)

    def _goal_point(self, box, h: float) -> np.ndarray:
        """(right edge u, centre v) of the brick's box: what grasp_goal aims. When the box
        shows the brick's whole length, remember that length scaled to grasp height, for the
        along-length check before closing."""
        x0, y0, x1, y1 = box
        if y0 > 0.01 and y1 < 0.99:
            self.brick_len_v = (y1 - y0) * (CAM_ABOVE_TIPS_M + h) / (CAM_ABOVE_TIPS_M + GRASP_LIFT_M)
        return np.array([x1, (y0 + y1) / 2])

    def touch_floor(self, search: float = TOUCH_SEARCH_M) -> Result:
        """One straight descent the Pi stops itself where the jaws meet the floor (a
        contact_stop trajectory: it watches the measured against the commanded TCP height every
        30 Hz tick), searching up to TOUCH_SEARCH_M below the start: no stored floor height is
        needed, so it works on a new floor. The height found becomes the session floor."""
        p = self.cmd_tcp()
        self.log.ev("touch_start", tcp=[round(float(v), 4) for v in self.tcp()], cmd=[round(float(v), 4) for v in p], joints=self.arm.targets())
        goal = [p[0], p[1], p[2] - search]
        z_min = goal[2] - 0.015 - 0.005  # the Pi checks the jaw tips, 1.5 cm below the TCP
        wps, durs = plan_ee_line(self.arm.calib, self.arm.targets(), goal, PITCH, self.roll, TOUCH_SPEED, z_min=goal[2] - 0.002)
        res = self.robot.trajectory(wps, durs, z_min=z_min, contact_stop=True)
        r = res.get("result", {})
        self.log.ev("touch", **r)
        st = self.robot.status()
        self.log.ev("touch_end", tcp=[round(float(v), 4) for v in self.tcp()], cmd=[round(float(v), 4) for v in self.cmd_tcp()], joints=st["joints"])
        if not r.get("contact"):
            return Result(False, f"no floor within {100 * search:.0f} cm below the start")
        z = float(r["z_meas"])
        if self.predicted_floor is not None and z > self.predicted_floor + ON_BRICK_M:
            # Stopped on the brick, not the floor: don't take it for the floor, don't close.
            self.log.ev("on_brick", z=round(z, 4), floor=round(self.predicted_floor, 4))
            p = self.cmd_tcp()
            self.move_tcp([p[0], p[1], self.predicted_floor + SERVO_HEIGHTS[0]], speed=FREE_EE_SPEED)
            return Result(False, f"a jaw came down on the brick ({1000 * (z - self.predicted_floor):.0f} mm above the floor)")
        add_floor_touch(float(p[0]), float(p[1]), z, self.log.out.name)
        free_gap = float(r.get("free_gap_mm", 0.0)) / 1000.0  # measured - commanded in free air (sag: negative)
        self.log.ev("floor", previous=round(self.floor_tcp, 4), now=round(z, 4), shift_mm=round(1000 * (z - self.floor_tcp), 1))
        self.floor_tcp = z
        return Result(True, data={"z": z, "z_cmd": z - free_gap, "how": r.get("how"), "start_xy": [float(p[0]), float(p[1])]})

    def center_along_length(self) -> list | None:
        """Just before closing: make sure part of the brick sticks out beyond the jaw tips
        (OUTSIDE_MIN..OUTSIDE_MAX of its length, aiming for OUTSIDE_AIM, i.e. its middle at
        the tips), so the jaws close on its middle with margin, not on one end with the rest
        deep inside the gripper. The wrist is rolled so the brick's long side is vertical
        (image v) and the jaws close along u; only the along-length error is corrected, so the
        gripper slides beside the brick and can't push it. Returns the last box seen (None if
        the brick wasn't boxed)."""
        for k in range(PRE_CLOSE_ALONG_TRIES + 1):
            got = self.see_block(f"pre_close_check_{k}", check_present=False, want_box=True)
            if got is None:
                self.log.ev("along", ok=False, why="brick not boxed before closing")
                return
            _, box, _ = got
            if box[1] < 0.02 or box[0] < 0.01 or box[2] > 0.99:
                # Not the brick: it is between the jaws, at the bottom of the frame or below
                # it (the camera zooms in on the way down). Oct 24 run 210909: Gemma boxed the
                # robot body in the top corner, "100% outside", and the slide took the jaws
                # 12 mm off a brick that was already between them. Close where it is.
                self.log.ev("along", k=k, ok=False, why="box at the top/side edge: not the brick", box=box)
                return
            length = self.brick_len_v or BRICK_LEN_V
            outside = float(np.clip((FIXED_JAW_TIP_V - box[1]) / length, 0.0, 1.0))
            self.log.ev("along", k=k, box=box, outside=round(outside, 2), length_v=round(length, 3))
            if OUTSIDE_MIN <= outside <= OUTSIDE_MAX or k == PRE_CLOSE_ALONG_TRIES:
                return box
            # More outside = the brick higher in the image (smaller v).
            ev = -(OUTSIDE_AIM - outside) * length
            h = max(self.height(), 0.0)
            j = jacobian(h, self.roll)
            # The TCP move whose image motion is (0, ev): along the brick's length only.
            d = np.linalg.solve(j, np.array([0.0, ev]))
            n = float(np.linalg.norm(d))
            if n > PRE_CLOSE_ALONG_MAX_M:
                d *= PRE_CLOSE_ALONG_MAX_M / n
            p = self.cmd_tcp()
            self.log.ev("along_step", k=k, dxy=d)
            self.move_tcp([p[0] + d[0], p[1] + d[1], p[2]], speed=0.008)

    def close(self) -> Result:
        g = self.robot.gripper(0, speed=30)
        res = g["result"]
        self.log.ev("close", **res)
        self.wrist("closed", settle=0.2)
        if not res.get("stalled"):
            return Result(False, f"closed fully ({res.get('pos')}) without stalling: nothing between the jaws", res)
        return Result(True, data=res)

    def lift_and_check(self) -> Result:
        """Lift the brick LIFT_M and decide held or not from the gripper motor: the close
        stalled on something (fetch's close step), the jaws haven't closed to empty, and the
        Pi's hold watch (auto_control) saw no drop on the way up. No camera: on Oct 26 the two
        lifted frames showed the brick held in the jaws, but E2B's box took in the jaws too and
        the image test called it "stayed behind", and the robot let go of a held brick. One
        wrist frame is logged for the record."""
        p = self.cmd_tcp()
        self.move_tcp([p[0], p[1], p[2] + LIFT_M], speed=FREE_EE_SPEED)
        self.wrist("lifted")
        g = self.robot.status()["joints"]["arm_gripper"]
        data = {"gripper": round(g, 1)}
        if g < GRIPPER_EMPTY_MAX or self.dropped():
            return Result(False, f"gripper closed to {g:.1f}: the brick slipped out", data)
        return Result(True, data=data)

    def see_front(self, tag: str, what: str, repeat: bool = False) -> tuple[np.ndarray, list] | None:
        """(u, v_bottom) and the box of `what` on the floor in a fresh front-camera frame, or
        None. Gemma's box is only accepted if it could be a thing on the floor: its bottom
        edge in the floor part of the image (with nothing there, Gemma has boxed a hat on a
        coat rack), and clear of the left/right border (it has boxed the rug at the edge).
        Gemma's yes/no isn't used: unreliable on these objects (see vision.plausible_brick).
        repeat: `what` was found a step ago, so the fast model may answer (see look())."""
        time.sleep(LOOK_SETTLE_S)
        jpeg, _ = self.robot.frame("front", after=time.time())
        self.log.frame(tag, jpeg)

        def accept(b) -> tuple[bool, str]:
            if not (b[3] > FRONT_FLOOR_V and b[0] > 0.01 and b[2] < 0.99):
                return False, "not on the floor or at the border"
            return colour_matches(jpeg, b, what)

        q = f"the {what} on the floor"
        a = self.look(jpeg, q, lambda b: accept(b)[0]) if repeat else self.gemma.box(jpeg, q)
        b = a.value
        ok, why = accept(b) if b is not None else (False, "no box")
        self.log.ev("front_box", tag=tag, what=what, box=b, accepted=ok, colour=why, ms=round(a.ms), by=a.by)
        if not ok:
            return None
        return np.array([(b[0] + b[2]) / 2, b[3]]), b

    def heading(self) -> float:
        """Base heading (deg, left positive) from the Pi's wheel odometry."""
        return float(self.robot.status()["odom"]["theta_deg"])

    def scan(self, objects: list[str]) -> Result:
        self.confirmed = None
        return self._scan(objects)

    def _scan(self, objects: list[str]) -> Result:
        """Find every object and keep a map of candidates (bearing, rough range, number of
        views), best first: from the overhead camera when one is attached and sees the robot,
        else by turning a full circle. approach() confirms each candidate up close."""
        r = self.scan_overhead(objects)
        if r is not None:
            return r
        if not self.turn_scan or not self.base_ok:
            # No turning (turn_scan off, or the base locked): one look straight ahead.
            return self.scan_rotate(objects, offsets=[0.0])
        if self.scan_sweep_deg:
            # A sweep from -sweep (right) to +sweep (left) in SCAN_STEP_DEG steps.
            n = int(round(self.scan_sweep_deg / SCAN_STEP_DEG))
            return self.scan_rotate(objects, offsets=[k * SCAN_STEP_DEG for k in range(-n, n + 1)])
        return self.scan_rotate(objects)

    def scan_overhead(self, objects: list[str]) -> Result | None:
        """The overhead-camera scan (see overhead.py): robot found, OVERHEAD_PROBE_M forward,
        robot found again; that move gives the heading and scale in the image. Returns None
        (use the rotation scan) when the camera isn't attached or can't measure the move."""
        if not overhead.external_camera_attached():
            self.log.ev("overhead", ok=False, why="no external camera attached")
            return None
        self.arm.reset()
        idx = getattr(self, "overhead_index", None)
        frame_a, box_a = None, None
        for i in [idx] if idx is not None else overhead.CANDIDATE_INDICES:
            f = overhead.grab(i)
            if f is None:
                continue
            b, inside = overhead.find_robot(self.gemma, f)
            self.log.ev("overhead_robot", index=i, box=b, inside=inside)
            if b is not None and inside:
                self.overhead_index, frame_a, box_a = i, f, b
                break
        if frame_a is None:
            self.log.ev("overhead", ok=False, why="the overhead camera does not see the whole robot")
            return None
        self.log.frame("overhead_a", overhead.encode(frame_a))
        res = self.robot.base(dx=OVERHEAD_PROBE_M)
        moved = float(np.hypot(*res["result"]["moved"][:2]))
        forward = OVERHEAD_PROBE_M > 0
        head = self.heading()
        frame_b = overhead.grab(self.overhead_index)
        box_b, inside = overhead.find_robot(self.gemma, frame_b) if frame_b is not None else (None, False)
        if box_b is None or not inside:
            self.log.ev("overhead", ok=False, why="lost the robot after the probe move")
            return None
        h, w = frame_b.shape[:2]
        px = np.array([w, h], dtype=float)
        d = (overhead.center(box_b) - overhead.center(box_a)) * px
        if not forward:
            d = -d  # the image vector of the robot's heading
        if np.linalg.norm(d) < OVERHEAD_MIN_SHIFT_PX:
            self.log.ev("overhead", ok=False, why=f"the robot's {100 * moved:.0f} cm move showed as only {np.linalg.norm(d):.0f} px")
            return None
        scale = moved / float(np.linalg.norm(d))  # m per pixel near the robot
        jpeg_b = overhead.encode(frame_b)
        self.log.frame("overhead_b", jpeg_b)
        self.log.ev("overhead_probe", moved_m=round(moved, 3), shift_px=np.round(d, 1), m_per_px=round(scale, 5), heading=head)
        rb = overhead.center(box_b) * px
        self.map = {}
        annotated = frame_b.copy()
        for o in objects:
            cands = []
            for f in overhead.find_boxes(self.gemma, frame_b, f"the {o} on the floor"):
                b = f["box"]
                if overhead.iou(b, box_b) > 0.25:
                    continue  # the robot itself (its box is tall with the folded arm: a small
                    # object just past the arm's tip sits inside it, so overlap, not containment)
                cls = self.classify(jpeg_b, b, o)
                keep = cls == category(o)
                v = overhead.center(b) * px - rb
                rel = overhead.relative_bearing(d, v)
                rng = float(np.linalg.norm(v)) * scale
                self.log.ev("overhead_box", what=o, box=b, views=f["views"], cls=cls, kept=keep, rel_deg=round(rel, 1), range_m=round(rng, 2))
                col = (0, 200, 0) if keep else (0, 0, 255)
                cv2.rectangle(annotated, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), col, 3)
                cv2.putText(annotated, f"{o[:12]}:{cls[:8]}", (int(b[0] * w), int(b[1] * h) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                if keep:
                    pos = v * scale
                    for c in cands:  # the same object boxed twice (full frame and a tile)
                        if np.linalg.norm(c["pos"] - pos) < OVERHEAD_MERGE_M:
                            c["views"] += [f] * f["views"]
                            break
                    else:
                        cands.append({"bearing": head + rel, "range_m": rng, "pos": pos, "views": [f] * f["views"], "tried": False})
            cands.sort(key=lambda c: (-len(c["views"]), c["range_m"]))
            self.map[o] = cands
        cv2.arrowedLine(annotated, tuple(int(x) for x in overhead.center(box_a) * px), tuple(int(x) for x in rb), (255, 0, 0), 4)
        self.log.frame("overhead_map", overhead.encode(annotated))
        return self._map_result(objects, "overhead camera")

    def _map_result(self, objects: list[str], method: str) -> Result:
        here = self.heading()
        summary = {
            o: [
                {"turn_to_face_deg": round(_wrap(c["bearing"] - here), 0), "about_m": round(c["range_m"], 1), "views": len(c["views"])}
                for c in self.map[o][:3]
            ]
            for o in objects
        }
        self.log.ev("scan_map", method=method, heading=here, map=summary)
        found = [o for o in objects if self.map[o]]
        missing = [o for o in objects if not self.map[o]]
        msg = f"({method}) " + "; ".join(
            f"{o}: {len(self.map[o])} candidate(s), best {summary[o][0]['turn_to_face_deg']:+.0f} deg, ~{summary[o][0]['about_m']} m"
            for o in found
        )
        if missing:
            msg += ("; " if found else "") + "not seen: " + ", ".join(missing)
        return Result(bool(found), msg, {"method": method, **summary})

    def scan_rotate(self, objects: list[str], offsets: list[float] | None = None) -> Result:
        """Look at each heading in `offsets` (degrees from where the robot faces now, left
        positive; default a full circle in SCAN_STEP_DEG stops, turning right), boxing every
        object at every look, and keep a map: for each object, its candidates (bearing, rough
        range, number of views), best first. Gemma's box is the only judge here, so a candidate
        can be wrong (E4B has boxed a backpack as a "short paper cup"); approach() confirms up
        close."""
        self.arm.reset()
        self.map = {o: [] for o in objects}
        sightings = {o: [] for o in objects}
        full = offsets is None
        if full:
            offsets = [-k * SCAN_STEP_DEG for k in range(int(round(360.0 / SCAN_STEP_DEG)))]
        at = 0.0  # where the robot faces now, relative to the start
        for k, off in enumerate(offsets):
            if abs(off - at) > 0.5:
                res = self.robot.base(dtheta_deg=off - at)
                self.log.ev("base", turn_deg=off - at, moved=res["result"].get("moved"))
                at = off
            head = self.heading()
            time.sleep(LOOK_SETTLE_S)
            jpeg, _ = self.robot.frame("front", after=time.time())
            self.log.frame(f"scan_{k:02d}", jpeg)
            for o in objects:
                a = self.gemma.box(jpeg, f"the {o} on the floor")
                b = a.value
                ok = b is not None and b[3] > FRONT_FLOOR_V and b[0] > 0.01 and b[2] < 0.99 and (b[2] - b[0]) < 0.6
                cls = None
                if ok:
                    ok, why = colour_matches(jpeg, b, o)
                    cls = None if ok else f"wrong colour ({why})"
                if ok:
                    cls = self.classify(jpeg, b, o)
                    ok = cls == category(o)
                if ok:
                    u = (b[0] + b[2]) / 2
                    bearing = head - (u - FRONT_U_TARGET) / FRONT_U_PER_DEG
                    rng = front_range(b[3])
                    sightings[o].append({"bearing": bearing, "range_m": rng, "stop": k, "box": b})
                self.log.ev("scan_box", k=k, heading=round(head, 1), what=o, box=b, cls=cls, accepted=ok, ms=round(a.ms))
        for o in objects:
            cands = []
            for sgt in sorted(sightings[o], key=lambda x: x["bearing"]):
                for c in cands:
                    if abs(_wrap(sgt["bearing"] - c["bearing"])) < SCAN_MERGE_DEG:
                        c["views"].append(sgt)
                        c["bearing"] = float(np.mean([v["bearing"] for v in c["views"]]))
                        break
                else:
                    cands.append({"bearing": sgt["bearing"], "views": [sgt]})
            for c in cands:
                c["range_m"] = float(np.median([v["range_m"] for v in c["views"]]))
                c["tried"] = False
            cands.sort(key=lambda c: (-len(c["views"]), c["range_m"]))
            self.map[o] = cands
        method = ("360-degree turn" if full else "one look straight ahead" if len(offsets) == 1
                  else f"sweep {offsets[0]:+.0f} to {offsets[-1]:+.0f} deg")
        return self._map_result(objects, method)

    def classify(self, jpeg: bytes, box, what: str) -> str:
        """Which CLASS_OPTIONS entry Gemma picks for the zoomed crop of `box` (lowercase)."""
        opts = list(CLASS_OPTIONS)
        if category(what) not in opts:
            opts.insert(0, category(what))
        a = self.gemma.ask_image(
            crop_around(jpeg, box),
            "Which one best describes the object in the middle of this picture? "
            f"Options: {', '.join(opts)}. Answer with one option.",
            max_tokens=6,
        )
        ans = a.text.lower().strip(" .")
        return next((o for o in opts if o in ans), ans)

    def map_key(self, what: str) -> str:
        """The scanned object name `what` refers to: Gemma may say "paper cup" after scanning
        for "short paper cup". Most shared words wins; no overlap keeps `what`."""
        names = list(getattr(self, "map", {}))
        words = set(what.split())
        best = max(names, key=lambda n: len(words & set(n.split())), default=None)
        return best if best and words & set(best.split()) else what

    def face(self, what: str) -> Result:
        """Turn to the best not-yet-rejected scan candidate for `what`."""
        what = self.map_key(what)
        cands = [c for c in getattr(self, "map", {}).get(what, []) if not c["tried"]]
        if not cands:
            known = getattr(self, "map", {}).get(what)
            return Result(False, f"no {'untried ' if known else ''}candidate for the {what}: scan first" if not known
                          else f"every candidate for the {what} was checked and rejected")
        c = cands[0]
        self.arm.reset()
        turn = _wrap(c["bearing"] - self.heading())
        while abs(turn) > MIN_TURN_DEG:
            step = float(np.clip(turn, -90, 90))
            self.robot.base(dtheta_deg=step)
            turn = _wrap(c["bearing"] - self.heading())
        self.current = (what, c)
        return Result(True, f"facing the {what} candidate seen in {len(c['views'])} view(s), ~{c['range_m']:.1f} m away",
                      {"views": len(c["views"]), "about_m": round(c["range_m"], 1)})

    def confirm(self, what: str) -> Result:
        """Close up, from the rest pose: the wrist camera looks at the floor just ahead."""
        self.arm.reset()
        jpeg = self.wrist(f"confirm_{what.replace(' ', '_')}")
        # A box and its colour, not a yes/no: Oct 26, "Is there a red block in this image?"
        # answered no six times with the brick in plain view (yes/no is biased to no; see the
        # Oct 26 benchmark), while boxes were right.
        a = self.gemma.box(jpeg, f"the {what} on the floor")
        ok, why = (False, "no box") if a.value is None else colour_matches(jpeg, a.value, what)
        self.log.ev("confirm", what=what, box=a.value, colour=why, ok=ok, ms=round(a.ms))
        return Result(ok, "" if ok else f"up close, the wrist camera does not see a {what} ({why})")

    def go_to(self, what: str) -> Result:
        """Face, approach and confirm `what`, moving on to the next scan candidate when the
        wrist camera says this one isn't it (up to CONFIRM_TRIES). Without a scan map, just
        approach whatever the front camera boxes."""
        what = self.map_key(what)
        self.confirmed = None
        self.closed_in = False
        if not self.base_ok:
            # Base locked: nothing to drive; the object has to be in reach already. Confirm with
            # the wrist camera from the rest pose, which looks at the floor just ahead.
            c = self.confirm(what)
            self.confirmed = what if c.ok else None
            return Result(c.ok, c.reason or f"base locked; the {what} is in front", {"base_locked": True})
        if not getattr(self, "map", {}).get(what):
            r = self.approach(what)
            if not r.ok:
                return r
            c = self.confirm(what)
            self.confirmed = what if c.ok else None
            return Result(c.ok, c.reason or "in reach and confirmed", r.data)
        reasons = []
        for _ in range(CONFIRM_TRIES):
            f = self.face(what)
            if not f.ok:
                break
            _, cand = self.current
            r = self.approach(what)
            if r.ok:
                c = self.confirm(what)
                if c.ok:
                    self.confirmed = what
                    return Result(True, f"in reach of the {what}, confirmed by the wrist camera", r.data)
                r = c
            cand["tried"] = True
            reasons.append(r.reason)
            self.log.ev("candidate_rejected", what=what, reason=r.reason)
        left = sum(not c["tried"] for c in self.map.get(what, []))
        return Result(False, f"could not reach a confirmed {what} ({'; '.join(reasons) or 'no candidates'}); "
                      f"{left} candidate(s) left", {"left": left})

    def approach(self, what: str | None = None) -> Result:
        """Drive until `what` is in arm reach, closing on the front camera: turn it to
        FRONT_U_TARGET, then step forward until its box's bottom edge (where it meets the
        floor) reaches FRONT_NEAR_BOTTOM. The arm stays at rest (folded, out of the front
        camera's view)."""
        what = what or self.target
        self.arm.reset()
        last_u = None
        for k in range(APPROACH_MAX_STEPS):
            got = self.see_front(f"approach_{k}", what, repeat=k > 0)
            if got is None:
                return Result(False, f"no {what} in the front camera")
            u, bottom = float(got[0][0]), float(got[0][1])
            if last_u is not None and abs(u - last_u) > 0.25:
                # A wild jump between frames is more likely a wrong box than a real move.
                got = self.see_front(f"approach_{k}_recheck", what)
                if got is None:
                    return Result(False, f"lost the {what} in the front camera")
                u, bottom = float(got[0][0]), float(got[0][1])
            last_u = u
            self.log.ev("approach", k=k, u=u, bottom=bottom)
            if abs(u - FRONT_U_TARGET) > FRONT_U_TOL:
                turn = -(u - FRONT_U_TARGET) / FRONT_U_PER_DEG
                turn = math.copysign(max(abs(turn), MIN_TURN_DEG), turn)
                res = self.robot.base(dtheta_deg=float(np.clip(turn, -30, 30)))
                self.log.ev("base", turn_deg=round(turn, 1), moved=res["result"].get("moved"))
                continue
            if bottom >= FRONT_NEAR_BOTTOM:
                return Result(True, data={"u": u, "bottom": bottom, "steps": k})
            step = APPROACH_FAR_STEP_M if bottom < FRONT_MID_BOTTOM else APPROACH_NEAR_STEP_M
            res = self.robot.base(dx=step)
            self.log.ev("base", dx=step, moved=res["result"].get("moved"))
        return Result(False, f"not in reach after {APPROACH_MAX_STEPS} steps")

    def line_up_base(self) -> Result:
        """With the gripper above ABOVE_XY, move the base so the brick lands within the arm's
        own servo range of the grasp pixel (the omni base can strafe)."""
        for k in range(3):
            # The first look is coarse: a brick at the edge of the view still says which way to go.
            c = self.see_block(f"lineup_{k}", check_present=(k == 0), allow_border=(k == 0))
            if c is None:
                return Result(False, f"no {self.target} in the wrist camera above the reach point")
            h = self.height()
            d = np.linalg.solve(jacobian(h, self.roll), grasp_pixel(h) - c)  # TCP move that would centre it
            self.log.ev("lineup", k=k, center=c, dxy=d)
            if np.linalg.norm(d) <= ARM_REACH_M:
                return Result(True, data={"dxy": d})
            if not self.base_ok:
                # Base locked: the arm takes the offset itself, then looks again. Returning here
                # (as before Oct 27) left a brick 6.4 cm to the side at the edge of the view, and
                # the roll step couldn't see it. Out of reach: the arm's planner refuses.
                p = self.cmd_tcp()
                self.move_tcp([p[0] + float(d[0]), p[1] + float(d[1]), p[2]], speed=FREE_EE_SPEED)
                self.log.ev("lineup_arm", dxy=d)
                continue
            # Move the base by the same displacement (arm and base frames share x forward, y
            # left), capped; the next frame measures what's left.
            d = np.clip(d, -0.15, 0.15)
            res = self.robot.base(dx=float(d[0]), dy=float(d[1]))
            self.log.ev("base", dxy=d, moved=res["result"].get("moved"))
        return Result(False, "the brick is still out of the arm's range after 3 base moves")

    def fetch(self, what: str | None = None) -> Result:
        """Drive to `what` and pick it: approach, line up, grasp."""
        self.target = what or self.target
        # Already brought here and confirmed by the last approach (nothing has moved the base
        # since): don't drive and look all over again.
        here = getattr(self, "confirmed", None) == self.map_key(self.target)
        for name, fn in () if here else (("go_to", lambda: self.go_to(self.target)),):
            r = fn()
            self.log.ev("step", name=name, ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"{name}: {r.reason}")
        if not getattr(self, "closed_in", False) and self.base_ok:
            res = self.robot.base(dx=CLOSE_IN_M)
            self.log.ev("base", dx=CLOSE_IN_M, moved=res["result"].get("moved"), why="close in")
            self.closed_in = True
        self.go_above(ABOVE_XY)
        r = self.line_up_base()
        self.log.ev("step", name="line_up", ok=r.ok, reason=r.reason, data=r.data)
        if not r.ok:
            return Result(False, f"line_up: {r.reason}")
        return self.grasp(None)

    def grasp(self, xy=(0.22, 0.0)) -> Result:
        """The whole pick from a rough block position (xy=None: the gripper is already above
        it, pointing down, open). Stops at the first failed step."""
        steps = [
            ("above", lambda: self.go_above(xy) if xy is not None else Result(True)),
            ("centre", self.centre),
        ]
        for name, fn in steps:
            r = fn()
            self.log.ev("step", name=name, ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"{name}: {r.reason}")
        for redo in range(GRASP_REDOS + 1):
            r = self.land()
            self.log.ev("step", name="land", ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"land: {r.reason}")
            gap = r.data.get("gap_m")
            if gap is None or 0.0 <= gap <= GRASP_GAP_MAX_M or redo == GRASP_REDOS:
                break
            # Landed off: re-aim by the landing error and come down again from the last servo
            # height. Positive shift: aim closer to the fixed jaw.
            self.aim_shift_m += gap - GRASP_MARGIN_M
            self.log.ev("regrasp", gap_mm=round(1000 * gap, 1), aim_shift_mm=round(1000 * self.aim_shift_m, 1))
            r = self.centre(start=len(SERVO_HEIGHTS) - 1)
            self.log.ev("step", name="centre_again", ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"centre again: {r.reason}")
        self.wrist("pre_close", settle=0.1)
        for name, fn in (("close", self.close), ("lift", self.lift_and_check)):
            r = fn()
            self.log.ev("step", name=name, ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"{name}: {r.reason}")
        self.holding = self.target
        return Result(True, f"holding the {self.target}")

    def land(self) -> Result:
        """From the last servo height: touch the floor, set the jaws GRASP_LIFT_M above it,
        slide along the brick's length if needed (center_along_length), and measure the fixed
        jaw's gap to the brick (grasp_gap_m) in that frame. data["gap_m"]: None when the brick
        wasn't boxed there."""
        touch = self.touch_floor()
        self.log.ev("step", name="touch", ok=touch.ok, reason=touch.reason, data=touch.data)
        if not touch.ok:
            return Result(False, f"touch: {touch.reason}")
        # Jaws GRASP_LIFT_M above the last free step, in commanded terms, plus at most
        # PRE_CLOSE_MAX_FIX_M of correction from the measured height: after a touch the
        # measured TCP has read 5-11 mm below the command, and chasing it fully (Oct 23 run
        # 222825: +17 mm of command) lifted the jaws over the brick.
        goal = touch.data["z"] + GRASP_LIFT_M
        z_cmd = touch.data["z_cmd"] + GRASP_LIFT_M
        # Back over where the centring left the brick: the "vertical" touch drifts ~8 mm
        # forward on the way down (Oct 24 run 211223, both touches), enough for the jaws to
        # miss the brick's end.
        p = self.cmd_tcp()
        drift = np.array(touch.data["start_xy"]) - p[:2]
        self.log.ev("touch_drift", dxy_mm=[round(1000 * float(v), 1) for v in drift])
        p[:2] = touch.data["start_xy"]
        self.move_tcp([p[0], p[1], z_cmd], speed=0.008)
        time.sleep(0.3)
        z = float(self.tcp()[2])
        fix = float(np.clip(goal - z, -PRE_CLOSE_MAX_FIX_M, PRE_CLOSE_MAX_FIX_M))
        if abs(fix) > 0.0015:
            self.move_tcp([p[0], p[1], z_cmd + fix], speed=0.008)
            time.sleep(0.3)
        self.log.ev("pre_close", z_cmd=z_cmd + fix, z_meas=float(self.tcp()[2]), goal=goal, fix_mm=round(1000 * fix, 1))
        box = self.center_along_length()
        gap = grasp_gap_m(box) if box is not None else None
        ok = gap is not None and 0.0 <= gap <= GRASP_GAP_MAX_M
        self.log.ev("grasp_check", gap_mm=None if gap is None else round(1000 * gap, 1), ok=ok,
                    aim_shift_mm=round(1000 * self.aim_shift_m, 1))
        return Result(True, data={"gap_m": gap})

    def front_guess(self, what: str) -> tuple[float, float] | None:
        """Rough arm-frame (x, y) of `what` on the floor from one front-camera look: bearing
        from the box's column, range from its bottom edge (RANGE_K fit) plus a few cm to the
        object's middle; clamped to where the arm can point down. The front camera sits just
        below the arm's base, so the camera's range is taken as the arm's. None if not seen."""
        self.arm.reset()
        got = self.see_front("front_guess", what)
        if got is None:
            return None
        (u, bottom), _ = got
        rng = float(np.clip(front_range(bottom), 0.12, 0.40)) + FRONT_GUESS_DEPTH_M
        bearing = math.radians(-(u - FRONT_U_TARGET) / FRONT_U_PER_DEG)
        x = float(np.clip(rng * math.cos(bearing), 0.15, 0.27))
        y = float(np.clip(rng * math.sin(bearing), -0.18, 0.18))
        self.log.ev("front_guess", what=what, u=u, bottom=bottom, range_m=round(rng, 3), bearing_deg=round(math.degrees(bearing), 1), xy=[x, y])
        return x, y

    def place_in(self, container: str) -> Result:
        """Drop what the gripper holds into `container` (an open container on the floor, in
        reach: run approach(container) first). Gripper pointing down with its tips
        DROP_TIP_HEIGHT_M above the floor; the container's box in the wrist camera is lined
        up under the gripper (arm first, base for what the arm can't reach), then the gripper
        opens."""
        if self.dropped():
            held = getattr(self, "holding", None) or "object"
            self.holding = None
            return Result(False, f"the {held} slipped out of the gripper on the way (the Pi saw the jaws close)")
        self.roll = DROP_ROLL_DEG
        z = self.floor_tcp + DROP_TIP_HEIGHT_M
        xy = ABOVE_XY
        if not self.base_ok:
            # Base locked: nothing can drive to the container, so start above where the front
            # camera puts it (direction from the box's column, rough range from where it meets
            # the floor) and let the arm do all the lining up.
            xy = self.front_guess(container) or ABOVE_XY
        self.arm.move_direct([xy[0], xy[1], z], pitch_deg=PITCH, roll_deg=self.roll)
        # The camera looks down at the rim: its height above that plane sets the pixel target
        # and scale (the taught grasp-pixel curve, without the fixed-jaw clearance shift).
        h = DROP_TIP_HEIGHT_M - DROP_RIM_HEIGHT_M
        err_px = float("inf")
        for k in range(DROP_ITERS):
            jpeg = self.wrist(f"drop_{k}")
            a = self.look(jpeg, f"the {container}", lambda b: (b[2] - b[0]) * (b[3] - b[1]) <= DROP_MAX_BOX_AREA)
            self.log.ev("drop_box", k=k, box=a.value, ms=round(a.ms), by=a.by)
            if a.value is None:
                return Result(False, f"no {container} in the wrist camera above the drop point")
            x0, y0, x1, y1 = a.value
            if (x1 - x0) * (y1 - y0) > DROP_MAX_BOX_AREA:
                # Gemma boxes the whole frame when it doesn't find the container (Oct 24 run
                # 190856: [0, 0, 1, 1] twice, and the base moved twice on it). Look again.
                self.log.ev("drop_box_rejected", k=k, why="covers most of the frame")
                continue
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            target = grasp_pixel(h) + jacobian(h, 0.0)[:, 0] * GRASP_CLEAR_FIXED_JAW_M  # camera frame: roll-free
            d = np.linalg.solve(jacobian(h, self.roll), target - c)
            err_px = float(np.linalg.norm((target - c) * IMAGE_PX))
            self.log.ev("drop_lineup", k=k, center=c, target=target, dxy=d, err_px=round(err_px, 1))
            if err_px < DROP_TOL_PX:
                break
            if np.linalg.norm(d) <= ARM_REACH_M or not self.base_ok:
                # The arm takes it (all of it when the base is locked, up to 5 cm per step).
                step = 0.8 * d
                n = float(np.linalg.norm(step))
                if n > DROP_ARM_STEP_M:
                    step *= DROP_ARM_STEP_M / n
                p = self.cmd_tcp()
                self.move_tcp([p[0] + step[0], p[1] + step[1], p[2]], speed=SERVO_EE_SPEED, z_min=self.arm.z_min)
            else:
                d = np.clip(d, -0.15, 0.15)
                res = self.robot.base(dx=float(d[0]), dy=float(d[1]))
                self.log.ev("base", dxy=d, moved=res["result"].get("moved"))
        else:
            return Result(False, f"could not line up over the {container} ({err_px:.0f} px off)")
        if self.dropped():
            self.holding = None
            return Result(False, f"the object slipped out of the gripper while lining up over the {container}")
        held = getattr(self, "holding", None) or "object"
        g0 = self.robot.status()["joints"]["arm_gripper"]
        g = self.robot.gripper(min(OPEN, g0 + RELEASE_OPEN_BY), speed=20)
        self.log.ev("release", from_pos=round(g0, 1), **g["result"])
        self.holding = None
        time.sleep(0.6)
        cup_box = a.value
        # Look before moving again: where did it go? Inside the container's box and clear of the
        # jaws (the bottom of the wrist image) = in. Gemma's box can be invented (it has boxed a
        # jaw when nothing was there), and that case fails this test, which is the safe side.
        jpeg = self.wrist("released", settle=0.1)
        b = self.gemma.box(jpeg, "the red Lego brick" if "red" in held else f"the {held}")
        p = self.cmd_tcp()
        self.move_tcp([p[0], p[1], p[2] + 0.04], speed=FREE_EE_SPEED, z_min=self.arm.z_min)
        self.robot.gripper(OPEN, speed=40)
        inside = False
        if b.value is not None:
            bc = np.array([(b.value[0] + b.value[2]) / 2, (b.value[1] + b.value[3]) / 2])
            x0, y0, x1, y1 = cup_box
            inside = x0 <= bc[0] <= x1 and y0 <= bc[1] <= y1 and bc[1] < 0.8
        self.log.ev("placed_check", held_box=b.value, container_box=cup_box, inside=inside)
        if not inside:
            return Result(False, f"released the {held}, but it is not seen inside the {container}: it may have fallen outside",
                          {"brick_box": b.value, "container_box": cup_box})
        return Result(True, f"dropped the {held} into the {container}")

    def place_next_to(self, ref: str) -> Result:
        """Set the held object down on the floor next to `ref` (a small object on the floor,
        in reach: run approach(ref) first), without touching `ref`. Lines up over ref with the
        wrist camera from NEXT_HOVER_M, moves NEXT_OFFSET_M sideways (toward the robot's
        centre line, else the other side), lowers until the held object meets the floor,
        opens, lifts and checks that ref is where it was and the object is on the floor."""
        if self.dropped():
            held = getattr(self, "holding", None) or "object"
            self.holding = None
            return Result(False, f"the {held} slipped out of the gripper on the way (the Pi saw the jaws close)")
        held = getattr(self, "holding", None) or "object"
        self.roll = DROP_ROLL_DEG
        xy = ABOVE_XY if self.base_ok else (self.front_guess(ref) or ABOVE_XY)
        # Not below the arm's floor guard: floor_tcp is the last touch's floor, which reads up to
        # 3 cm lower far out where the arm sags (Oct 26: -0.147 at x 0.28 vs -0.119 near), and
        # this hover is only the line-up height (the set-down stops on contact).
        z = max(self.floor_tcp + NEXT_HOVER_M, self.arm.z_min + 0.01)
        self.arm.move_direct([xy[0], xy[1], z], pitch_deg=PITCH, roll_deg=self.roll)
        h = NEXT_HOVER_M - NEXT_REF_HEIGHT_M
        err_px, c = float("inf"), None
        for k in range(DROP_ITERS):
            jpeg = self.wrist(f"next_{k}")
            # 26B, not the fast model: with a held brick filling the bottom of the view, E2B
            # boxed the jaws and the held brick in 17 of 25 recorded line-ups (Oct 26), so its
            # look plus the fallback cost more than asking 26B straight away.
            a = self.gemma.box(jpeg, f"the {ref} on the floor")
            self.log.ev("next_box", k=k, box=a.value, ms=round(a.ms))
            if a.value is None:
                return Result(False, f"no {ref} in the wrist camera above it")
            box, _ = snap_box(jpeg, a.value, what=ref)
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            target = grasp_pixel(h) + jacobian(h, 0.0)[:, 0] * GRASP_CLEAR_FIXED_JAW_M  # roll-free: under the jaws
            d = np.linalg.solve(jacobian(h, self.roll), target - c)
            err_px = float(np.linalg.norm((target - c) * IMAGE_PX))
            self.log.ev("next_lineup", k=k, center=c, target=target, dxy=d, err_px=round(err_px, 1))
            if err_px < DROP_TOL_PX:
                break
            if np.linalg.norm(d) <= ARM_REACH_M or not self.base_ok:
                step = 0.8 * d
                n = float(np.linalg.norm(step))
                if n > DROP_ARM_STEP_M:
                    step *= DROP_ARM_STEP_M / n
                p = self.cmd_tcp()
                self.move_tcp([p[0] + step[0], p[1] + step[1], p[2]], speed=SERVO_EE_SPEED, z_min=self.arm.z_min)
            else:
                d = np.clip(d, -0.15, 0.15)
                res = self.robot.base(dx=float(d[0]), dy=float(d[1]))
                self.log.ev("base", dxy=d, moved=res["result"].get("moved"))
        else:
            return Result(False, f"could not line up over the {ref} ({err_px:.0f} px off)")
        over = self.cmd_tcp()
        # Sideways (robot y) toward the centre line, so the arm stays in its comfortable reach.
        first = -1.0 if over[1] > 0.0 else 1.0
        for sign in (first, -first):
            try:
                self.move_tcp([over[0], over[1] + sign * NEXT_OFFSET_M, over[2]], speed=FREE_EE_SPEED, z_min=self.arm.z_min)
                break
            except PlanError as e:
                if sign == -first:
                    return Result(False, f"can't reach either side of the {ref}: {e}")
        # Down until the held object meets the floor; the Pi stops the arm on contact.
        p = self.cmd_tcp()
        goal = [p[0], p[1], p[2] - (NEXT_HOVER_M + 0.01)]
        wps, durs = plan_ee_line(self.arm.calib, self.arm.targets(), goal, PITCH, self.roll, TOUCH_SPEED, z_min=goal[2] - 0.002)
        res = self.robot.trajectory(wps, durs, z_min=goal[2] - 0.02, contact_stop=True)
        r = res.get("result", {})
        self.log.ev("set_down", **r)
        if not r.get("contact"):
            return Result(False, f"the {held} never met the floor: still holding it")
        g0 = self.robot.status()["joints"]["arm_gripper"]
        g = self.robot.gripper(min(OPEN, g0 + RELEASE_OPEN_BY), speed=20)
        self.log.ev("release", from_pos=round(g0, 1), **g["result"])
        # Let go = not holding it any more, whatever happens next (Oct 26: the retreat below
        # was refused, the brick already lay on the floor, and a retry "placed" it again).
        self.holding = None
        time.sleep(0.4)
        p = self.cmd_tcp()
        # The set-down ends below the arm's floor guard (the contact stop goes past it), so the
        # way up starts there; as in Arm.reset.
        z_min = min(self.arm.z_min, p[2] - JAW_TIP_OFFSET_M - 0.01)
        self.move_tcp([p[0], p[1], over[2]], speed=FREE_EE_SPEED, z_min=z_min)
        self.robot.gripper(OPEN, speed=40)
        # Back over the reference: it should be where it was in the image; the set-down object
        # beside it.
        self.move_tcp(over, speed=FREE_EE_SPEED, z_min=self.arm.z_min)
        jpeg = self.wrist("next_check")
        a = self.gemma.box(jpeg, f"the {ref} on the floor")
        b = self.gemma.box(jpeg, f"the {held} on the floor")
        ref_shift = None
        if a.value is not None:
            box, _ = snap_box(jpeg, a.value, what=ref)
            ref_shift = float(np.linalg.norm(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) - c))
        self.log.ev("next_check", ref_box=a.value, held_box=b.value, ref_shift=ref_shift)
        if ref_shift is None or ref_shift > NEXT_MAX_REF_SHIFT:
            return Result(False, f"set the {held} down, but the {ref} seems to have moved (or isn't seen)",
                          {"ref_shift": ref_shift})
        if b.value is None:
            return Result(False, f"set the {held} down next to the {ref}, but can't see it there now")
        return Result(True, f"put the {held} down next to the {ref}; the {ref} did not move")


__all__ = ["Skills", "Log", "Result", "grasp_pixel", "jacobian"]
