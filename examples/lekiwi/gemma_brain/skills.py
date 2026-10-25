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
  3. touch: step down until the floor holds the jaws up (measured TCP above its free-air
     sag), refusing if the touch comes much higher than the floor (landed on the block);
  4. lift GRASP_LIFT_M off the floor, close, read the stall;
  5. lift, and ask Gemma whether the gripper holds the brick.
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

from arm import Arm
from gemma import Gemma
from arm import plan_ee_line
from robot import Robot, RobotError
import overhead
from vision import brick_angle, crop_around, plausible_brick

CALIB = json.loads(Path(__file__).with_name("calib.json").read_text())
FLOOR_TCP_Z = CALIB["floor_tcp_z_m"]["value"]
J_WRIST = np.array(CALIB["wrist_image_jacobian"]["value"])  # d(u, v) / d(x, y), at J_HEIGHT
J_HEIGHT = 0.099
# Approximate camera distance above the jaw tips along the approach axis: pixel motion per
# metre scales as 1 / (height + this). A guess (the servo's gain absorbs the error).
CAM_ABOVE_TIPS_M = 0.06
GRASP_PIXEL = [(p["height_m"], np.array(p["center"])) for p in CALIB["wrist_grasp_pixel"]["value"]]

PITCH = 90.0
# Wrist roll: the camera rides on the wrist, so rolling by r turns the image by -r (roll test
# 2026-10-23, runs/20261023T222148-roll-test). The grasp was taught with the brick's long
# side vertical in the image (jaws closing across the 16 mm side), so roll by the measured
# angle to get back there. The angle measurement is good to ~+/-15 deg; squeezing jaws square
# a brick up, so that's enough.
ROLL_TOL_DEG = 12.0
ROLL_TRIES = 2
# The skills' roll range, owner's choice (2026-10-24): a 180 deg span, so with the brick's
# 180 deg symmetry every orientation is reachable by the wrist alone. 0 grips a brick whose
# long side runs left-right (the taught grasp), -90 one whose long side points at the robot
# (joint ~+89 deg; at rest the joint is +79), +90 the same from the other side (joint ~-84).
# Corrections after the first roll may go ROLL_OVERSHOOT_DEG past the ends rather than flip
# the wrist 180 deg; only a need beyond that turns the base.
ROLL_MIN_DEG, ROLL_MAX_DEG = -90.0, 90.0
ROLL_OVERSHOOT_DEG = 10.0
BASE_TURN_MARGIN_DEG = 10.0
IMAGE_PX = np.array([640.0, 480.0])
OPEN = 40.0
# The last stage, 1.5 cm (tips just above the 13 mm brick's studs), was added after the
# owner saw the jaws closing on the brick's end (runs 005149: after a blind descent from 3 cm
# the brick sat ~0.05 of the frame lower than taught, along its length).
SERVO_HEIGHTS = (0.10, 0.06, 0.03, 0.015)
# Per-height tolerance (pixels): 10 cm is the coarse line-up, 3 cm sets the grasp.
# 1.5 cm: 20 px, not 10: that close the jaws partly hide the brick and Gemma's box jitters by
# 15-30 px frame to frame (Oct 24 run 170013 oscillated 15-29 px at 10); the along-length
# check before closing does the fine placement.
SERVO_TOL_PX = (20.0, 14.0, 12.0, 20.0)
SERVO_GAIN_LAST = 0.5  # gentler at the last stage, so noisy boxes don't make it overshoot
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
GRASP_LIFT_M = 0.004  # jaw tips this far above the floor when closing (block body is 11.4 mm)
PRE_CLOSE_MAX_FIX_M = 0.003
# Just before closing, part of the brick must stick out beyond the jaw tips (owner's rule:
# never fully inside the gripper, so it has margin not to miss): aim for OUTSIDE_AIM of its
# length beyond the tips, accept OUTSIDE_MIN..OUTSIDE_MAX, else slide the gripper along the
# brick's length (never toward it) by at most PRE_CLOSE_ALONG_MAX_M per try.
# The fixed jaw is rigid with the camera, so its tip is always on the same image row
# (0.769 in 4 of 5 frames, Oct 23-24; the moving jaw's tip shows lower when open). The
# brick's full length at grasp height is BRICK_LEN_V (teach frame): its box is often cut
# off by the image bottom, so the outside share is measured against this, not the box.
FIXED_JAW_TIP_V = 0.769
BRICK_LEN_V = 0.30
OUTSIDE_AIM, OUTSIDE_MIN, OUTSIDE_MAX = 0.5, 0.35, 0.65
PRE_CLOSE_ALONG_MAX_M = 0.006
PRE_CLOSE_ALONG_TRIES = 2
GRASP_CLEAR_FIXED_JAW_M = 0.006
LIFT_M = 0.05
HOLD_CHECK_LIFT_M = 0.04  # second lift for the held-or-not test
HOLD_MAX_SHIFT = 0.03  # image fraction: a held brick moves less than this between the lifts
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


def jacobian(height: float, roll_deg: float = 0.0) -> np.ndarray:
    """d(image u, v)/d(TCP x, y) at this height and wrist roll: the roll-0 measurement,
    scaled for height and rotated (in pixel units: the image isn't square) by the image's
    turn of -roll."""
    j = J_WRIST * (J_HEIGHT + CAM_ABOVE_TIPS_M) / (max(height, 0.0) + CAM_ABOVE_TIPS_M)
    a = math.radians(-roll_deg)
    # Clockwise-positive rotation in a v-down image.
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (rot @ (j * IMAGE_PX[:, None])) / IMAGE_PX[:, None]


def measured_angle(jpeg: bytes, box) -> tuple[float, str]:
    """The brick's long-axis angle from image vertical, and how it was measured: the
    OpenCV fit when its blob is plausible, else the box's shape (vertical vs horizontal only)."""
    m = brick_angle(jpeg, box)
    if m is not None and 0.4 <= m["fill"] <= 1.15 and m["aspect"] >= 1.4:
        return m["angle_deg"], "fit"
    w_px = (box[2] - box[0]) * IMAGE_PX[0]
    h_px = (box[3] - box[1]) * IMAGE_PX[1]
    return (90.0 if w_px > 1.15 * h_px else 0.0), "box shape"


class Skills:
    def __init__(self, robot: Robot, arm: Arm, gemma: Gemma, log: Log):
        self.robot, self.arm, self.gemma, self.log = robot, arm, gemma, log
        self.roll = 0.0  # current wrist roll target for top-down moves
        self.target = DEFAULT_TARGET  # what pick() is after, as Gemma will be asked for it
        # The floor under the gripper (TCP z when the closed jaws touch it): calib.json's value
        # until the first floor touch of the session, then what that touch measured.
        self.floor_tcp = FLOOR_TCP_Z
        # The base can be locked (robot.base_enabled False): then the arm alone lines up.
        self.base_ok = getattr(robot, "base_enabled", True)
        # False: scan never turns the robot around; without the overhead camera it takes one
        # look straight ahead.
        self.turn_scan = True

    # --- sensing ---

    def tcp(self) -> np.ndarray:
        return self.arm.tcp()[:3, 3]

    def height(self) -> float:
        return float(self.tcp()[2] - self.floor_tcp)

    def wrist(self, tag: str, settle: float = LOOK_SETTLE_S) -> bytes:
        time.sleep(settle)
        jpeg, _ = self.robot.frame("wrist", after=time.time())
        self.log.frame(tag, jpeg)
        return jpeg

    def see_block(self, tag: str, check_present: bool = True, want_box: bool = False):
        """Centre (u, v) of the brick in a fresh wrist frame, or None if it isn't there.
        want_box: return (centre, box, jpeg) instead."""
        jpeg = self.wrist(tag)
        a = self.gemma.box(jpeg, f"the {self.target} on the floor")
        self.log.ev("box", tag=tag, box=a.value, ms=round(a.ms))
        if a.value is None:
            return None
        if check_present:
            # Not Gemma's yes/no: unreliable both ways on this brick (see vision.plausible_brick).
            ok, why = plausible_brick(jpeg, a.value)
            self.log.ev("present", tag=tag, ok=ok, why=why)
            if not ok:
                return None
        x0, y0, x1, y1 = a.value
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        return (c, a.value, jpeg) if want_box else c

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
        self.arm.move_direct([xy[0], xy[1], self.floor_tcp + height], pitch_deg=PITCH, roll_deg=0.0)
        g = self.robot.gripper(OPEN)
        self.log.ev("above", tcp=self.tcp(), gripper=g["result"])
        return Result(True)

    def align_roll(self) -> Result:
        """Roll the wrist so the jaws close on the brick's two long faces (squeezing across its
        16 mm width: the most pad contact; the owner's rule, and the taught grasp), i.e. its
        long side vertical in the image. Image angle changes by -roll (roll test, Oct 23), and
        the brick looks the same every 180 deg, so the roll needed is the measured angle, or
        that minus 180. Rolls stay in ROLL_MIN_DEG..ROLL_MAX_DEG; when neither fits, the
        result asks for a base turn ("base_turn_deg"): the camera turns with the base, a turn
        of b changing the image angle by +b (left positive), which moves the need back in range.
        Gripping across the 32 mm length (to roll less) was tried and missed (run 002847)."""
        for k in range(ROLL_TRIES + 1):
            got = self.see_block(f"roll_{k}", check_present=(k == 0), want_box=True)
            if got is None:
                return Result(False, "no brick in view to measure its angle")
            _, box, jpeg = got
            ang, how = measured_angle(jpeg, box)  # long side from image vertical, (-90, 90]
            self.log.ev("brick_angle", k=k, angle=ang, how=how, roll=self.roll)
            if abs(ang) <= ROLL_TOL_DEG or k == ROLL_TRIES:
                return Result(True, data={"roll": self.roll, "residual_deg": ang})
            options = [self.roll + ang, self.roll + ang - 180.0, self.roll + ang + 180.0]
            slack = ROLL_OVERSHOOT_DEG if k > 0 else 0.0
            fits = [r for r in options if ROLL_MIN_DEG - slack <= r <= ROLL_MAX_DEG + slack]
            if not fits:
                # A base turn of b adds b to the needed roll: turn so the nearest option lands
                # BASE_TURN_MARGIN_DEG inside the range (room for the angle's +/-5-15 deg noise).
                near = min(options, key=lambda r: min(abs(r - ROLL_MIN_DEG), abs(r - ROLL_MAX_DEG)))
                excess = (near - ROLL_MAX_DEG + BASE_TURN_MARGIN_DEG) if near > ROLL_MAX_DEG else (near - ROLL_MIN_DEG - BASE_TURN_MARGIN_DEG)
                return Result(False, f"the brick needs a {near:+.0f} deg roll, outside {ROLL_MIN_DEG:+.0f}..{ROLL_MAX_DEG:+.0f}",
                              {"base_turn_deg": -excess})
            new = min(fits, key=lambda r: abs(r - self.roll))
            p = self.cmd_tcp()
            self.roll = new
            self.move_tcp(p, speed=0.02)
        return Result(True)

    def centre_turning_base(self) -> Result:
        """centre(), and when the brick needs a wrist roll outside the allowed range, turn the
        base instead (the gripper stays 10 cm up, clear of the brick), line the base up
        again (the turn swings the reach point sideways), and centre once more."""
        r = self.centre()
        turn = r.data.get("base_turn_deg") if not r.ok else None
        if turn is None:
            return r
        self.roll = 0.0
        p = self.cmd_tcp()
        self.move_tcp(p, speed=0.02)  # back to roll 0 before the base turn
        res = self.robot.base(dtheta_deg=float(turn))
        self.log.ev("base", turn_deg=round(turn, 1), moved=res["result"].get("moved"), why="roll out of range")
        lu = self.line_up_base()
        self.log.ev("step", name="line_up_after_turn", ok=lu.ok, reason=lu.reason, data=lu.data)
        if not lu.ok:
            return lu
        return self.centre()

    def centre(self) -> Result:
        """Servo the block onto the grasp pixel at each SERVO_HEIGHTS, going down between."""
        r = self.align_roll()
        self.log.ev("step", name="roll", ok=r.ok, reason=r.reason, data=r.data)
        if not r.ok:
            return r
        last_err = None
        for i, h in enumerate(SERVO_HEIGHTS):
            if h in SKIPPABLE_HEIGHTS and last_err is not None and last_err < SERVO_TOL_PX[i]:
                self.log.ev("servo_skip", h=h, last_err_px=round(last_err, 1))
                continue
            if i > 0:
                p = self.cmd_tcp()
                self.move_tcp([p[0], p[1], self.floor_tcp + h], speed=FREE_EE_SPEED)
            for it in range(SERVO_ITERS):
                h_meas = self.height()
                c = self.see_block(f"servo_h{int(h * 100)}_{it}", check_present=False)
                if c is None:
                    return Result(False, f"no brick in view at {100 * h:.0f} cm")
                target = grasp_pixel(h_meas)
                e = target - c
                err_px = float(np.linalg.norm(e) * 640)
                self.log.ev("servo", h=h_meas, center=c, target=target, err_px=round(err_px, 1))
                last_err = err_px
                if err_px < SERVO_TOL_PX[i]:
                    break
                gain = SERVO_GAIN_LAST if i == len(SERVO_HEIGHTS) - 1 else SERVO_GAIN
                d = gain * np.linalg.solve(jacobian(h_meas, self.roll), e)
                n = float(np.linalg.norm(d))
                if n > SERVO_MAX_STEP_M:
                    d *= SERVO_MAX_STEP_M / n
                p = self.cmd_tcp()
                self.log.ev("servo_step", dxy=d)
                self.move_tcp([p[0] + d[0], p[1] + d[1], p[2]], speed=SERVO_EE_SPEED)
            else:
                return Result(False, f"did not converge at {100 * h:.0f} cm ({err_px:.0f} px off)")
        return Result(True)

    def touch_floor(self) -> Result:
        """One straight descent the Pi stops itself where the jaws meet the floor (a
        contact_stop trajectory: it watches the measured against the commanded TCP height every
        30 Hz tick), searching up to TOUCH_SEARCH_M below the start: no stored floor height is
        needed, so it works on a new floor. The height found becomes the session floor."""
        p = self.cmd_tcp()
        goal = [p[0], p[1], p[2] - TOUCH_SEARCH_M]
        z_min = goal[2] - 0.015 - 0.005  # the Pi checks the jaw tips, 1.5 cm below the TCP
        wps, durs = plan_ee_line(self.arm.calib, self.arm.targets(), goal, PITCH, self.roll, TOUCH_SPEED, z_min=goal[2] - 0.002)
        res = self.robot.trajectory(wps, durs, z_min=z_min, contact_stop=True)
        r = res.get("result", {})
        self.log.ev("touch", **r)
        if not r.get("contact"):
            return Result(False, f"no floor within {100 * TOUCH_SEARCH_M:.0f} cm below the start")
        z = float(r["z_meas"])
        free_gap = float(r.get("free_gap_mm", 0.0)) / 1000.0  # measured - commanded in free air (sag: negative)
        self.log.ev("floor", previous=round(self.floor_tcp, 4), now=round(z, 4), shift_mm=round(1000 * (z - self.floor_tcp), 1))
        self.floor_tcp = z
        return Result(True, data={"z": z, "z_cmd": z - free_gap, "how": r.get("how")})

    def center_along_length(self) -> None:
        """Just before closing: make sure part of the brick sticks out beyond the jaw tips
        (OUTSIDE_MIN..OUTSIDE_MAX of its length, aiming for OUTSIDE_AIM, i.e. its middle at
        the tips), so the jaws close on its middle with margin, not on one end with the rest
        deep inside the gripper. The wrist is rolled so the brick's long side is vertical
        (image v) and the jaws close along u; only the along-length error is corrected, so the
        gripper slides beside the brick and can't push it."""
        for k in range(PRE_CLOSE_ALONG_TRIES + 1):
            got = self.see_block(f"pre_close_check_{k}", check_present=False, want_box=True)
            if got is None:
                self.log.ev("along", ok=False, why="brick not boxed before closing")
                return
            _, box, _ = got
            outside = float(np.clip((FIXED_JAW_TIP_V - box[1]) / BRICK_LEN_V, 0.0, 1.0))
            self.log.ev("along", k=k, box=box, outside=round(outside, 2))
            if OUTSIDE_MIN <= outside <= OUTSIDE_MAX or k == PRE_CLOSE_ALONG_TRIES:
                return
            # More outside = the brick higher in the image (smaller v).
            ev = -(OUTSIDE_AIM - outside) * BRICK_LEN_V
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
        """Held or not, from geometry rather than one frame: the wrist camera rides on the
        gripper, so a held brick stays put in the image as the gripper rises, while a brick
        left on the floor shifts ~0.1 of the frame per 3-4 cm (the teach run). Gemma only
        boxes the brick; a yes/no on a single frame can't tell held from not held (tested
        2026-10-23: every phrasing got at least one of 4 known frames wrong). Also the
        gripper must still be stalled open on something, not closed to empty."""
        boxes = []
        for k, dz in enumerate((LIFT_M, HOLD_CHECK_LIFT_M)):
            p = self.cmd_tcp()
            self.move_tcp([p[0], p[1], p[2] + dz], speed=FREE_EE_SPEED)
            jpeg = self.wrist(f"lifted_{k}")
            a = self.gemma.box(jpeg, f"the {self.target}")
            boxes.append(a.value)
            self.log.ev("held_box", k=k, height=self.height(), box=a.value, ms=round(a.ms))
        g = self.robot.status()["joints"]["arm_gripper"]
        if g < GRIPPER_EMPTY_MAX:
            return Result(False, f"gripper closed to {g:.1f}: the brick slipped out", {"gripper": g})
        if None in boxes:
            return Result(False, "brick not seen after lifting", {"gripper": g, "boxes": boxes})
        (a0, b0, a1, b1), (c0, d0, c1, d1) = boxes
        shift = float(np.hypot((c0 + c1 - a0 - a1) / 2, (d0 + d1 - b0 - b1) / 2))
        size = ((c1 - c0) * (d1 - d0)) / max(1e-6, (a1 - a0) * (b1 - b0))
        data = {"gripper": round(g, 1), "shift": round(shift, 3), "size_ratio": round(size, 2)}
        if shift > HOLD_MAX_SHIFT or not 0.8 < size < 1.25:
            return Result(False, "the brick stayed behind: it moved in the image as the gripper rose", data)
        return Result(True, data=data)

    def see_front(self, tag: str, what: str) -> tuple[np.ndarray, list] | None:
        """(u, v_bottom) and the box of `what` on the floor in a fresh front-camera frame, or
        None. Gemma's box is only accepted if it could be a thing on the floor: its bottom
        edge in the floor part of the image (with nothing there, Gemma has boxed a hat on a
        coat rack), and clear of the left/right border (it has boxed the rug at the edge).
        Gemma's yes/no isn't used: unreliable on these objects (see vision.plausible_brick)."""
        time.sleep(LOOK_SETTLE_S)
        jpeg, _ = self.robot.frame("front", after=time.time())
        self.log.frame(tag, jpeg)
        a = self.gemma.box(jpeg, f"the {what} on the floor")
        b = a.value
        ok = b is not None and b[3] > FRONT_FLOOR_V and b[0] > 0.01 and b[2] < 0.99
        self.log.ev("front_box", tag=tag, what=what, box=b, accepted=ok, ms=round(a.ms))
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
            # No full turn (turn_scan off, or the base locked): one look straight ahead.
            return self.scan_rotate(objects, stops=1)
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

    def scan_rotate(self, objects: list[str], stops: int | None = None) -> Result:
        """Turn a full circle in SCAN_STEP_DEG stops (or look `stops` times; 1 = only straight
        ahead, no turning), boxing every object at every stop, and keep a map: for each object,
        its candidates (bearing, rough range, number of views), best first. Gemma's box is the
        only judge here, so a candidate can be wrong (E4B has boxed a backpack as a "short paper
        cup"); approach() confirms up close."""
        self.arm.reset()
        self.map = {o: [] for o in objects}
        sightings = {o: [] for o in objects}
        full = stops is None
        stops = int(round(360.0 / SCAN_STEP_DEG)) if full else stops
        for k in range(stops):
            if k:
                res = self.robot.base(dtheta_deg=-SCAN_STEP_DEG)
                self.log.ev("base", turn_deg=-SCAN_STEP_DEG, moved=res["result"].get("moved"))
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
                    cls = self.classify(jpeg, b, o)
                    ok = cls == category(o)
                if ok:
                    u = (b[0] + b[2]) / 2
                    bearing = head - (u - FRONT_U_TARGET) / FRONT_U_PER_DEG
                    rng = RANGE_K / max(0.005, b[3] - RANGE_HORIZON_V)
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
        return self._map_result(objects, "360-degree turn" if full else "one look straight ahead")

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
        yn = self.gemma.yes_no(jpeg, f"Is there a {what} in this image?")
        self.log.ev("confirm", what=what, answer=yn.text, ms=round(yn.ms))
        return Result(yn.value is True, "" if yn.value else f"up close, the wrist camera does not see a {what}")

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
            got = self.see_front(f"approach_{k}", what)
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
            c = self.see_block(f"lineup_{k}", check_present=(k == 0))
            if c is None:
                return Result(False, f"no {self.target} in the wrist camera above the reach point")
            h = self.height()
            d = np.linalg.solve(jacobian(h, self.roll), grasp_pixel(h) - c)  # TCP move that would centre it
            self.log.ev("lineup", k=k, center=c, dxy=d)
            if np.linalg.norm(d) <= ARM_REACH_M:
                return Result(True, data={"dxy": d})
            if not self.base_ok:
                # Base locked: the arm's own centring (centre()) takes the whole offset, if the
                # pose is reachable; it reports if it isn't.
                return Result(True, data={"dxy": d, "base_locked": True})
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
            ("centre", self.centre_turning_base),
            ("touch", self.touch_floor),
        ]
        touch = None
        for name, fn in steps:
            r = fn()
            self.log.ev("step", name=name, ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"{name}: {r.reason}")
            touch = r if name == "touch" else touch
        # Jaws GRASP_LIFT_M above the last free step, in commanded terms, plus at most
        # PRE_CLOSE_MAX_FIX_M of correction from the measured height: after a touch the
        # measured TCP has read 5-11 mm below the command, and chasing it fully (Oct 23 run
        # 222825: +17 mm of command) lifted the jaws over the brick.
        goal = touch.data["z"] + GRASP_LIFT_M
        z_cmd = touch.data["z_cmd"] + GRASP_LIFT_M
        p = self.cmd_tcp()
        self.move_tcp([p[0], p[1], z_cmd], speed=0.008)
        time.sleep(0.3)
        z = float(self.tcp()[2])
        fix = float(np.clip(goal - z, -PRE_CLOSE_MAX_FIX_M, PRE_CLOSE_MAX_FIX_M))
        if abs(fix) > 0.0015:
            self.move_tcp([p[0], p[1], z_cmd + fix], speed=0.008)
            time.sleep(0.3)
        self.log.ev("pre_close", z_cmd=z_cmd + fix, z_meas=float(self.tcp()[2]), goal=goal, fix_mm=round(1000 * fix, 1))
        self.center_along_length()
        self.wrist("pre_close", settle=0.1)
        for name, fn in (("close", self.close), ("lift", self.lift_and_check)):
            r = fn()
            self.log.ev("step", name=name, ok=r.ok, reason=r.reason, data=r.data)
            if not r.ok:
                return Result(False, f"{name}: {r.reason}")
        self.holding = self.target
        return Result(True, f"holding the {self.target}")

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
        rng = float(np.clip(RANGE_K / max(0.005, bottom - RANGE_HORIZON_V), 0.12, 0.40)) + FRONT_GUESS_DEPTH_M
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
            a = self.gemma.box(jpeg, f"the {container}")
            self.log.ev("drop_box", k=k, box=a.value, ms=round(a.ms))
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
        held = getattr(self, "holding", None) or "object"
        g0 = self.robot.status()["joints"]["arm_gripper"]
        g = self.robot.gripper(min(OPEN, g0 + RELEASE_OPEN_BY), speed=20)
        self.log.ev("release", from_pos=round(g0, 1), **g["result"])
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
        self.holding = None
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


__all__ = ["Skills", "Log", "Result", "grasp_pixel", "jacobian"]
