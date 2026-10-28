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

"""v2 layer 2, goals: what "done" looks like in an image, for servo(). Each goal turns a
detection into an error, says when the error is small enough, and turns it into one move of a
part. Everything is computed from the detection's box and the measured geometry (CALIB) with
params.py values; nothing per object.

    JawGoal          wrist: the object where the owner's midpoint teach had it just before
                     closing (its right edge beside the fixed jaw, its middle on the jaws)
    ReachGoal        wrist: the same point, within the arm's reach (for the base's line-up)
    AngleGoal        wrist: the object's long side across the jaws (for the wrist roll)
    FrontApproachGoal front: the object at the reach point, straight ahead (for the base)
    AboveGoal        wrist: a container where the owner's demo had it before lowering in
    BesideGoal       wrist: a reference object just beside where the held object will land
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v2 import params as prm
from v2.geometry import IMAGE_PX, MissingCalibrationError, require_grasp_target


@dataclass
class Detection:
    box: list  # x0, y0, x1, y1 image fractions (the object's top face, from above)
    jpeg: bytes = b""
    by: str = ""
    angle_deg: float | None = None  # long side from image vertical, when measured
    angle_how: str = ""

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def _cap(d: np.ndarray, image_shift: np.ndarray, cap: float) -> np.ndarray:
    """Scale a move so the image moves at most `cap` of its size: the object stays in view."""
    n = float(np.linalg.norm(image_shift))
    return d * (cap / n) if n > cap else d


class _WristPointGoal:
    """The object's (right edge u, middle v) on its taught spot, carried to the current height
    along a vertical descent of its top face (BRICK_HEIGHT_M up)."""

    cam = "wrist"
    tol_m = prm.SERVO_TOL_M
    part = "arm"

    def __init__(self, tol_m=None):
        if tol_m is not None:  # a looser tolerance, e.g. REACH_TOL_M for the base's line-up
            self.tol_m = tol_m

    def target(self, sk, height: float) -> np.ndarray:
        return self._carry(sk, require_grasp_target(sk.geom), height)  # the taught (right edge, middle)

    def _carry(self, sk, point, height: float) -> np.ndarray:
        """A top-face image point seen at the taught height, as it appears at `height`."""
        g, h_obj = sk.geom, prm.BRICK_HEIGHT_M.value
        h_taught = g.grasp_target_height if g.grasp_target_height is not None else prm.GRASP_LIFT_M.value
        s = (g.cam_above_tips + h_taught - h_obj) / (g.cam_above_tips + height - h_obj)
        return g.descent_p0 + (np.asarray(point, float) - g.descent_p0) * s

    def _jac(self, sk, height: float) -> np.ndarray:
        # Image motion of a point on the top face (nearer the camera than the floor).
        g, h_obj = sk.geom, prm.BRICK_HEIGHT_M.value
        return sk.geom.jacobian(height, sk.roll) * (g.cam_above_tips + height) / (g.cam_above_tips + height - h_obj)

    def error(self, det: Detection, sk) -> np.ndarray:
        h = sk.height()
        return self.target(sk, h) - np.array([det.box[2], det.center[1]])

    def done(self, err: np.ndarray, sk) -> bool:
        j = self._jac(sk, sk.height())
        tol = np.abs(np.linalg.norm(j, axis=1)) * self.tol_m.value  # image fraction per axis
        return bool(np.all(np.abs(err) <= tol))

    def step(self, err: np.ndarray, sk) -> dict:
        j = self._jac(sk, sk.height())
        d = prm.SERVO_GAIN.value * np.linalg.solve(j, err)
        d = _cap(d, (j @ d) * IMAGE_PX / IMAGE_PX[0], prm.STEP_IN_VIEW_FRACTION.value)
        return {"delta": (float(d[0]), float(d[1]))}


class JawGoal(_WristPointGoal):
    pass


class ReachGoal(_WristPointGoal):
    """Within REACH_TOL_M of the grasp spot: the base lines up, the arm takes the rest. The base
    shares the arm's x (forward) and y (left) axes, so the same move vector drives it."""

    tol_m = prm.REACH_TOL_M
    part = "base"


class AngleGoal:
    cam = "wrist"
    part = "roll"

    def error(self, det: Detection, sk) -> np.ndarray:
        if det.angle_deg is None:
            raise ValueError("AngleGoal needs a measured angle (find(..., angle=True))")
        return np.array([det.angle_deg])

    def done(self, err: np.ndarray, sk) -> bool:
        return abs(float(err[0])) <= prm.ANGLE_TOL_DEG.value

    def step(self, err: np.ndarray, sk) -> dict:
        # Image angle changes by -roll; the object looks the same every 180 deg: the nearest
        # roll inside the wrist's range (rest +/-90 deg), else its nearest end.
        lo, hi = sk.roll_range
        want = sk.roll + float(err[0])
        options = [want, want - 180.0, want + 180.0]  # convention: a brick looks the same every 180 deg
        fits = [r for r in options if lo <= r <= hi]
        roll = min(fits, key=lambda r: abs(r - sk.roll)) if fits else float(np.clip(min(options, key=lambda r: min(abs(r - lo), abs(r - hi))), lo, hi))
        return {"roll": roll}


class FrontApproachGoal:
    """Front camera: the object's column straight ahead of the arm (u_target) and its bottom
    edge on the row where it sits at the reach point (near_bottom): calibrate_front.py."""

    cam = "front"
    part = "base"

    def error(self, det: Detection, sk) -> np.ndarray:
        f = sk.geom.front
        return np.array([f["u_target"] - det.center[0], f["near_bottom"] - det.box[3]])

    def done(self, err: np.ndarray, sk) -> bool:
        f = sk.geom.front
        u_tol = f["u_per_deg"] * prm.MIN_BASE_TURN_DEG.value  # about what the smallest turn fixes
        return abs(float(err[0])) <= u_tol and float(err[1]) <= 0.0  # bottom at or past the row

    def step(self, err: np.ndarray, sk) -> dict:
        f = sk.geom.front
        if abs(float(err[0])) > f["u_per_deg"] * prm.MIN_BASE_TURN_DEG.value:
            # A left turn moves the object right in the image.
            turn = float(err[0]) / f["u_per_deg"]
            return {"roll": float(np.sign(turn) * max(abs(turn), prm.MIN_BASE_TURN_DEG.value))}
        # Drive a share of the way to where its bottom edge would reach the reach-point row.
        bottom = f["near_bottom"] - float(err[1])
        here, there = sk.geom.front_range(bottom), sk.geom.front_range(f["near_bottom"])
        return {"delta": (prm.SERVO_GAIN.value * max(here - there, 0.0), 0.0)}


class AboveGoal:
    """A container's box centre on the spot the owner's demo had it at the pause above it,
    seen from the same height (PLACE_HOVER_M): image-to-image, no depth model needed."""

    cam = "wrist"
    part = "arm"
    tol_m = prm.PLACE_TOL_M

    def __init__(self, tol_m=None):
        if tol_m is not None:
            self.tol_m = tol_m
        view = prm.demo_place_view()
        if view is None:
            raise MissingCalibrationError("no container view in the demo extract: run demo_extract.py --place-target")
        self.centre = np.array(view[0])

    def error(self, det: Detection, sk) -> np.ndarray:
        return self.centre - det.center

    def done(self, err: np.ndarray, sk) -> bool:
        tol = np.abs(np.linalg.norm(sk.geom.jacobian(sk.height(), sk.roll), axis=1)) * self.tol_m.value
        return bool(np.all(np.abs(err) <= tol))

    def step(self, err: np.ndarray, sk) -> dict:
        j = sk.geom.jacobian(sk.height(), sk.roll)
        d = prm.SERVO_GAIN.value * np.linalg.solve(j, err)
        d = _cap(d, (j @ d) * IMAGE_PX / IMAGE_PX[0], prm.STEP_IN_VIEW_FRACTION.value)
        return {"delta": (float(d[0]), float(d[1]))}


class BesideGoal(_WristPointGoal):
    """The reference object's right edge a gap left of where the held object lands (the taught
    grasp box), their middles level: the held object comes down beside it. The gap is
    NEXT_GAP_FRACTION of the taught box's width."""

    def target(self, sk, height: float) -> np.ndarray:
        g = sk.geom
        if g.grasp_box is None:
            raise MissingCalibrationError("calib.json 'wrist_grasp_target' has no box: run teach_grasp.py --write")
        x0, y0, x1, y1 = g.grasp_box
        spot = np.array([x0 - prm.NEXT_GAP_FRACTION.value * (x1 - x0), (y0 + y1) / 2])
        return self._carry(sk, spot, height)
