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

"""The measured geometry the v2 skills work in, loaded once from calib.json and
floor_touches.json, each with where it was measured (CALIB). No defaults: a missing
calibration is an error naming the script that makes it.

    floor       TCP z where the vertical jaws meet the floor, a plane fitted to the touches
    wrist       image Jacobian d(u, v)/d(x, y) at a height, scaled for other heights and rolls
    descent     camera height above the jaw tips and the point a vertical descent heads for
    jaws        the fixed jaw's tip row and inner edge in the wrist image
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2 import params as prm

HERE = Path(__file__).resolve().parent.parent
IMAGE_PX = np.array([640.0, 480.0])  # convention: the cameras' 640x480 frames
MAKERS = {
    "floor_tcp_z_m": "calibrate_floor.py",
    "wrist_image_jacobian": "wrist_calib.py",
    "wrist_descent": "calibrate_descent.py",
    "wrist_jaws": "calibrate_jaws.py",
    "front_camera": "calibrate_front.py",
}


class MissingCalibrationError(RuntimeError):
    pass


@dataclass
class Geometry:
    floor_coef: np.ndarray  # z = a + b x + c y (TCP z at floor contact, vertical gripper)
    floor_points: int
    jac: np.ndarray  # d(u, v)/d(x, y) at jac_height, roll 0 (image fractions per metre)
    jac_height: float
    cam_above_tips: float
    descent_p0: np.ndarray
    fixed_jaw_inner_u: float
    fixed_jaw_tip_v: float
    sources: dict
    front: dict = None  # calibrate_front.py: u_per_deg, u_target, near_bottom, ...
    grasp_target: np.ndarray | None = None  # taught (u of the box's right edge, v of its middle)
    grasp_target_height: float | None = None  # jaw tips above the floor when it was taught
    grasp_box: list | None = None  # the whole taught box (x0, y0, x1, y1)

    @classmethod
    def load(cls, calib_path: Path = HERE / "calib.json", touches_path: Path = HERE / "floor_touches.json") -> Geometry:
        calib = json.loads(calib_path.read_text())
        for key, maker in MAKERS.items():
            if key not in calib:
                raise MissingCalibrationError(f"calib.json has no {key!r}: run {maker}")
        pts = json.loads(touches_path.read_text())["points"] if touches_path.exists() else []
        coef = fit_floor(pts, calib["floor_tcp_z_m"]["value"])
        jw = calib["wrist_image_jacobian"]
        return cls(
            floor_coef=coef,
            floor_points=len(pts),
            jac=np.array(jw["value"]),
            jac_height=float(jw["height_m"]),  # wrist_calib.py records it (Oct 27 on)
            cam_above_tips=float(calib["wrist_descent"]["value"]["depth_m"]),
            descent_p0=np.array(calib["wrist_descent"]["value"]["p0"]),
            fixed_jaw_inner_u=float(calib["wrist_jaws"]["value"]["fixed_inner_u"]),
            fixed_jaw_tip_v=float(calib["wrist_jaws"]["value"]["fixed_tip_v"]),
            sources={k: {f: calib[k].get(f) for f in ("method", "date", "log") if f in calib[k]} for k in MAKERS},
            front=dict(calib["front_camera"]["value"]),
            grasp_target=np.array(calib["wrist_grasp_target"]["value"]) if "wrist_grasp_target" in calib else None,
            grasp_target_height=calib["wrist_grasp_target"]["height_m"] if "wrist_grasp_target" in calib else None,
            grasp_box=calib["wrist_grasp_target"].get("box") if "wrist_grasp_target" in calib else None,
        )

    def floor(self, x: float, y: float) -> float:
        return float(self.floor_coef @ np.array([1.0, x, y]))

    def jacobian(self, height: float, roll_deg: float = 0.0) -> np.ndarray:
        """d(u, v)/d(TCP x, y) with the jaw tips `height` above the floor: the measured one,
        scaled by the camera's distance to the floor and turned by -roll (pixel units, as the
        image isn't square)."""
        d = self.cam_above_tips
        j = self.jac * (self.jac_height + d) / (max(height, 0.0) + d)
        a = math.radians(-roll_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return (rot @ (j * IMAGE_PX[:, None])) / IMAGE_PX[:, None]

    def front_range(self, bottom_v: float) -> float:
        """Distance of an object on the floor whose box bottom is at image row bottom_v in the
        front camera (calibrate_front.py's model: k / (bottom - horizon) - offset), metres."""
        f = self.front
        return f["range_k"] / max(bottom_v - f["range_horizon_v"], 1e-3) - f["range_offset_m"]  # convention: keeps the division finite at the horizon

    def tips_to_top(self, point, height: float, object_height: float) -> np.ndarray:
        """Where a point at the jaw tips' depth appears on the top face of an object
        `object_height` tall, with the tips `height` above the floor: nearer the camera, so
        further from descent_p0 (the top face is what the camera boxes)."""
        s = self.cam_above_tips / (self.cam_above_tips + height - object_height)
        return self.descent_p0 + (np.asarray(point, float) - self.descent_p0) * s

    def descend(self, point, h_from: float, h_to: float) -> np.ndarray:
        """Where a floor point seen at image `point` with the tips h_from up appears at h_to."""
        s = (self.cam_above_tips + h_from) / (self.cam_above_tips + h_to)
        return self.descent_p0 + (np.asarray(point, float) - self.descent_p0) * s


def front_object_height(geom: Geometry, box) -> float:
    """An object's height (m) from its front-camera box: the camera's height over the floor is
    range_k / f (the range model k / (bottom - horizon) is a pinhole's distance), and the object
    spans (bottom - top) / (bottom - horizon) of it. All from calibrate_front.py; a paper cup
    came out 4.4 cm in 18 views from 0.5 m to the reach point, within 0.25 cm (Oct 28)."""
    f = geom.front
    f_v = f["u_per_deg"] * np.degrees(1.0) * IMAGE_PX[0] / IMAGE_PX[1]  # height fractions per radian
    bottom, top = box[3], box[1]
    return float(f["range_k"] / f_v * (bottom - top) / max(bottom - f["range_horizon_v"], 1e-3))  # convention: keeps the division finite at the horizon


def locate_from_one_view(geom: Geometry, c, h: float, height: float, roll_deg: float = 0.0) -> np.ndarray:
    """(dx, dy) in metres from the point under the TCP to an object's point seen at image `c`,
    the jaw tips h above the floor, the point `height` above the floor. The point under the TCP
    at a depth is where the taught brick's middle was (it was held between the jaws), moved
    along the descent from descent_p0; the image offset from it, divided by the Jacobian scaled
    to that depth, is the offset in metres."""
    d = geom.cam_above_tips
    x0, y0, x1, y1 = geom.grasp_box
    taught_centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    z_taught = d + geom.grasp_target_height - prm.BRICK_HEIGHT_M.value
    z = d + h - height
    under_tcp = geom.descent_p0 + (taught_centre - geom.descent_p0) * z_taught / z
    j = geom.jacobian(h, roll_deg) * (d + h) / z
    return -np.linalg.solve(j, np.asarray(c, float) - under_tcp)


def view_half_width(geom: Geometry, h: float, height: float) -> float:
    """Half the wrist view's narrower side, in metres, on a plane `height` above the floor with
    the jaw tips h above it: anything located further off can't be in view."""
    z = geom.cam_above_tips + h - height
    j = geom.jacobian(h) * (geom.cam_above_tips + h) / z  # image fraction per metre at that plane
    return float(min(0.5 / np.linalg.norm(j[0]), 0.5 / np.linalg.norm(j[1])))  # convention: half the view


def grasp_spot(geom: Geometry) -> np.ndarray:
    """The wrist-image spot under the gripper: the taught brick's box centre (it was held there)."""
    if geom.grasp_box is None:
        raise MissingCalibrationError("calib.json 'wrist_grasp_target' has no box: run teach_grasp.py --write")
    x0, y0, x1, y1 = geom.grasp_box
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def look_offsets(step_deg: float) -> list[float]:
    """Headings (deg from the start, left positive) for a look around: alternately left and
    right in widening steps, ending turned right round."""
    offsets, k = [], 1
    while k * step_deg < 180:  # convention: half a turn either way covers the circle
        offsets += [k * step_deg, -k * step_deg]
        k += 1
    return offsets + [180.0]  # convention: facing backwards


def wrap_deg(a: float) -> float:
    """An angle in degrees as the short way round, -180..180."""
    return (a + 180.0) % 360.0 - 180.0  # convention: a full turn


def require_grasp_target(geom: Geometry) -> np.ndarray:
    if geom.grasp_target is None:
        raise MissingCalibrationError("calib.json has no 'wrist_grasp_target': run teach_grasp.py (the owner's midpoint teach)")
    return geom.grasp_target


def fit_floor(points: list[dict], fallback_z: float) -> np.ndarray:
    """Plane z = a + b x + c y through the floor touches (least squares). Fewer than 3 points:
    flat at their mean, or at the calibrated value with none. With the gripper vertical the
    floor came out flat to 3.3 mm over 12 cm (Oct 27), so no sag term."""
    if len(points) >= 3:  # convention: a plane needs 3 points
        a = np.array([[1.0, p["x"], p["y"]] for p in points])
        coef, *_ = np.linalg.lstsq(a, np.array([p["z"] for p in points]), rcond=None)
        return coef
    z = float(np.mean([p["z"] for p in points])) if points else float(fallback_z)
    return np.array([z, 0.0, 0.0])
