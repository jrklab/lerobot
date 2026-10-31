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

"""The measured geometry the skills work in, loaded once from calib.json and
floor_touches.json, each with where it was measured (CALIB). No defaults: a missing
calibration is an error naming the script that makes it.

    floor       TCP z where the vertical jaws meet the floor, a plane fitted to the touches
    cameras     both cameras' lenses and poses, the floor plane they see and the point the base
                turns about (brain/cameras.py; capture_cameras.py + solve_cameras.py): image
                points to floor or object points and back, for the wrist camera at any height
                and roll, the front camera at any distance
    front       the box bottom of an object at the reach point (calibrate_front.py's near_bottom,
                a brick): the front approach's stopping distance
    gripper     the jaw opening (mm) against the gripper angle (deg open from closed)

The wrist's image-space values used to be fitted (an image Jacobian at one height, a descent
point, the camera's depth above the tips; the front camera's image widths per degree and a
range curve): until Oct 30 they were, and the camera model predicts the logged views at least
as well (front turns: 16 px median error against 27; wrist: 14 px against 11, within half a
millimetre at the brick, and through roll turns the fitted ones didn't model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.arm import tool_orientation

from brain import params as prm
from brain.cameras import Cameras, on_plane, to_image

HERE = Path(__file__).resolve().parent.parent
MAKERS = {
    "floor_tcp_z_m": "calibrate_floor.py",
    "cameras": "capture_cameras.py scene + solve_cameras.py --write",
    "front_camera": "calibrate_front.py",
    "gripper_opening": "calibrate_gripper.py",
}
IMAGE_CENTRE = np.array([0.5, 0.5])  # convention: image fractions
MIDDLE_ROW_ENDS = ((0.0, 0.5), (1.0, 0.5))  # convention: image fractions
BOTTOM_CENTRE = (0.5, 1.0)  # convention: image fractions
EDGE_MIDPOINTS = ((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0))  # convention: image fractions
FAR_M = 100.0  # convention: a floor point far enough ahead to stand for the horizon
JACOBIAN_STEP_M = 1e-4  # convention: a TCP move small next to the view, large next to rounding


class MissingCalibrationError(RuntimeError):
    pass


@dataclass
class Geometry:
    floor_coef: np.ndarray  # z = a + b x + c y (TCP z at floor contact, vertical gripper)
    floor_points: int
    cams: Cameras
    floor_seen_z: float  # the floor plane the cameras were placed against (the floor board's)
    sources: dict
    front_near_x: float | None = None  # the box bottom of an object at the reach point: its distance
    # from the turning centre once straight ahead, as an x (front_turn_and_distance measures so)
    grasp_target: np.ndarray | None = None  # taught (u of the box's right edge, v of its middle)
    grasp_target_height: float | None = None  # jaw tips above the floor when it was taught
    grasp_box: list | None = None  # the whole taught box (x0, y0, x1, y1)
    gripper_curve: list | None = None  # [[deg, mm], ...]: calibrate_gripper.py

    @classmethod
    def load(
        cls,
        calib_path: Path = HERE / "data" / "calib.json",
        touches_path: Path = HERE / "data" / "floor_touches.json",
    ) -> Geometry:
        calib = json.loads(calib_path.read_text())
        for key, maker in MAKERS.items():
            if key not in calib:
                raise MissingCalibrationError(f"calib.json has no {key!r}: run {maker}")
        pts = json.loads(touches_path.read_text())["points"] if touches_path.exists() else []
        cams = Cameras.from_calib(calib["cameras"]["value"])
        floor_seen = float(calib["cameras"]["value"]["floor_z"])
        f = calib["front_camera"]["value"]
        near = on_plane(cams.front, cams.T_base_cam, (f["u_target"], f["near_bottom"]), floor_seen)
        return cls(
            floor_coef=fit_floor(pts, calib["floor_tcp_z_m"]["value"]),
            floor_points=len(pts),
            cams=cams,
            floor_seen_z=floor_seen,
            sources={
                k: {f: calib[k].get(f) for f in ("method", "date", "log") if f in calib[k]} for k in MAKERS
            },
            front_near_x=None if near is None else near_distance(cams, near),
            grasp_target=np.array(calib["wrist_grasp_target"]["value"])
            if "wrist_grasp_target" in calib
            else None,
            grasp_target_height=calib["wrist_grasp_target"]["height_m"]
            if "wrist_grasp_target" in calib
            else None,
            grasp_box=calib["wrist_grasp_target"].get("box") if "wrist_grasp_target" in calib else None,
            gripper_curve=calib["gripper_opening"]["value"]["curve"],
        )

    def _gripper(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.gripper_curve:
            raise MissingCalibrationError("calib.json has no 'gripper_opening': run calibrate_gripper.py")
        c = np.asarray(self.gripper_curve, float)
        return c[:, 0], c[:, 1]

    def gripper_mm(self, deg: float) -> float:
        """The jaw opening (mm, at the tips) for a gripper angle (deg open from closed): straight
        between the measured points (the jaw swings on a pivot: 1.85 mm/deg near closed, 1.3
        from 16 to 48 mm, Oct 29), beyond the last at its last segment's slope; 0 at or below
        the angle where the jaws meet."""
        d, m = self._gripper()
        if deg > d[-1]:
            return float(m[-1] + (deg - d[-1]) * (m[-1] - m[-2]) / (d[-1] - d[-2]))
        return float(np.interp(deg, d, m))

    def gripper_deg(self, mm: float) -> float:
        """The gripper angle (deg open from closed) for a jaw opening in mm (gripper_mm's inverse)."""
        d, m = self._gripper()
        if mm > m[-1]:
            return float(d[-1] + (mm - m[-1]) * (d[-1] - d[-2]) / (m[-1] - m[-2]))
        return float(np.interp(mm, m, d))

    def floor(self, x: float, y: float) -> float:
        return float(self.floor_coef @ np.array([1.0, x, y]))

    # --- the wrist camera, the gripper vertical over the reach point ---

    def tcp_pose(self, height: float, roll_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        """The TCP pointing straight down over the reach point (plus dx, dy), the jaw tips
        `height` above the floor, turned roll_deg about its axis: the pose the wrist skills
        look from."""
        x, y = prm.REACH_POINT_X_M.value + dx, dy
        pose = np.eye(4)  # convention: a 4x4 pose
        pose[:3, :3] = tool_orientation(x, y, 90.0, roll_deg)  # convention: pointing straight down
        pose[:3, 3] = [x, y, self.floor(x, y) + height]
        return pose

    def wrist_point(self, uv, height: float, plane_h: float, roll_deg: float = 0.0) -> np.ndarray | None:
        """The point plane_h above the floor that the wrist camera sees at image point uv, the
        jaw tips `height` up (arm frame)."""
        cam = self.cams.wrist_pose(self.tcp_pose(height, roll_deg))
        return on_plane(self.cams.wrist, cam, uv, self.floor_seen_z + plane_h)

    def wrist_image(
        self, p, height: float, roll_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0
    ) -> np.ndarray | None:
        return to_image(self.cams.wrist, self.cams.wrist_pose(self.tcp_pose(height, roll_deg, dx, dy)), p)

    def carry(self, point, plane_h: float, h_from: float, h_to: float, roll_deg: float = 0.0) -> np.ndarray:
        """Where a point plane_h above the floor, seen at image `point` with the tips h_from up,
        appears with them h_to up (a vertical descent or rise)."""
        return self.wrist_image(self.wrist_point(point, h_from, plane_h, roll_deg), h_to, roll_deg)

    def descend(self, point, h_from: float, h_to: float) -> np.ndarray:
        """Where a floor point seen at image `point` with the tips h_from up appears at h_to."""
        return self.carry(point, 0.0, h_from, h_to)

    def tips_to_top(self, point, height: float, object_height: float) -> np.ndarray:
        """Where a point at the jaw tips' depth appears on the top face of an object
        `object_height` tall, with the tips `height` above the floor (the point straight above
        or below it on that face: what the camera boxes)."""
        p = self.wrist_point(point, height, height)
        return self.wrist_image([p[0], p[1], self.floor_seen_z + object_height], height)

    def jacobian(self, height: float, roll_deg: float = 0.0, plane_h: float = 0.0) -> np.ndarray:
        """d(u, v)/d(TCP x, y): how a point plane_h above the floor moves in the wrist image per
        metre of TCP move, the tips `height` up, at the image spot under the gripper (the taught
        grasp spot, else the image centre)."""
        spot = grasp_spot(self) if self.grasp_box is not None else IMAGE_CENTRE
        p = self.wrist_point(spot, height, plane_h, roll_deg)
        here = self.wrist_image(p, height, roll_deg)
        cols = [
            (self.wrist_image(p, height, roll_deg, *step) - here) / JACOBIAN_STEP_M
            for step in ((JACOBIAN_STEP_M, 0.0), (0.0, JACOBIAN_STEP_M))
        ]
        return np.column_stack(cols)

    # --- the front camera ---

    def front_point(self, uv) -> np.ndarray | None:
        """The floor point the front camera sees at image point uv; None above the horizon."""
        return on_plane(self.cams.front, self.cams.T_base_cam, uv, self.floor_seen_z)

    def front_horizon_v(self) -> float:
        """The image row of the floor's horizon straight ahead: nothing on the floor ends above it."""
        far = self.cams.T_base_cam[:3, 3] + [FAR_M, 0.0, 0.0]
        return float(to_image(self.cams.front, self.cams.T_base_cam, [far[0], far[1], self.floor_seen_z])[1])

    def front_view_deg(self) -> float:
        """The front camera's view across, in degrees (its image's middle row, edge to edge)."""
        a, b = (self.cams.T_base_cam[:3, :3] @ self.cams.front.ray(uv) for uv in MIDDLE_ROW_ENDS)
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))

    def front_turn_and_distance(self, p) -> tuple[float, float]:
        """For a floor point p: the base turn (deg, left positive) that puts it straight ahead of
        the arm, and how far it then lies beyond the approach's stop (front_near_x), metres. The
        base turns about cams.turn_centre."""
        v = np.asarray(p[:2], float) - self.cams.turn_centre
        return float(np.degrees(np.arctan2(v[1], v[0]))), near_distance(self.cams, p) - self.front_near_x


def near_distance(cams: Cameras, p) -> float:
    """A floor point's x once the base has turned it straight ahead of the arm."""
    return float(cams.turn_centre[0] + np.hypot(*(np.asarray(p[:2], float) - cams.turn_centre)))


def front_view_margin(geom: Geometry) -> float:
    """How much floor (m) the front camera still sees beyond the approach's stop, to the frame's
    bottom: an overshoot larger than this loses the object."""
    return geom.front_near_x - near_distance(geom.cams, geom.front_point(BOTTOM_CENTRE))


def front_object_height(geom: Geometry, box) -> float:
    """An object's height (m) from its front-camera box: its bottom edge is on the floor, which
    gives its distance; the ray through its top edge, at that distance, gives the height."""
    u = (box[0] + box[2]) / 2
    bottom = geom.front_point((u, box[3]))
    if bottom is None:
        return 0.0
    o = geom.cams.T_base_cam[:3, 3]
    d = geom.cams.T_base_cam[:3, :3] @ geom.cams.front.ray((u, box[1]))
    t = np.hypot(*(bottom[:2] - o[:2])) / np.hypot(*d[:2])
    return float(o[2] + t * d[2] - geom.floor_seen_z)


def locate_from_one_view(geom: Geometry, c, h: float, height: float, roll_deg: float = 0.0) -> np.ndarray:
    """(dx, dy) in metres from the point under the gripper to an object's point seen at image
    `c`, the jaw tips h above the floor, the point `height` above the floor. The point under the
    gripper is where the taught brick's middle was (it was held between the jaws), carried with
    the TCP to this height and roll."""
    x0, y0, x1, y1 = geom.grasp_box
    taught = geom.wrist_point(
        ((x0 + x1) / 2, (y0 + y1) / 2), geom.grasp_target_height, prm.TAUGHT_OBJECT_HEIGHT_M.value
    )
    in_tcp = np.linalg.inv(geom.tcp_pose(geom.grasp_target_height)) @ np.append(taught, 1.0)
    under = (geom.tcp_pose(h, roll_deg) @ in_tcp)[:2]
    return geom.wrist_point(c, h, height, roll_deg)[:2] - under


def view_half_width(geom: Geometry, h: float, height: float) -> float:
    """Half the wrist view's narrower side, in metres, on a plane `height` above the floor with
    the jaw tips h above it: anything located further off can't be in view."""
    edges = [geom.wrist_point(uv, h, height) for uv in EDGE_MIDPOINTS]
    across, along = np.linalg.norm(edges[1] - edges[0]), np.linalg.norm(edges[3] - edges[2])
    return float(min(across, along) / 2)  # convention: half the view


def grasp_spot(geom: Geometry) -> np.ndarray:
    """The wrist-image spot under the gripper: the taught brick's box centre (it was held there)."""
    if geom.grasp_box is None:
        raise MissingCalibrationError(
            "calib.json 'wrist_grasp_target' has no box: run teach_grasp.py --write"
        )
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
        raise MissingCalibrationError(
            "calib.json has no 'wrist_grasp_target': run teach_grasp.py (the owner's midpoint teach)"
        )
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
