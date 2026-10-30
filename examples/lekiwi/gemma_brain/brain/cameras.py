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

"""The two cameras as measured (calib.json "cameras": calibration/solve_cameras.py, the ChArUco
boards): an image point to a ray, a ray to a point on a level plane, and a point back to the
image. Image points are fractions of the 640x480 frame (u right, v down), as the detector's
boxes are; points and poses are in the arm's frame (so101_kinematics), metres.

    wrist   rides the gripper: its pose is the TCP's pose times T_tcp_cam
    front   fixed on the base: T_base_cam

No fitted image-space constants: what the old geometry fitted per camera (an image Jacobian at
one height, a descent point, a range curve, degrees per image width) follows from these.
"""

# ruff: noqa: N803, N806
# (N8: transforms and camera matrices keep their textbook capitals, as in calibration/solve_cameras.py)

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_PX = np.array([640.0, 480.0])  # convention: the cameras' 640x480 frames
UNDISTORT_CRITERIA = (
    cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
    50,
    1e-10,
)  # convention: to convergence


def _undistort(px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Normalized image coordinates, iterated to UNDISTORT_CRITERIA: OpenCV 4 has it as
    undistortPointsIter, OpenCV 5 (the brain's .venv) as undistortPoints' criteria."""
    if hasattr(cv2, "undistortPointsIter"):
        return cv2.undistortPointsIter(px, K, dist, None, None, UNDISTORT_CRITERIA)
    return cv2.undistortPoints(px, K, dist, criteria=UNDISTORT_CRITERIA)


@dataclass
class Lens:
    K: np.ndarray  # 3x3 camera matrix, pixels
    dist: np.ndarray  # OpenCV distortion coefficients

    def ray(self, uv) -> np.ndarray:
        """The unit direction, in the camera's frame (x right, y down, z forward), of the image point uv."""
        px = (np.asarray(uv, float) * IMAGE_PX).reshape(1, 1, 2)
        # Iterated to convergence: OpenCV's default few iterations leave 0.1 px near the corners
        # of these strongly barrel-distorted lenses (k1 ~ -0.43).
        x, y = _undistort(px, self.K, self.dist).ravel()
        d = np.array([x, y, 1.0])
        return d / np.linalg.norm(d)

    def project(self, p_cam) -> np.ndarray | None:
        """The image point (fractions) of a point in the camera's frame; None behind the camera."""
        p = np.asarray(p_cam, float).reshape(1, 3)
        if p[0, 2] <= 0:
            return None
        px, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), self.K, self.dist)
        return px.ravel() / IMAGE_PX


def inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def on_plane(lens: Lens, T_base_cam: np.ndarray, uv, z: float) -> np.ndarray | None:
    """Where the ray through image point uv meets the level plane at height z (arm frame);
    None when it doesn't (the ray points away from the plane: above the horizon)."""
    d = T_base_cam[:3, :3] @ lens.ray(uv)
    o = T_base_cam[:3, 3]
    if abs(d[2]) < 1e-9:  # convention: parallel to the plane
        return None
    t = (z - o[2]) / d[2]
    return o + t * d if t > 0 else None


def to_image(lens: Lens, T_base_cam: np.ndarray, p) -> np.ndarray | None:
    """The image point (fractions) of a point in the arm's frame; None behind the camera."""
    T = inv(T_base_cam)
    return lens.project(T[:3, :3] @ np.asarray(p, float) + T[:3, 3])


@dataclass
class Cameras:
    wrist: Lens
    T_tcp_cam: np.ndarray
    front: Lens
    T_base_cam: np.ndarray
    turn_centre: np.ndarray  # (x, y) the base turns about, in the arm's frame

    @classmethod
    def from_calib(cls, value: dict) -> Cameras:
        w, f = value["wrist"], value["front"]
        return cls(
            wrist=Lens(np.asarray(w["K"], float), np.asarray(w["dist"], float)),
            T_tcp_cam=np.asarray(w["T_tcp_cam"], float),
            front=Lens(np.asarray(f["K"], float), np.asarray(f["dist"], float)),
            T_base_cam=np.asarray(f["T_base_cam"], float),
            turn_centre=np.asarray(value["base_turn_centre"], float),
        )

    def wrist_pose(self, T_base_tcp: np.ndarray) -> np.ndarray:
        return T_base_tcp @ self.T_tcp_cam
