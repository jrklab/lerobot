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

"""Container markers (calibration/targets/make_markers.py): ArUco markers stuck on a container's
side, measured by the front camera without the box models. Nov 6: the black shoe box's
model-measured height varied 9.8-11.5 cm between runs, and the fast model boxed other dark
things as it. A marker gives its face exactly: where its middle is and which way it faces.
"""

# ruff: noqa: N803 (T_base_cam, as brain/cameras.py)
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from brain.cameras import IMAGE_PX, Lens

DICTIONARY = cv2.aruco.DICT_5X5_100  # the calibration boards' (ids 0-68)
CONTAINER_MARKER_IDS = (90, 91)  # the shoe box's two long sides (make_markers.py)
MARKER_SIZE_M = 0.060  # the printed black square's side (make_markers.py; measure the print)
# Which containers carry which markers: a container whose name has the word (the planner's
# "shoe box", "black shoe box") is placed by its marker (chains.place_in_marked).
MARKED_CONTAINERS = {"box": CONTAINER_MARKER_IDS}
# A marker seen this narrow against its height (its face turned past ~60 deg from the camera)
# isn't measured: its two side edges nearly meet and their solve goes wild (Nov 6: id 90 at the
# frame's edge, ~10 px wide, read 13 cm from the camera, 83 mm of spread over 10 frames).
MIN_WIDTH_SHARE = 0.5  # cos 60 deg
MARGIN_M = 0.010  # the white margin printed round the marker (make_markers.py's MARGIN_MM)
# The lower share of the columns' edges taken as the face's top (face_top_z): a quarter of them
# over a background unlike the face is enough; the lowest tenth read 1 mm under the tape once.
FACE_TOP_SHARE = 0.25
EDGE_BLUR_PX = 3  # an edge spreads over ~3 rows: the 5x5 blur below and the camera's own
TOP_DELTA_E = 12.0  # a row is the same colour within this (Lab), as vision.snap_top's TOP_MIN_DELTA_E


def marker_ids(container: str) -> tuple[int, ...]:
    """The markers a container carries, by its name; () for an unmarked one."""
    words = container.lower().split()
    return next((ids for word, ids in MARKED_CONTAINERS.items() if word in words), ())


@dataclass
class MarkerPose:
    id: int
    centre: np.ndarray  # the marker's middle, arm frame (m)
    normal: np.ndarray  # unit, out of its face towards the camera, arm frame
    corners: np.ndarray  # 4x2 image points (fractions): top left, top right, bottom right, bottom left
    distance_m: float  # from the camera

    def heading_deg(self) -> float:
        """The face's facing in the floor plane, degrees from the arm's -x (a face square to the
        robot faces straight back at it: 0; positive turned to the left)."""
        return float(np.degrees(np.arctan2(-self.normal[1], -self.normal[0])))


def find_markers(img: np.ndarray, lens: Lens, T_base_cam: np.ndarray, ids=CONTAINER_MARKER_IDS,
                 size_m: float = MARKER_SIZE_M) -> list[MarkerPose]:
    """The listed markers in a front-camera image (BGR or grey, 640x480), each as a pose in the arm
    frame. The marker stands upright on a container's side, so each of its two side edges is a
    vertical segment size_m long: along its two corners' rays (the lens, distortion included)
    that fixes both ends. A full pose solve (solvePnP) on a marker ~40 px across put its middle
    10-20 mm long at 45-60 cm and its facing off by up to 180 deg (synthetic, Nov 6)."""
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    corners, found, _ = detector.detectMarkers(grey)
    if found is None:
        return []
    rot, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    up_cam = rot.T @ np.array([0.0, 0.0, 1.0])
    out = []
    for c, mid in zip(corners, found.ravel(), strict=True):
        if int(mid) not in ids:
            continue
        q = c.reshape(4, 2).astype(np.float64)  # the marker's own top left, top right, bottom right, bottom left
        # Stuck on in any of four turns (Nov 6: id 90 went on a quarter turn round): the image's
        # top left, top right, bottom right, bottom left, the corners kept in their order round it.
        q = np.roll(q, -max(range(4), key=lambda k: (lambda r: r[3, 1] - r[0, 1] + r[2, 1] - r[1, 1])(np.roll(q, -k, axis=0))), axis=0)
        wide = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        tall = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if wide < MIN_WIDTH_SHARE * tall:
            continue
        uv = q / IMAGE_PX
        rays = [lens.ray(p) for p in uv]
        pts = [None] * 4
        for top, bottom in ((0, 3), (1, 2)):
            # t_top * ray_top - t_bottom * ray_bottom = size_m * up
            (t_top, t_bottom), *_ = np.linalg.lstsq(np.c_[rays[top], -rays[bottom]], size_m * up_cam, rcond=None)
            pts[top], pts[bottom] = t_top * rays[top], t_bottom * rays[bottom]
        p_cam = np.array(pts)
        across = rot @ ((p_cam[1] + p_cam[2]) - (p_cam[0] + p_cam[3])) / 2  # left to right, seen from the front
        across[2] = 0.0
        normal = np.cross(across, [0.0, 0.0, 1.0])  # out of the face, towards whoever reads it
        normal /= np.linalg.norm(normal)
        centre_cam = p_cam.mean(axis=0)
        out.append(MarkerPose(
            id=int(mid),
            centre=rot @ centre_cam + o,
            normal=normal,
            corners=uv,
            distance_m=float(np.linalg.norm(centre_cam)),
        ))
    return out


def face_top_z(img: np.ndarray, m: MarkerPose, lens: Lens, T_base_cam: np.ndarray) -> float | None:
    """The arm-frame height of the top edge of the face the marker is stuck on. Each image
    column across the face (the marker's width and as much again each side) is walked up from
    just above the marker's white margin while it keeps the face's colour; the edge is the lower
    FACE_TOP_SHARE of where they end, and its ray meets the face's plane there. The front camera
    sits below the box's top and sees the edge against what's behind it: a background the face's
    colour only carries the walk on upwards, never stops it short, so the columns over a
    contrasting background give the edge (Nov 11: one column above the marker, over a black bag,
    read the 11.5 cm box at 13.4-17.8 cm; the lower quarter of all columns 11.6-12.8). Columns
    that don't start on the face's colour (off the face) and ones that reach the frame's top
    (the face may go on past it) don't count; None if none is left."""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(float)
    h, w = lab.shape[:2]
    px = m.corners * IMAGE_PX
    x0, x1 = px[:, 0].min(), px[:, 0].max()
    top = min(px[0, 1], px[1, 1])
    tall = max(px[3, 1], px[2, 1]) - top
    y0 = int(top - tall * MARGIN_M / MARKER_SIZE_M) - 2 * EDGE_BLUR_PX  # on the face, above the margin
    if y0 < 2 * EDGE_BLUR_PX:
        return None
    ref = np.median(lab[y0 - 2 : y0 + 1, int(x0) : int(x1)].reshape(-1, 3), axis=0)  # the face's colour
    rows = []
    for c in range(max(0, int(2 * x0 - x1)), min(w, int(2 * x1 - x0))):
        if np.linalg.norm(lab[y0, c] - ref) > TOP_DELTA_E:
            continue  # off the face
        y = y0
        while y - 1 >= 0 and np.linalg.norm(lab[y - 1, c] - ref) <= TOP_DELTA_E:
            y -= 1
        if y >= 2 * EDGE_BLUR_PX:
            rows.append(y)
    if not rows:
        return None
    edge = float(np.percentile(rows, 100 * (1 - FACE_TOP_SHARE)))  # rows count down: the lower edges
    rot, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d = rot @ lens.ray(((x0 + x1) / 2 / IMAGE_PX[0], edge / IMAGE_PX[1]))
    t = float((m.centre - o) @ m.normal) / float(d @ m.normal)
    return float(o[2] + t * d[2]) if t > 0 else None
