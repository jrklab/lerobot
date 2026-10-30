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

"""The three printed ChArUco boards (calibration/targets/, make_boards.py) and their detection.

    wrist  10x7 squares of 25 mm, markers 0-34   flat on the floor around the reach point
    front   7x5 squares of 35 mm, markers 35-51  upright in front, seen by both cameras
    side    7x5 squares of 35 mm, markers 52-68  upright to one side, for the front camera

Each board has its own marker IDs, so several can be in one view. A board's frame: origin at
its first chessboard corner, x along its rows, y along its columns, z out of the printed side
(OpenCV's convention); its inner corners are object points in metres in that frame.
"""

from __future__ import annotations

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_5X5_100
# name: (squares x, y, square m, marker m, first marker id); the same as make_boards.py
SPECS = {
    "wrist": ((10, 7), 0.025, 0.018, 0),
    "front": ((7, 5), 0.035, 0.026, 35),
    "side": ((7, 5), 0.035, 0.026, 52),
}
# A view of a board counts when it has at least this many inner corners: enough for a pose
# (4 are the minimum for a planar homography) with some to spare against a bad one.
MIN_CORNERS = 6


def _board(name: str):
    (sx, sy), sq, mk, first = SPECS[name]
    d = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard((sx, sy), sq, mk, d, np.arange(first, first + (sx * sy) // 2))


BOARDS = {name: _board(name) for name in SPECS}
_DETECTORS = {name: cv2.aruco.CharucoDetector(b) for name, b in BOARDS.items()}


def object_points(name: str, corner_ids) -> np.ndarray:
    """(N, 3) metres: the board's inner corners with these ids, in the board's frame."""
    return np.asarray(BOARDS[name].getChessboardCorners(), float)[np.asarray(corner_ids, int).ravel()]


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)


def detect(image) -> dict[str, dict]:
    """{board: {"ids": [...], "px": [[u, v], ...]}} for each board seen with at least
    MIN_CORNERS inner corners. image: a grey or BGR array, or JPEG bytes."""
    if isinstance(image, (bytes, bytearray)):
        image = decode(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = {}
    for name, det in _DETECTORS.items():
        corners, ids, _, _ = det.detectBoard(image)
        if ids is not None and len(ids) >= MIN_CORNERS:
            out[name] = {
                "ids": ids.ravel().astype(int).tolist(),
                "px": corners.reshape(-1, 2).astype(float).tolist(),
            }
    return out
