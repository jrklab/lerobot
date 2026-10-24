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

"""The overhead camera: a USB camera on this Mac looking roughly straight down at the floor.

No stored calibration. Each scan measures what it needs with the robot's own motion: the
robot is found in a frame, drives a short known distance forward (wheel odometry), and is
found again. The displacement in the image is the robot's heading there, and its length
against the odometry distance is the scale (metres per pixel). Objects then get a bearing
(the angle from that heading to the robot-to-object line) and a range. This holds while the
camera looks roughly straight down; moving or re-aiming it between scans doesn't matter.
The front-camera approach and the wrist-camera check close the remaining error.
"""

from __future__ import annotations

import math
import subprocess
import time

import cv2
import numpy as np

from gemma import Gemma

ROBOT = "the small wheeled robot"
# A camera other than the Mac's own must be attached for the overhead scan to be tried.
BUILTIN_CAMERA_WORDS = ("FaceTime",)
TILE_FRACTION = 0.575  # 2x2 tiles of this size (15% overlap), for small objects
CANDIDATE_INDICES = (0, 1, 2)  # macOS camera indices to try (order changes as devices come and go)


def external_camera_attached() -> bool:
    """True if macOS lists a camera that isn't the built-in one."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    names = [l.strip().rstrip(":") for l in out.splitlines() if l.startswith("    ") and not l.startswith("      ") and l.strip()]
    return any(not any(w in n for w in BUILTIN_CAMERA_WORDS) for n in names)


def encode(img: np.ndarray, quality: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def grab(index: int, warmup: int = 12) -> np.ndarray | None:
    """One fresh frame from camera `index` (opened and released each time: it isn't needed
    between scans, and macOS reorders indices when devices come and go)."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    frame = None
    for _ in range(warmup):  # auto-exposure settles over the first frames
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.02)
    cap.release()
    return frame


def center(box) -> np.ndarray:
    """A box's centre in image fractions. The camera looks straight down, so that's over the
    robot's base, or over an object's middle (a cup's rim centre, a few cm off its base)."""
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def find_robot(gemma: Gemma, frame: np.ndarray):
    """Gemma's box of the robot (image fractions), and whether it lies fully inside the frame."""
    a = gemma.box(encode(frame), ROBOT)
    if a.value is None:
        return None, False
    x0, y0, x1, y1 = a.value
    return a.value, (x0 > 0.005 and y0 > 0.005 and x1 < 0.995 and y1 < 0.995)


def find_boxes(gemma: Gemma, frame: np.ndarray, what: str) -> list[dict]:
    """Candidate boxes of `what` (full-frame image fractions), from the full frame and from
    2x2 tiles, duplicates merged; "views" counts how many of those looks found it. No
    judgement here: the caller classifies each crop."""
    h, w = frame.shape[:2]
    tw, th = int(w * TILE_FRACTION), int(h * TILE_FRACTION)
    looks = [(0, 0, w, h)] + [(tx, ty, tw, th) for ty in (0, h - th) for tx in (0, w - tw)]
    found: list[dict] = []
    for tx, ty, vw, vh in looks:
        a = gemma.box(encode(frame[ty : ty + vh, tx : tx + vw]), what)
        if a.value is None:
            continue
        b = a.value
        full = [(tx + b[0] * vw) / w, (ty + b[1] * vh) / h, (tx + b[2] * vw) / w, (ty + b[3] * vh) / h]
        if (full[2] - full[0]) > 0.5 or (full[3] - full[1]) > 0.5:
            continue  # a tile-sized box is not an object
        for f in found:
            if iou(full, f["box"]) > 0.3:
                f["views"] += 1
                break
        else:
            found.append({"box": full, "views": 1})
    return found


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def relative_bearing(heading_px: np.ndarray, to_obj_px: np.ndarray) -> float:
    """Turn (deg, left positive like the robot's theta) from the robot's heading to the
    object, both as image vectors in pixels (x right, y down). A camera looking down sees
    the floor as from above, not mirrored, so a clockwise angle on screen is a right turn."""
    cross = heading_px[0] * to_obj_px[1] - heading_px[1] * to_obj_px[0]
    dot = float(heading_px @ to_obj_px)
    return -math.degrees(math.atan2(cross, dot))
