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

"""Classical image geometry on top of Gemma's boxes: Gemma finds the brick (an axis-aligned
box), OpenCV measures what a box can't say, its in-plane orientation."""

from __future__ import annotations

import math

import cv2
import numpy as np


def brick_angle(jpeg: bytes, box, pad: float = 0.25) -> dict | None:
    """Long-axis angle of the brick inside `box` (x0, y0, x1, y1 image fractions), in degrees
    from image vertical, in (-90, 90], positive = clockwise in the image (top leaning right).

    The crop around the box is segmented by colour distance from the floor around it (the
    median L*a*b* of the crop's border ring), Otsu-thresholded, and the largest blob's
    minimum-area rectangle gives the angle. Works for any brick colour that differs from the
    floor. Returns None if no plausible blob is found.
    """
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, (x0 - pad * bw) * w)), int(max(0, (y0 - pad * bh) * h))
    X1, Y1 = int(min(w, (x1 + pad * bw) * w)), int(min(h, (y1 + pad * bh) * h))
    crop = img[Y0:Y1, X0:X1]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    lab = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(float), (5, 5), 0)
    ring = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3), lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    score = np.linalg.norm(lab - bg, axis=2)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (cx, cy), (rw, rh), a = cv2.minAreaRect(c)
    # OpenCV 4.5+: a in (0, 90], rw measured along the rotated x axis.
    long_angle = a if rw < rh else a - 90.0
    long_angle = (long_angle + 90.0) % 180.0 - 90.0
    box_area = bw * w * bh * h
    return {
        "angle_deg": round(float(long_angle), 1),
        "aspect": round(float(max(rw, rh) / max(1.0, min(rw, rh))), 2),
        "fill": round(float(cv2.contourArea(c) / max(1.0, box_area)), 2),
        "center": [round((X0 + cx) / w, 4), round((Y0 + cy) / h, 4)],
    }


def plausible_brick(jpeg: bytes, box, target: str = "") -> tuple[bool, str]:
    """Is Gemma's box really a brick on the floor in a wrist frame? Gemma's own yes/no is
    unreliable here both ways (2026-10-23: false negatives on a plainly visible brick, and a
    box invented on the blue gripper jaw when nothing was there), so check geometry instead:
    the box must not touch the image border and must be mostly free of the jaws' saturated
    blue. (A brick-shaped-blob test was tried too and rejected: the whiteness segmentation
    spills onto the light floor in about half of the real-brick frames.)"""
    x0, y0, x1, y1 = box
    if min(x0, y0) < 0.01 or max(x1, y1) > 0.99:
        return False, "box touches the image border"
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    roi = img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 120) & (hsv[..., 2] > 50)
    # The jaws only show at the bottom of the wrist image (tips at ~0.77 of its height), and a
    # blue target is blue like them (Oct 24: a blue brick in mid-image was rejected as "56%
    # gripper blue"), so the test only applies low in the frame and for non-blue targets.
    low = (y0 + y1) / 2 > 0.65
    if low and "blue" not in target.lower() and blue.mean() > 0.15:
        return False, f"box is {100 * blue.mean():.0f}% gripper blue"
    return True, "ok"


def crop_around(jpeg: bytes, box, pad: float = 0.6, min_px: int = 160) -> bytes:
    """JPEG of the box with `pad` x its size of context on each side, upscaled so its short
    side is at least min_px (far objects are only ~20 px across in the front camera)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 0.05), max(y1 - y0, 0.05)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(max(0, (cx - (0.5 + pad) * bw) * w)), int(max(0, (cy - (0.5 + pad) * bh) * h))
    X1, Y1 = int(min(w, (cx + (0.5 + pad) * bw) * w)), int(min(h, (cy + (0.5 + pad) * bh) * h))
    c = img[Y0:Y1, X0:X1]
    scale = max(1.0, min_px / max(1, min(c.shape[:2])))
    c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", c)[1].tobytes()


def jaw_tip_row(jpeg: bytes) -> float | None:
    """Image row (fraction, 0 = top) of the gripper's jaw tips in a wrist frame: the highest
    point of the two largest saturated-blue blobs in the lower half (the jaws are blue and
    enter from the bottom of the image). The average of both tips, or the one found."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 120) & (hsv[..., 2] > 50)).astype(np.uint8)
    blue[: h // 2] = 0
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    blobs = sorted((stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] > 0.004 * h * w), key=lambda s: -s[cv2.CC_STAT_AREA])[:2]
    if not blobs:
        return None
    return float(np.mean([b[cv2.CC_STAT_TOP] for b in blobs])) / h


def outside_fraction(box, tip_row: float) -> float:
    """Share of the brick's box (along the image vertical: its length, with the wrist rolled
    to put the long side vertical) that lies beyond the jaw tips, i.e. outside the gripper."""
    y0, y1 = box[1], box[3]
    if y1 <= y0:
        return 0.0
    return float(np.clip((tip_row - y0) / (y1 - y0), 0.0, 1.0))


def snap_box(jpeg: bytes, box, min_chroma: float = 25.0, what: str = "") -> tuple[list[float], str]:
    """Gemma's box moved onto the coloured blob it was aimed at. E4B's wrist boxes can land a
    brick-height off the brick (Oct 24, red brick next to the jaws: the box on bare floor just
    above it, while the centring chased it); the brick itself is plain to colour segmentation.

    Searches a window of the box padded by one box size each way for strongly coloured pixels
    (L*a*b* chroma, not the jaws' blue, not dark), and takes the blob that overlaps the box
    most, else the nearest one. The box is replaced by that blob's bounding box when the blob
    is a plausible size for it (0.3x-3x the area); otherwise the box comes back unchanged.

    `what` naming a colour ("blue block") skips blobs of another colour: Oct 26, lining up a
    held red brick over a blue one, the blue brick was masked out as jaw and the box snapped
    onto the held red brick. A saturated blue target is still masked as jaw, so its box comes
    back unsnapped. Returns (box, how)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return list(box), "no image"
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, (x0 - bw) * w)), int(max(0, (y0 - bh) * h))
    X1, Y1 = int(min(w, (x1 + bw) * w)), int(min(h, (y1 + bh) * h))
    crop = img[Y0:Y1, X0:X1]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return list(box), "empty window"
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(float)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    jaw = (hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 120)
    words = [c for c in HUES if c in what.lower()]
    mask = ((chroma > min_chroma) & ~jaw & (lab[..., 0] > 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return list(box), "no coloured blob"
    bx0, by0, bx1, by1 = x0 * w - X0, y0 * h - Y0, x1 * w - X0, y1 * h - Y0
    best, best_key = None, None
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if area < 0.1 * bw * w * bh * h:
            continue
        # The whole blob's colour, not each pixel's: a brick's shaded side drifts out of the hue
        # range while the blob as a whole stays in it.
        if words and not _hue_matches(hsv[labels == i], words)[0]:
            continue
        ov = max(0, min(bx1, sx + sw) - max(bx0, sx)) * max(0, min(by1, sy + sh) - max(by0, sy))
        d = math.hypot(cents[i][0] - (bx0 + bx1) / 2, cents[i][1] - (by0 + by1) / 2)
        key = (ov > 0, ov, -d)
        if best_key is None or key > best_key:
            best, best_key = i, key
    if best is None:
        return list(box), "blobs too small"
    sx, sy, sw, sh, _ = stats[best]
    ratio = sw * sh / max(1.0, bw * w * bh * h)
    if not 0.3 <= ratio <= 3.0:
        return list(box), f"blob {ratio:.1f}x the box"
    return [(X0 + sx) / w, (Y0 + sy) / h, (X0 + sx + sw) / w, (Y0 + sy + sh) / h], "snapped"


# OpenCV hue (0-180) ranges of the colour words a goal can use for an object.
HUES = {"red": [(0, 10), (165, 180)], "pink": [(150, 175)], "orange": [(10, 20)], "yellow": [(18, 30)],
        "green": [(30, 90)], "blue": [(95, 130)], "purple": [(130, 150)]}


def colour_matches(jpeg: bytes, box, what: str) -> tuple[bool, str]:
    """Does the box hold the colour `what` names ("light green block" -> green)? The median
    hue of its strongly coloured pixels (not the floor's) against HUES. With no colour word
    in `what`, or too few coloured pixels to tell, it passes. Oct 25: asked for the light
    green block, Gemma boxed the red one, and the kind-of-object check accepted it."""
    words = [w for w in HUES if w in what.lower()]
    if not words:
        return True, "no colour named"
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    roi = img[int(y0 * h) : max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w) : max(int(x0 * w) + 1, int(x1 * w))]
    return _hue_matches(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3), words)


def _hue_matches(hsv: np.ndarray, words: list[str]) -> tuple[bool, str]:
    """Is the median hue of these HSV pixels' strongly coloured ones (N x 3) one of `words`'
    colours? Too few coloured pixels to tell passes."""
    sat = hsv[:, 1] > 80
    if sat.sum() < 15:
        return True, "too few coloured pixels to tell"
    hue = float(np.median(hsv[:, 0][sat]))
    # Red wraps around 0/180: take the median on the circle near 0 when most pixels are red.
    reds = ((hsv[:, 0] < 10) | (hsv[:, 0] > 165))[sat].mean()
    if reds > 0.5:
        hue = 0.0
    ok = any(lo <= hue <= hi for word in words for lo, hi in HUES[word])
    return ok, f"hue {hue:.0f} for {'/'.join(words)}"
