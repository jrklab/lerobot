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


def brick_angle(jpeg: bytes, box, pad: float = 0.25, what: str = "") -> dict | None:
    """Long-axis angle of the brick inside `box` (x0, y0, x1, y1 image fractions), in degrees
    from image vertical, in (-90, 90], positive = clockwise in the image (top leaning right).

    The crop around the box is segmented by colour distance from the floor around it (the
    median L*a*b* of the crop's border ring), Otsu-thresholded, and the largest blob's
    minimum-area rectangle gives the angle. Works for any brick colour that differs from the
    floor. When `what` names a colour (HUES), the blob is that colour's pixels instead: the
    floor-distance mask also takes in the brick's shadow (Oct 27, run 20261027T032148: a brick
    ~20 deg off square came out square with its shadow, fill 1.45, and fell back to 0 deg).
    Returns None if no plausible blob is found.
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
    words = [word for word in HUES if word in what.lower()]
    mask = None
    if words:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, strong = hsv[..., 0], (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
        mask = np.zeros(hue.shape, bool)
        for word in words:
            for lo, hi in HUES[word]:
                mask |= strong & (hue >= lo) & (hue <= hi)
        mask = mask.astype(np.uint8) * 255 if mask.sum() >= 30 else None
    if mask is None:
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


def jaw_geometry(jpeg: bytes, fixed: dict | None = None, band: float = 0.06) -> dict | None:
    """Where the gripper's jaws are in a wrist frame, as image fractions: for the fixed jaw
    (rigid with the camera, on the right) and the moving one (left), the tip row (top of its
    blob) and the inner edge column in a band of `band` of the height just below that tip; and
    the opening between the inner edges. Jaws are saturated-blue blobs in the lower half that
    reach the bottom edge (they enter from below; a blue brick lying apart from them doesn't).
    `fixed` (calib.json "wrist_jaws": fixed_tip_v, fixed_inner_u) replaces the fixed jaw's
    measurement, since it can't move in the image, and keeps a blue brick touching it from
    shifting it. None if no fixed jaw; the moving jaw's keys are None when it isn't seen."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 120) & (hsv[..., 2] > 50)).astype(np.uint8)
    blue[: h // 2] = 0
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(blue)
    blobs = [i for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] > 0.004 * h * w
             and stats[i][cv2.CC_STAT_TOP] + stats[i][cv2.CC_STAT_HEIGHT] >= h - 2]

    def jaw(i: int, inner_is_left: bool) -> tuple[float, float]:
        tip = stats[i][cv2.CC_STAT_TOP]
        rows = labels[tip : tip + max(1, int(band * h))] == i
        cols = np.nonzero(rows.any(axis=0))[0]
        return tip / h, (cols.min() if inner_is_left else cols.max()) / w

    if fixed is not None:
        fixed_tip, fixed_inner = fixed["fixed_tip_v"], fixed["fixed_inner_u"]
        left = [i for i in blobs if (stats[i][cv2.CC_STAT_LEFT] + stats[i][cv2.CC_STAT_WIDTH]) / w < fixed_inner - 0.01]
    else:
        blobs = sorted(blobs, key=lambda i: -stats[i][cv2.CC_STAT_AREA])[:2]
        if not blobs:
            return None
        blobs.sort(key=lambda i: cents[i][0])  # left to right
        fixed_tip, fixed_inner = jaw(blobs[-1], inner_is_left=True)
        left = blobs[:-1]
    out = {"fixed_tip_v": fixed_tip, "fixed_inner_u": fixed_inner,
           "moving_tip_v": None, "moving_inner_u": None, "opening_u": None}
    if left:
        moving_tip, moving_inner = jaw(max(left, key=lambda i: stats[i][cv2.CC_STAT_AREA]), inner_is_left=False)
        out.update(moving_tip_v=moving_tip, moving_inner_u=moving_inner, opening_u=fixed_inner - moving_inner)
    return out


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
# A pixel is strongly coloured above this saturation (not the floor's grey-brown); a colour
# needs this many such pixels to tell.
STRONG_SAT = 80
MIN_COLOUR_PIXELS = 15
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
    sat = hsv[:, 1] > STRONG_SAT
    if sat.sum() < MIN_COLOUR_PIXELS:
        return True, "too few coloured pixels to tell"
    hue = float(np.median(hsv[:, 0][sat]))
    # Red wraps around 0/180: take the median on the circle near 0 when most pixels are red.
    reds = ((hsv[:, 0] < 10) | (hsv[:, 0] > 165))[sat].mean()
    if reds > 0.5:
        hue = 0.0
    ok = any(lo <= hue <= hi for word in words for lo, hi in HUES[word])
    return ok, f"hue {hue:.0f} for {'/'.join(words)}"


def measured_angle_of(jpeg: bytes, box, what: str = "") -> tuple[float | None, str]:
    """The object's long-side angle from image vertical and how it was measured, or (None,
    "unmeasured") when the colour blob isn't brick-shaped: v2 doesn't fall back to a guess (v1's
    box-shape fallback said 0 deg for bricks 15-22 deg off, Oct 27)."""
    m = brick_angle(jpeg, box, what=what)
    if m is not None and 0.4 <= m["fill"] <= 1.15 and m["aspect"] >= 1.4:
        return m["angle_deg"], "fit"
    return None, "unmeasured"

