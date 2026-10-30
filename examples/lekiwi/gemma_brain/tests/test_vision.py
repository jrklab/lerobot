
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

"""Tests for jaw_geometry on synthetic wrist frames: two blue jaws entering from the bottom."""

import cv2
import numpy as np
from common.vision import jaw_geometry

W, H = 640, 480
BLUE = (200, 40, 20)  # BGR


def frame(moving_right: int, brick: tuple[int, int, int, int] | None = None) -> bytes:
    img = np.full((H, W, 3), 170, np.uint8)
    cv2.rectangle(img, (360, 362), (470, H - 1), BLUE, -1)  # fixed jaw
    cv2.rectangle(img, (moving_right - 60, 375), (moving_right, H - 1), BLUE, -1)
    if brick:
        cv2.rectangle(img, brick[:2], brick[2:], BLUE, -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_measures_both_jaws():
    g = jaw_geometry(frame(moving_right=140))
    assert abs(g["fixed_inner_u"] - 360 / W) < 0.005 and abs(g["fixed_tip_v"] - 362 / H) < 0.005
    assert abs(g["moving_inner_u"] - 140 / W) < 0.005
    assert abs(g["opening_u"] - 220 / W) < 0.01


def test_blue_brick_touching_fixed_jaw_does_not_move_it_when_calibrated():
    fixed = {"fixed_tip_v": 362 / H, "fixed_inner_u": 360 / W}
    jpeg = frame(moving_right=140, brick=(300, 300, 380, 365))
    assert jaw_geometry(jpeg)["fixed_tip_v"] < 0.7  # uncalibrated: the brick joins the jaw
    g = jaw_geometry(jpeg, fixed)
    assert g["fixed_inner_u"] == fixed["fixed_inner_u"]
    assert abs(g["moving_inner_u"] - 140 / W) < 0.005


def test_brick_apart_from_the_jaws_is_not_a_jaw():
    fixed = {"fixed_tip_v": 362 / H, "fixed_inner_u": 360 / W}
    g = jaw_geometry(frame(moving_right=140, brick=(200, 250, 280, 300)), fixed)
    assert abs(g["moving_inner_u"] - 140 / W) < 0.005


# --- an object's colour, sampled from the main model's box: no hue table ---

from common.vision import look_matches, measured_angle_of, sample_look  # noqa: E402

FLOOR_BGR = (165, 185, 200)  # the beige floor
LIME_BGR = (120, 215, 175)  # a pale lime brick: low saturation, Oct 30's failing one
RED_BGR = (40, 40, 210)


def _floor_with(bricks) -> bytes:
    img = np.full((H, W, 3), FLOOR_BGR, np.uint8)
    for (cx, cy), angle, colour in bricks:  # a 2x4 brick seen from above: 60 x 120 px
        box = cv2.boxPoints(((cx, cy), (60, 120), angle)).astype(np.int32)
        cv2.fillPoly(img, [box], colour)
    # A camera's pixel noise (~3 levels): what an object's colour scatter is measured against.
    noise = np.random.default_rng(0).normal(0, 3, img.shape)
    return cv2.imencode(".jpg", np.clip(img + noise, 0, 255).astype(np.uint8))[1].tobytes()


def test_a_pale_brick_gets_its_angle_and_only_its_own_box_matches():
    jpeg = _floor_with([((200, 220), 30.0, LIME_BGR), ((470, 220), 0.0, RED_BGR)])
    # boxes as tight as the model's: the 30 deg brick spans x 144-256, y 153-287 px
    lime_box, red_box, floor_box = [0.22, 0.31, 0.41, 0.61], [0.68, 0.33, 0.79, 0.59], [0.3, 0.75, 0.5, 0.95]
    look = sample_look(jpeg, lime_box)
    assert look is not None and look.distinct()
    angle, how = measured_angle_of(jpeg, lime_box, look)
    assert how == "fit" and abs(abs(angle) - 30.0) < 4.0, angle  # the old hue-table mask: no angle
    assert look_matches(jpeg, lime_box, look)[0]
    assert not look_matches(jpeg, red_box, look)[0]
    assert not look_matches(jpeg, floor_box, look)[0]


def test_an_object_colour_cannot_tell_from_the_floor_passes():
    # The limiting case: a "sample" of bare floor (an object of the floor's very colour). Colour
    # can't decide, so it doesn't: the main model's box stands.
    jpeg = _floor_with([])
    look = sample_look(jpeg, [0.4, 0.37, 0.6, 0.63])
    assert look is None or not look.distinct()
    assert look_matches(jpeg, [0.1, 0.1, 0.3, 0.3], look)[0]
