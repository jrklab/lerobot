
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

from vision import jaw_geometry

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
