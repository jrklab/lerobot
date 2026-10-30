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

"""The camera model (brain/cameras.py) on this robot's measured cameras."""

import json
from pathlib import Path

import numpy as np
from brain.cameras import Cameras, on_plane, to_image

CAMS = Cameras.from_calib(json.loads((Path(__file__).parents[1] / "data" / "calib.json").read_text())["cameras"]["value"])
FLOOR = -0.112  # the floor surface in the arm's frame (Oct 29)


def test_a_floor_point_and_its_image_point_map_onto_each_other():
    for uv in ((0.5, 0.7), (0.1, 0.95), (0.9, 0.6)):  # below the front camera's horizon
        p = on_plane(CAMS.front, CAMS.T_base_cam, uv, FLOOR)
        assert p is not None and abs(p[2] - FLOOR) < 1e-9
        assert np.allclose(to_image(CAMS.front, CAMS.T_base_cam, p), uv, atol=1e-6)


def test_above_the_horizon_nothing_meets_the_floor_and_behind_nothing_is_seen():
    assert on_plane(CAMS.front, CAMS.T_base_cam, (0.5, 0.05), FLOOR) is None
    assert to_image(CAMS.front, CAMS.T_base_cam, CAMS.T_base_cam[:3, 3] - [0.5, 0, 0]) is None


def test_the_wrist_camera_rides_the_tcp():
    tcp = np.eye(4)
    tcp[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # pointing down
    tcp[:3, 3] = [0.25, 0.0, FLOOR + 0.12]
    cam = CAMS.wrist_pose(tcp)
    uv = to_image(CAMS.wrist, cam, [0.25, 0.0, FLOOR])  # the floor under the TCP
    assert uv is not None and 0 < uv[0] < 1 and 0 < uv[1] < 1
    moved = tcp.copy()
    moved[0, 3] += 0.02
    assert not np.allclose(to_image(CAMS.wrist, CAMS.wrist_pose(moved), [0.25, 0.0, FLOOR]), uv)
