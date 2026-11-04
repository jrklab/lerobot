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

"""calibration/calibrate_gripper.py's close lags: what a stall-lag threshold must sit between."""

import pytest
from calibration.calibrate_gripper import close_lags


def _close(stop_deg, lag=1.5, speed=39.0, dt=0.05, n=30):
    """A close from 40 deg at `speed`, the jaws `lag` behind their target, stopped at `stop_deg`."""
    trace = []
    for k in range(n):
        tgt = max(0.0, 40.0 - speed * dt * k)
        trace.append({"t": k * dt, "joints": {"arm_gripper": max(stop_deg, tgt + lag)}, "targets": {"arm_gripper": tgt}})
    return trace


def test_a_close_on_an_object_reports_its_free_and_stall_lags():
    lags = close_lags(_close(stop_deg=22.0)[:15])  # the Pi ends the close soon after the stall
    assert lags["free_lag_max_deg"] == 1.5 and lags["stall_lag_deg"] > 4.0


def test_too_short_a_trace_reports_nothing():
    assert close_lags(_close(22.0)[:2]) == {}


def test_a_brick_side_snaps_to_its_lego_width():
    # The camera reads a side +/-1 mm (Nov 4: 14.8-16.7 for the 15.8 mm side): whole studs.
    from calibration import calibrate_gripper as cg

    assert [cg.lego_width_mm(m) for m in (0.0148, 0.0167, 0.0237, 0.0312, 0.0325)] == [15.8, 15.8, 23.8, 31.8, 31.8]


def test_the_curve_keeps_the_widths_this_run_did_not_measure():
    from calibration import calibrate_gripper as cg

    old = [{"mm": 0.0, "deg": 1.3}, {"mm": 15.8, "deg": 9.0}, {"mm": 47.8, "deg": 33.0}]
    new = [{"mm": 0.0, "deg": 1.4}, {"mm": 15.8, "deg": 9.6}, {"mm": 31.8, "deg": 21.0}]
    assert cg.curve(cg.merged(new, old)) == [[1.4, 0.0], [9.6, 15.8], [21.0, 31.8], [33.0, 47.8]]


def test_the_stall_lag_is_proposed_between_moving_and_stalled():
    from calibration import calibrate_gripper as cg

    pts = [{"free_lag_max_deg": 2.1, "stall_lag_deg": 6.0, "stalled": True},
           {"free_lag_max_deg": 2.5, "stall_lag_deg": 5.5, "stalled": True}]
    assert cg.propose_stall_lag(pts)["proposed_deg"] == 4.0
    assert cg.propose_stall_lag([{"free_lag_max_deg": 6.0, "stall_lag_deg": 5.0, "stalled": True}]) is None


def test_the_brick_is_looked_for_past_the_boards_right_hand_edge():
    import numpy as np
    from calibration import boards, calibrate_gripper as cg

    b = boards.BOARDS["wrist"]
    corners = boards.object_points("wrist", np.arange(len(b.getChessboardCorners())))
    y = cg.brick_side_y(np.eye(4))
    assert y == pytest.approx(corners[:, 1].min() - b.getSquareLength() - cg.BRICK_PAST_EDGE_M)


def test_the_colour_finder_takes_the_brick_not_the_jaws_or_the_floor():
    import cv2
    import numpy as np
    from common import vision

    img = np.full((480, 640, 3), (150, 170, 190), np.uint8)  # a light wooden floor (BGR)
    img[300:480, 200:260] = (200, 40, 20)  # a blue jaw, from the bottom edge
    img[200:260, 300:420] = (40, 40, 200)  # the red brick
    found = vision.coloured_blobs(cv2.imencode(".jpg", img)[1].tobytes())
    assert len(found) == 1 and abs(found[0][0] - 300 / 640) < 0.01 and abs(found[0][3] - 260 / 480) < 0.01
