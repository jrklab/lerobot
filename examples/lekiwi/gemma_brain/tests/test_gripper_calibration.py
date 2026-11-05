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

"""calibration/calibrate_gripper.py: the gripper step's widths, curve, and brick finding."""

import pytest


def test_a_brick_side_snaps_to_its_lego_width():
    # The camera reads a side +/-1 mm (Nov 4: 14.8-16.7 for the 15.8 mm side): whole studs.
    from calibration import calibrate_gripper as cg

    assert [cg.lego_width_mm(m) for m in (0.0148, 0.0167, 0.0237, 0.0312, 0.0325)] == [15.8, 15.8, 23.8, 31.8, 31.8]


def test_the_curve_keeps_the_widths_this_run_did_not_measure():
    from calibration import calibrate_gripper as cg

    old = [{"mm": 0.0, "deg": 1.3}, {"mm": 15.8, "deg": 9.0}, {"mm": 47.8, "deg": 33.0}]
    new = [{"mm": 0.0, "deg": 1.4}, {"mm": 15.8, "deg": 9.6}, {"mm": 31.8, "deg": 21.0}]
    assert cg.curve(cg.merged(new, old)) == [[1.4, 0.0], [9.6, 15.8], [21.0, 31.8], [33.0, 47.8]]


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


def test_the_wide_end_is_measured_until_the_jaw_leaves_the_view(monkeypatch, tmp_path):
    import json

    from calibration import calibrate_gripper as cg

    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"wrist_jaws": {"value": {"fixed_tip_v": 0.75, "fixed_inner_u": 0.56}}}))
    monkeypatch.setattr(cg, "CALIB", calib)
    monkeypatch.setattr(cg, "JAW_SETTLE_S", 0.0)

    class R:
        deg = 0.0

        def gripper(self, deg):
            self.deg = deg

        def status(self):
            return {"joints": {"arm_gripper": self.deg}}

        def frame(self, cam, after=None):
            return (b"x", {})

    robot = R()
    # The jaw seen up to 70 deg, at 1.3 mm per degree; past it out of the view.
    monkeypatch.setattr(cg.vision, "jaw_geometry", lambda jpeg, fixed: {"moving_inner_u": 0.2} if robot.deg <= 70 else None)
    monkeypatch.setattr(cg, "jaw_opening", lambda geom, jaws, h: None if jaws is None else 0.0013 * robot.deg)
    sk = type("S", (), {"geom": None, "height": lambda self: 0.095})()
    pts = cg.wide_end(sk, robot, wide=130.0, start_deg=43.0)
    assert [p["deg"] for p in pts] == [43.0, 55.0, 67.0] and pts[-1]["mm"] == pytest.approx(87.1)
    assert all(p["what"] == "camera" for p in pts)
    # Kept separately from the brick points: re-measured each run, not carried over.
    old = [{"mm": 47.8, "deg": 34.67}, {"mm": 60.0, "deg": 45.0, "what": "camera"}]
    assert [p["mm"] for p in cg.merged(pts, old)] == [55.9, 71.5, 87.1, 47.8]


def test_the_quarter_turn_goes_the_way_nearer_zero():
    # Nov 4: from about -65 deg it turned to -155, and the centring there lost the brick twice.
    from calibration import calibrate_gripper as cg

    sk = type("S", (), {"roll_range": (-180.0, 180.0)})()
    assert cg._turned(sk, -65.0) == 25.0 and cg._turned(sk, 70.0) == -20.0
    sk.roll_range = (-160.0, 10.0)  # only one way fits: the range decides
    assert cg._turned(sk, -65.0) == -155.0
