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

"""calibration/drive_test.py's arithmetic on a simulated base: a pan error turns every drive the
same way as the front camera sees it; wheel errors turn each direction its own way."""

import math

import numpy as np
import pytest
from calibration import drive_test as dt

CENTRE = np.array([-0.030, 0.0])  # the base's turning centre in the arm's frame


class Base:
    """The arm frame's origin as the front camera measures it, against the start. pan_err: a pan
    zero that self_calibrate.py correct shoulder_pan=+pan_err fixes turns the drives the cameras
    see by +pan_err (Nov 3, measured); wheel: per
    command direction, the angle (deg) the base actually drives off it; turn: deg it turns per leg."""

    def __init__(self, pan_err=0.0, wheel=None, turn=0.0, scale=1.0):
        self.pos, self.head = np.zeros(2), 0.0
        self.pan_err, self.wheel, self.turn, self.scale = pan_err, wheel or {}, turn, scale

    def base(self, dx=0.0, dy=0.0):
        cmd = np.array([dx, dy])
        key = {(-1, 0): "back", (1, 0): "forward", (0, 1): "left", (0, -1): "right"}[
            (int(np.sign(dx)), int(np.sign(dy)))
        ]
        drive = dt._rot(self.head + self.wheel.get(key, 0.0)) @ cmd * self.scale  # in the true frame
        self.pos = self.pos + dt._rot(self.pan_err) @ drive + (np.eye(2) - dt._rot(self.turn)) @ CENTRE
        self.head += self.turn

    def measure(self):
        return float(self.pos[0]), float(self.pos[1]), self.head


def _run(base, repeats=2):
    return dt.verdict(dt.drive_legs(base, base.measure, 0.2, repeats, CENTRE))


def test_a_pan_error_turns_every_direction_the_same_and_is_named():
    v = _run(Base(pan_err=3.2))
    assert v["same_every_way"] and v["pan_offset_deg"] == pytest.approx(3.2, abs=0.05)
    assert all(a == pytest.approx(3.2, abs=0.05) for a in v["by_direction_deg"].values())


def test_wheel_errors_turn_each_direction_its_own_way_and_name_no_pan_correction():
    v = _run(Base(wheel={"back": 3.0, "forward": -2.0, "left": 4.0, "right": 0.5}))
    assert (
        not v["same_every_way"]
        and v["pan_offset_deg"] is None
        and v["spread_deg"] == pytest.approx(6.0, abs=0.1)
    )


def test_a_turn_during_a_leg_is_taken_off_the_drive():
    legs = dt.drive_legs(*(lambda b: (b, b.measure))(Base(turn=1.5)), 0.2, 1, CENTRE)
    assert all(abs(leg["angle_deg"]) < 0.05 and leg["turned_deg"] == pytest.approx(1.5) for leg in legs)


def test_the_distance_ratio():
    legs = dt.drive_legs(*(lambda b: (b, b.measure))(Base(scale=0.97)), 0.2, 1, CENTRE)
    assert all(leg["ratio"] == pytest.approx(0.97, abs=1e-3) for leg in legs)


def test_angles_wrap_round():
    r = dt.leg_result(
        (0.0, 0.0, 0.0),
        (-0.2 * math.cos(math.radians(179)), 0.2 * math.sin(math.radians(179)), 0.0),
        (0.2, 0.0),
        np.zeros(2),
    )
    assert -180.0 <= r["angle_deg"] <= 180.0


def test_one_bad_leg_doesnt_sway_a_direction_with_three_repeats():
    # Nov 3's eight legs (the first back +8.3 deg), plus a third repeat in line with the rest.
    angles = {
        "back": [8.3, 1.7, 2.6],
        "forward": [2.8, 3.2, 2.9],
        "left": [3.1, 2.6, 2.8],
        "right": [3.0, 2.0, 2.7],
    }
    legs = [{"dir": d, "repeat": k, "angle_deg": a} for d, v in angles.items() for k, a in enumerate(v)]
    v = dt.verdict(legs)
    assert v["same_every_way"] and v["by_direction_deg"]["back"] == pytest.approx(2.6)
    assert v["pan_offset_deg"] == pytest.approx(2.75, abs=0.01)
    assert v["outliers"] == [{"dir": "back", "repeat": 0, "angle_deg": 8.3}] and v["repeats"] == 3


def _legs(angles):
    return [{"dir": d, "repeat": k, "angle_deg": a} for d, v in angles.items() for k, a in enumerate(v)]


def test_scattered_legs_widen_what_counts_as_the_same_angle():
    # Nov 4, after the pan correction: medians -1.5..+1.2 (spread 2.7), legs scattered ~1.3 deg.
    v = dt.verdict(
        _legs(
            {
                "back": [-0.43, -1.53, -4.23],
                "forward": [1.56, 1.16, -3.55],
                "left": [2.4, -0.09, -0.83],
                "right": [-0.3, -1.31, -0.93],
            }
        )
    )
    assert v["spread_deg"] > dt.SAME_ANGLE_DEG and v["same_every_way"]
    assert v["pan_offset_deg"] == pytest.approx(-0.34, abs=0.01)


def test_tight_legs_keep_the_floor_and_still_tell_the_wheels():
    # Each direction's legs agree within 0.2 deg but the directions differ by 3: the wheels.
    v = dt.verdict(
        _legs(
            {
                "back": [3.0, 3.1, 2.9],
                "forward": [0.0, 0.1, -0.1],
                "left": [1.0, 1.1, 0.9],
                "right": [2.0, 1.9, 2.1],
            }
        )
    )
    assert v["tolerance_deg"] == dt.SAME_ANGLE_DEG and not v["same_every_way"]
