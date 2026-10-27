
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

"""Tests for the wrist servo's image goals (skills.descend, skills.grasp_goal)."""

import numpy as np
import skills


def test_descend_round_trip_and_growth():
    p = np.array([0.4, 0.7])
    down = skills.descend(p, 0.10, 0.03)
    assert np.allclose(skills.descend(down, 0.03, 0.10), p)
    # going down, a floor point moves away from where the descent heads
    assert np.linalg.norm(down - skills.DESCENT_P0) > np.linalg.norm(p - skills.DESCENT_P0)


def test_grasp_goal_puts_the_brick_beside_the_fixed_jaw_on_the_pad_middle():
    h = skills.GRASP_LIFT_M
    u, v = skills.top_to_tips(skills.grasp_goal(h), h)
    margin_u = skills.GRASP_MARGIN_M * np.linalg.norm(skills.jacobian(0.0)[0])
    assert np.isclose(u, skills.FIXED_JAW_INNER_U - margin_u)
    assert np.isclose(v, skills.PAD_MID_V)
    assert skills.PAD_TOP_V < skills.PAD_MID_V < skills.PAD_BOTTOM_V
    assert 0 < skills.GRASP_MARGIN_M < 0.01
    # higher up, the goal moves towards where the descent heads
    assert np.linalg.norm(skills.grasp_goal(0.10) - skills.DESCENT_P0) < np.linalg.norm(skills.grasp_goal(h) - skills.DESCENT_P0)


def test_top_to_tips_round_trip():
    p = np.array([0.4, 0.8])
    assert np.allclose(skills.tips_to_top(skills.top_to_tips(p, 0.03), 0.03), p)


def test_goal_point_remembers_brick_length_only_when_whole():
    sk = skills.Skills.__new__(skills.Skills)
    sk.brick_len_v = None
    assert np.allclose(sk._goal_point([0.4, 0.5, 0.5, 0.7], 0.03), [0.5, 0.6])
    assert sk.brick_len_v > 0.2  # 0.2 at 3 cm, longer at grasp height
    sk.brick_len_v = None
    sk._goal_point([0.4, 0.6, 0.5, 1.0], 0.03)
    assert sk.brick_len_v is None


def test_roll_range_is_centred_on_the_rest_joint_angle():
    import arm

    calib = arm.Calibration()  # no calibration: units map linearly to +/-180 deg
    lo, hi = skills.roll_range(calib)
    assert hi - lo <= 2 * skills.ROLL_FROM_REST_DEG + 1
    # the drop's roll is the rest pose's wrist angle: inside the range, near its middle
    assert lo < skills.DROP_ROLL_DEG < hi
    assert abs((lo + hi) / 2 - skills.DROP_ROLL_DEG) < 20


def test_grasp_gap_is_zero_when_the_edge_at_tip_depth_meets_the_jaw():
    # the top face's edge as it appears when the brick's side is right at the jaw's inner edge
    u = skills.tips_to_top([skills.FIXED_JAW_INNER_U, 0.5], skills.GRASP_LIFT_M)[0]
    assert abs(skills.grasp_gap_m([0.3, 0.5, u, 0.8])) < 1e-9
    assert skills.grasp_gap_m([0.3, 0.5, u - 0.02, 0.8]) > 0.002


class _Log:
    def __init__(self):
        self.events = []

    def ev(self, event, **kw):
        self.events.append((event, kw))


def _grasper(gaps):
    sk = skills.Skills.__new__(skills.Skills)
    sk.log, sk.aim_shift_m, sk.target = _Log(), 0.0, "red block"
    sk.centre_calls = []
    sk.centre = lambda start=0: (sk.centre_calls.append(start), skills.Result(True))[1]
    sk.land = lambda: skills.Result(True, data={"gap_m": gaps.pop(0)})
    sk.wrist = lambda tag, settle=0.0: b""
    sk.close = lambda: skills.Result(True)
    sk.lift_and_check = lambda: skills.Result(True)
    return sk


def test_grasp_redoes_once_from_the_last_servo_height_when_the_gap_is_wide():
    sk = _grasper([0.009, 0.004])
    assert sk.grasp(None).ok
    assert sk.centre_calls == [0, len(skills.SERVO_HEIGHTS) - 1]
    assert abs(sk.aim_shift_m - (0.009 - skills.GRASP_MARGIN_M)) < 1e-9  # aim closer to the jaw


def test_grasp_closes_without_redo_when_the_gap_is_in_range_and_after_one_redo_anyway():
    sk = _grasper([0.003])
    assert sk.grasp(None).ok and sk.centre_calls == [0]
    sk = _grasper([-0.004, -0.003])
    assert sk.grasp(None).ok and sk.centre_calls == [0, len(skills.SERVO_HEIGHTS) - 1]
    assert sk.aim_shift_m < 0  # brick under the jaw: aim further from it
