
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


def test_grasp_goal_at_grasp_height_is_beside_the_fixed_jaw_on_the_tip_row():
    u, v = skills.grasp_goal(skills.GRASP_LIFT_M)
    margin_u = skills.GRASP_MARGIN_M * np.linalg.norm(skills.jacobian(skills.GRASP_LIFT_M)[0])
    assert np.isclose(u, skills.FIXED_JAW_INNER_U - margin_u)
    assert np.isclose(v, skills.FIXED_JAW_TIP_V)
    assert 0 < skills.GRASP_MARGIN_M < 0.01


def test_goal_point_remembers_brick_length_only_when_whole():
    sk = skills.Skills.__new__(skills.Skills)
    sk.brick_len_v = None
    assert np.allclose(sk._goal_point([0.4, 0.5, 0.5, 0.7], 0.03), [0.5, 0.6])
    assert sk.brick_len_v > 0.2  # 0.2 at 3 cm, longer at grasp height
    sk.brick_len_v = None
    sk._goal_point([0.4, 0.6, 0.5, 1.0], 0.03)
    assert sk.brick_len_v is None
