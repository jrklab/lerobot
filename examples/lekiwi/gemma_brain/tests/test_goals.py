
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

"""The goals (brain/goals.py) and servo() converging on them, with the true Jacobian off."""

from pathlib import Path

import numpy as np
from brain import goals, params as prm
from brain.goals import Detection
from brain.primitives import Skills
from helpers import (
    SimServo,
    geom,
    jaw_geom,
)

HERE = Path(__file__).resolve().parent


def test_servo_converges_and_stays_in_view():
    sim = SimServo(jaw_geom(), start_err=[0.15, -0.12])
    r = sim.servo("red block", sim.goal)
    assert r.ok, r
    assert sim.goal.done(sim.goal.error(sim.find("", ""), sim), sim)

def test_servo_converges_with_a_30_percent_wrong_jacobian():
    sim = SimServo(jaw_geom(), start_err=[0.10, 0.08], true_scale=1.3)
    assert sim.servo("red block", sim.goal).ok

def test_servo_converges_at_the_gain_with_the_jacobian_30_percent_off_either_way():
    for scale in (0.7, 1.3):
        sim = SimServo(jaw_geom(), start_err=[0.12, -0.10], true_scale=scale)
        r = sim.servo("red block", sim.goal)
        assert r.ok and r.data["looks"] <= 6, (scale, r)

def test_jaw_goal_needs_the_midpoint_teach():
    import pytest
    from brain.geometry import MissingCalibrationError

    sim = SimServo.__new__(SimServo)
    sim.geom, sim.roll, sim.h = geom(-0.11), 0.0, 0.03
    with pytest.raises(MissingCalibrationError):
        goals.JawGoal().target(sim, 0.03)

def test_angle_goal_rolls_within_the_range_using_the_brick_symmetry():
    class S:
        roll, roll_range = 0.0, (-144.0, 14.0)

    g = goals.AngleGoal()
    # needs +30 or -150: neither fits -144..14; -150 is nearer its end
    assert g.step(np.array([30.0]), S())["roll"] == -144.0
    assert g.step(np.array([-40.0]), S())["roll"] == -40.0
    assert g.done(np.array([10.0]), S()) and not g.done(np.array([20.0]), S())

def test_front_approach_turns_before_it_drives():
    class S:
        geom = geom(-0.11)

    S.geom.front = {"u_per_deg": 0.0184, "u_target": 0.451, "near_bottom": 0.699, "range_k": 0.0583,
                    "range_horizon_v": 0.335, "range_offset_m": 0.1}
    g = goals.FrontApproachGoal()
    turn = g.step(g.error(Detection(box=[0.6, 0.4, 0.7, 0.5]), S()), S())
    # right of the target column: turn right (negative), since a left turn moves it right
    assert "roll" in turn and turn["roll"] < 0
    drive = g.step(g.error(Detection(box=[0.40, 0.4, 0.50, 0.5]), S()), S())
    assert "delta" in drive and drive["delta"][0] > 0

def test_front_approach_stops_turning_once_a_turn_carries_the_object_across_the_column():
    class S:
        geom = geom(-0.11)

    S.geom.front = {"u_per_deg": 0.0184, "u_target": 0.451, "near_bottom": 0.699, "range_k": 0.0583,
                    "range_horizon_v": 0.335, "range_offset_m": 0.1}
    g = goals.FrontApproachGoal()
    # Oct 28 trial 4: the brick ~3 cm past the reach point, each 3 deg turn moved it 0.11 across.
    first = np.array([0.0875, -0.115])
    assert not g.done(first, S()) and "roll" in g.step(first, S())
    crossed = np.array([-0.072, -0.105])
    assert g.done(crossed, S())  # close enough in range; the line-up takes the column's rest
    # Not yet in range: drive instead of turning back.
    g2 = goals.FrontApproachGoal()
    g2.step(np.array([0.0875, 0.2]), S())
    assert "delta" in g2.step(np.array([-0.072, 0.2]), S())

def test_front_approach_is_done_within_the_line_ups_reach_and_aims_just_short():
    class S:
        geom = geom(-0.11)

    S.geom.front = {"u_per_deg": 0.0184, "u_target": 0.451, "near_bottom": 0.699, "range_k": 0.0583,
                    "range_horizon_v": 0.335, "range_offset_m": 0.1}
    g = goals.FrontApproachGoal()
    far = g.error(Detection(box=[0.43, 0.40, 0.47, 0.45]), S())
    rem = g.remaining(far, S())
    assert not g.done(far, S()) and abs(g.step(far, S())["delta"][0] - (rem - prm.REACH_TOL_M.value / 2)) < 1e-9
    near = g.error(Detection(box=[0.43, 0.60, 0.47, 0.675]), S())  # ~1 cm to go
    assert g.remaining(near, S()) <= prm.REACH_TOL_M.value and g.done(near, S())
    not_yet = g.error(Detection(box=[0.43, 0.56, 0.47, 0.653]), S())  # Oct 27 run: 2.3 cm to go
    assert not g.done(not_yet, S())

def test_beside_goal_aims_left_of_the_taught_spot():
    g = jaw_geom()
    g.grasp_box, g.grasp_target_height = [0.47, 0.70, 0.55, 0.90], prm.GRASP_LIFT_M.value

    class S:
        geom, roll = g, 0.0

    t = goals.BesideGoal().target(S(), prm.GRASP_LIFT_M.value)
    assert np.isclose(t[0], 0.47 - prm.NEXT_GAP_FRACTION.value * 0.08) and np.isclose(t[1], 0.80)

def test_above_goal_converges_on_a_container(monkeypatch):
    monkeypatch.setattr(prm, "demo_place_view", lambda: ([0.15, 0.05, 0.75, 0.80], 0.106))
    goal = goals.AboveGoal()

    class World:
        geom, roll, h = jaw_geom(), 0.0, 0.106
        centre = np.array([0.62, 0.55])

        def height(self):
            return self.h

        def tcp(self):  # servo() logs the TCP at each look
            return np.zeros(3)

        cmd_tcp = tcp

        def move_measured(self, delta):
            return self.move("arm", delta=delta)

        def find(self, obj, cam, angle=False):
            u, v = self.centre
            return Detection(box=[u - 0.3, v - 0.35, u + 0.3, v + 0.35])

        def move(self, part, delta=None, roll=None, **kw):
            # the container's rim is nearer the camera than the floor: it moves 1.5x more
            self.centre = self.centre + 1.5 * (self.geom.jacobian(self.h) @ np.asarray(delta[:2]))
            from brain.primitives import Result

            return Result(True)

        def ev(self, *a, **k):
            pass

        servo = Skills.servo

    w = World()
    assert w.servo("paper cup", goal).ok

def test_round_centre_ignores_where_the_hidden_near_side_ends():
    # two consecutive looks of the still cup (Oct 27), the near side ending at 0.953 and 0.753
    a = goals.round_centre([0.113, 0.025, 0.776, 0.953])
    b = goals.round_centre([0.116, 0.033, 0.784, 0.753])
    assert np.linalg.norm(a - b) < 0.02
    boxes_centre_v = ((0.025 + 0.953) / 2, (0.033 + 0.753) / 2)
    assert abs(boxes_centre_v[0] - boxes_centre_v[1]) > 0.09  # what the box centre did

def test_placed_beside_measures_the_gap_and_level():
    from brain.goals import placed_beside

    # Beside: the reference's right edge left of the object by about half its width.
    held = [0.45, 0.5, 0.55, 0.7]
    assert placed_beside(held, [0.30, 0.52, 0.40, 0.72])  # gap 0.05 = intended
    assert not placed_beside(held, [0.40, 0.52, 0.50, 0.72])  # overlapping
    assert not placed_beside(held, [0.10, 0.52, 0.20, 0.72])  # far off
    assert not placed_beside(held, [0.30, 0.80, 0.40, 1.0])  # not level
