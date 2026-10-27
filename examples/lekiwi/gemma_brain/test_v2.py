
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

# ruff: noqa: E402

"""Offline checks for the v2 skills (v2/): geometry, and move/land/grip run by the Pi's
AutoController in the web_server simulation (as test_arm.py)."""

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web_server"))

from arm import Arm, Calibration  # noqa: E402
from test_arm import SimRobot as _SimRobot  # noqa: E402
from test_auto_control import Sim  # noqa: E402
from v2 import actions, goals, params as prm  # noqa: E402
from v2.geometry import Geometry, fit_floor  # noqa: E402
from v2.goals import Detection  # noqa: E402
from v2.skills import Skills2  # noqa: E402

CAL = Calibration()


class SimRobot(_SimRobot):
    def trajectory(self, waypoints, durations, z_min=None, contact_stop=False):
        cmd = {"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min}
        if contact_stop:
            cmd["contact_stop"] = True
        return self.run(cmd)

    def gripper(self, pos, speed=None):
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed})


def geom(floor_z):
    return Geometry(floor_coef=np.array([floor_z, 0.0, 0.0]), floor_points=3, jac=np.array([[4.2, 0.5], [0.7, -5.3]]),
                    jac_height=0.094, cam_above_tips=0.1075, descent_p0=np.array([0.5, 0.15]),
                    fixed_jaw_inner_u=0.564, fixed_jaw_tip_v=0.754, sources={})


def skills_over(real_floor_tcp_z, predicted):
    # The sim's floor stops the TCP; ours is TCP z at contact, so the same number.
    sim = Sim(floor_z=real_floor_tcp_z)
    sk = Skills2(SimRobot(sim), Arm(SimRobot(sim), CAL, body_check=False), geom(predicted))
    assert sk.move(xy=(0.22, 0.0), height=0.05).ok
    return sim, sk


def test_floor_fit_is_a_plane_and_falls_back_flat():
    pts = [{"x": x, "y": y, "z": -0.11 + 0.01 * x - 0.02 * y} for x, y in ((0.18, 0), (0.3, 0), (0.24, 0.05), (0.2, -0.04))]
    assert np.allclose(fit_floor(pts, -0.2), [-0.11, 0.01, -0.02])
    assert np.allclose(fit_floor(pts[:2], -0.2), [np.mean([p["z"] for p in pts[:2]]), 0, 0])
    assert np.allclose(fit_floor([], -0.2), [-0.2, 0, 0])


def test_jacobian_grows_closer_to_the_floor_and_turns_with_roll():
    g = geom(-0.11)
    assert np.linalg.norm(g.jacobian(0.03)) > np.linalg.norm(g.jacobian(0.094))
    assert np.allclose(g.jacobian(0.094), g.jac)
    j90 = g.jacobian(0.094, 90.0) * [[640], [480]]
    assert np.allclose(np.linalg.svd(j90, compute_uv=False), np.linalg.svd(g.jac * [[640], [480]], compute_uv=False))


def test_every_param_names_its_source():
    assert prm.ALL and all(p.source in ("CALIB", "DEMO", "SPEC", "RULE") and p.why for p in prm.ALL.values())


def test_land_on_the_predicted_floor_does_not_press():
    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    r = actions.land(sk)
    assert r.ok and not r.data["contact"]
    assert abs(sk.height() - prm.GRASP_LIFT_M.value) < 0.002


def test_land_on_a_slightly_higher_floor_stops_on_it():
    # 5 mm higher than predicted: the goal is ~1 mm below the real floor, under the Pi's
    # contact threshold, so the arm just stops on it (pressing at most that 1 mm).
    sim, sk = skills_over(real_floor_tcp_z=-0.115, predicted=-0.12)
    assert actions.land(sk).ok
    assert sk.tcp()[2] >= -0.115 - 0.0005


def test_land_treats_a_contact_over_half_a_brick_up_as_not_the_floor():
    # 7.5 mm above the predicted floor: more than half the brick's height, so it might be the
    # brick's edge; land refuses rather than close there (the fit is good to a few mm).
    sim, sk = skills_over(real_floor_tcp_z=-0.1125, predicted=-0.12)
    r = actions.land(sk)
    assert not r.ok and "above the floor" in r.reason, r


def test_land_refuses_to_close_on_something_tall():
    sim, sk = skills_over(real_floor_tcp_z=-0.12 + 0.03, predicted=-0.12)  # 3 cm up: not the floor
    r = actions.land(sk)
    assert not r.ok and "above the floor" in r.reason, r


def test_move_out_of_vertical_reach_is_refused_without_moving():
    sim, sk = skills_over(-0.12, -0.12)
    before = dict(sim.pos)
    r = sk.move(xy=(0.6, 0.0), height=0.10)
    assert not r.ok and "can't move" in r.reason
    assert sim.pos == before


def test_base_moves_are_one_kind_at_a_time():
    sim, sk = skills_over(-0.12, -0.12)
    assert not sk.move("base", delta=(0.05, 0.0), roll=10.0).ok


def test_actions_and_chains_hold_no_numbers():
    # Layer 3 composes primitives; every value must come from params.py or geometry.py.
    import ast

    for name in ("actions.py", "chains.py"):
        path = HERE / "v2" / name
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                assert node.value in (0, 1, 2, 1000), f"{name}:{node.lineno}: number {node.value} (put it in params.py with its source)"


# --- find / servo / goals ---


class FakeModel:
    def __init__(self, boxes):
        self.boxes, self.calls = list(boxes), 0

    def box(self, jpeg, what):
        from gemma import Answer

        self.calls += 1
        return Answer(text="", value=self.boxes.pop(0) if self.boxes else None)


def red_frame(box):
    img = np.full((480, 640, 3), 170, np.uint8)
    x0, y0, x1, y1 = (int(v * s) for v, s in zip(box, (640, 480, 640, 480), strict=True))
    cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 220), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


class FrameRobot:
    def __init__(self, jpeg):
        self.jpeg = jpeg

    def frame(self, cam, after=0.0):
        return self.jpeg, after


def finder(jpeg, main, fast):
    sk = Skills2.__new__(Skills2)
    sk.robot, sk.log, sk.main, sk.fast, sk.seen = FrameRobot(jpeg), None, main, fast, {}
    return sk


def test_find_uses_the_fast_model_only_when_tracking_and_checks_its_colour(monkeypatch):
    box = [0.40, 0.40, 0.50, 0.55]
    jpeg = red_frame(box)
    main, fast = FakeModel([box, box]), FakeModel([[0.05, 0.05, 0.15, 0.15], box])
    sk = finder(jpeg, main, fast)
    import v2.skills as skills_v2

    monkeypatch.setattr(skills_v2.time, "sleep", lambda s: None)
    assert sk.find("red block") is not None and (main.calls, fast.calls) == (1, 0)  # first sight: main
    d = sk.find("red block")  # tracking: fast boxed grey floor -> rejected by colour -> main
    assert d is not None and d.by == "main" and (main.calls, fast.calls) == (2, 1)
    d = sk.find("red block")  # tracking: fast boxed the brick
    assert d is not None and d.by == "fast"


class SimServo:
    """A world for servo(): the object's image point responds to arm moves through a TRUE
    Jacobian, which may differ from the calibrated one the goal uses."""

    def __init__(self, geom, start_err, true_scale=1.0, height=0.03):
        self.geom, self.roll, self.h, self.true_scale = geom, 0.0, height, true_scale
        self.goal = goals.JawGoal()
        self.point = self.goal.target(self, height) - np.asarray(start_err, float)
        self.moves = 0

    def height(self):
        return self.h

    def find(self, obj, cam, angle=False):
        u, v = self.point
        return Detection(box=[u - 0.08, v - 0.1, u, v + 0.1])

    def move(self, part, delta=None, roll=None, **kw):
        j = self.goal._jac(self, self.h) * self.true_scale
        self.point = self.point + j @ np.asarray(delta[:2], float)  # the object, seen from the moving camera
        self.moves += 1
        from v2.skills import Result

        return Result(True)

    def ev(self, *a, **k):
        pass

    servo = Skills2.servo


def jaw_geom():
    g = geom(-0.11)
    g.grasp_target = np.array([0.55, 0.80])
    return g


def test_servo_converges_and_stays_in_view():
    sim = SimServo(jaw_geom(), start_err=[0.15, -0.12])
    r = sim.servo("red block", sim.goal)
    assert r.ok, r
    assert sim.goal.done(sim.goal.error(sim.find("", ""), sim), sim)


def test_servo_converges_with_a_30_percent_wrong_jacobian():
    sim = SimServo(jaw_geom(), start_err=[0.10, 0.08], true_scale=1.3)
    assert sim.servo("red block", sim.goal).ok


def test_jaw_goal_needs_the_midpoint_teach():
    import pytest
    from v2.geometry import MissingCalibrationError

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
