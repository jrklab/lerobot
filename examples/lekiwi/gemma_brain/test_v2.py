
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


def _stray_numbers(path):
    """Numeric literals in a v2 module that aren't plain 0/1/2, an index, a rounding digit
    count, or on a line marked "# convention: <why>" (pitch 90 = straight down, the +/-100
    joint-unit ends, the brick's 180 deg symmetry, the image size, ...)."""
    import ast

    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):  # indices and slices
            allowed.update(id(n) for n in ast.walk(node.slice))
        if isinstance(node, ast.Call):  # round(x, 4), arr.round(4)
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn == "round":
                allowed.update(id(a) for a in node.args[1 if isinstance(node.func, ast.Name) else 0:])
    stray = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if node.value in (0, 1, 2) or id(node) in allowed or "# convention" in lines[node.lineno - 1]:
                continue
            stray.append(f"{path.name}:{node.lineno}: {node.value!r}")
    return stray


def test_v2_code_holds_no_stray_numbers():
    # Layers 1-3: every value comes from params.py (with its source) or geometry.py (CALIB);
    # actions and chains may not even mark conventions.
    for name in ("skills.py", "goals.py", "geometry.py", "actions.py", "chains.py"):
        path = HERE / "v2" / name
        stray = _stray_numbers(path)
        assert not stray, f"numbers outside params.py: {stray} (move them there with their source, or mark a convention)"
    for name in ("actions.py", "chains.py"):
        assert "# convention" not in (HERE / "v2" / name).read_text()


def test_the_stray_number_check_catches_one(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("A = 0.002\nB = round(0.5, 4)\nC = s[9]\nD = 90.0  # convention: straight down\n")
    assert _stray_numbers(p) == ["x.py:1: 0.002", "x.py:2: 0.5"]


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


def test_pick_chain_runs_its_steps_in_order_and_stops_at_the_first_failure(monkeypatch):
    from v2 import chains
    from v2.skills import Result

    calls = []

    def ok(name):
        return lambda *a, **k: (calls.append(name), Result(True))[1]

    for name in ("approach", "hover", "line_up", "align", "land", "lift"):
        monkeypatch.setattr(chains.actions, name, ok(name))
    sk = Skills2.__new__(Skills2)
    sk.servo = lambda obj, goal, part=None: (calls.append(type(goal).__name__), Result(True))[1]
    sk.move = lambda part, **k: (calls.append("down"), Result(True))[1]
    sk.grip = lambda how, **k: (calls.append(how), Result(True))[1]
    r = chains.pick(sk, "red block")
    assert r.ok
    assert calls == ["approach", "hover", "line_up", "align", "JawGoal", "down", "JawGoal", "land", "close", "lift"]
    calls.clear()
    sk.grip = lambda how, **k: Result(False, "closed to 1.6 without holding anything")
    r = chains.pick(sk, "red block", drive=False)
    assert not r.ok and r.data["failed"] == "close" and "close:" in r.reason
    assert calls[0] == "hover"


def _chain_sk(calls, held=True):
    from v2.skills import Result

    sk = Skills2.__new__(Skills2)
    sk.servo = lambda obj, goal, part=None: (calls.append(f"{type(goal).__name__}:{part or goal.part}"), Result(True))[1]
    sk.move = lambda part, **k: (calls.append("move"), Result(True))[1]
    sk.grip = lambda how, **k: (calls.append(how), Result(True))[1]
    sk.held = lambda: held
    return sk


def test_place_in_runs_over_lower_release(monkeypatch):
    from v2 import chains

    monkeypatch.setattr(prm, "demo_place_view", lambda: ([0.15, 0.05, 0.75, 0.80], 0.106))
    monkeypatch.setattr(chains.actions, "approach", lambda sk, o: chains.Result(True))
    calls = []
    monkeypatch.setattr(chains.actions, "over_container", lambda sk, c: (calls.append("over"), chains.Result(True))[1])
    sk = _chain_sk(calls)
    seq = iter([True, True, False])  # held before, ... , empty after the release
    sk.held = lambda: next(seq)
    r = chains.place_in(sk, "red block", "paper cup")
    assert r.ok, r
    assert calls == ["move", "over", "move", "open", "move"]


def test_place_needs_something_held():
    from v2 import chains

    r = chains.place_in(_chain_sk([], held=False), "red block", "paper cup", drive=False)
    assert not r.ok and r.data["failed"] == "holding"


def test_place_next_to_lines_up_twice_and_lands(monkeypatch):
    from v2 import chains

    calls = []
    sk = _chain_sk(calls)
    seq = iter([True, True, False])
    sk.held = lambda: next(seq)
    monkeypatch.setattr(chains.actions, "land", lambda sk: (calls.append("land"), chains.Result(True))[1])
    r = chains.place_next_to(sk, "red block", "green block", drive=False)
    assert r.ok, r
    assert calls == ["move", "BesideGoal:base", "BesideGoal:arm", "move", "BesideGoal:arm", "land", "open", "move"]


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

        def find(self, obj, cam, angle=False):
            u, v = self.centre
            return Detection(box=[u - 0.3, v - 0.35, u + 0.3, v + 0.35])

        def move(self, part, delta=None, roll=None, **kw):
            # the container's rim is nearer the camera than the floor: it moves 1.5x more
            self.centre = self.centre + 1.5 * (self.geom.jacobian(self.h) @ np.asarray(delta[:2]))
            from v2.skills import Result

            return Result(True)

        def ev(self, *a, **k):
            pass

        servo = Skills2.servo

    w = World()
    assert w.servo("paper cup", goal).ok


def test_lift_right_after_landing_is_allowed():
    # Oct 27 teach: the lift from the grasp height was refused by the free-move floor guard.
    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    assert actions.land(sk).ok
    before = sk.tcp()[2]
    r = sk.move("arm", delta=(0.0, 0.0, prm.LIFT_M.value))
    assert r.ok, r
    assert sk.tcp()[2] > before + prm.LIFT_M.value - 0.01


def test_lift_halves_where_the_full_lift_is_out_of_reach():
    from v2.skills import Result

    tried = []

    class S:
        def move(self, part, delta=None, **k):
            tried.append(round(delta[2], 4))
            return Result(delta[2] < 0.1, "" if delta[2] < 0.1 else "can't move there")

        def held(self):
            return True

    r = actions.lift(S())
    assert r.ok and tried == [round(prm.LIFT_M.value, 4), round(prm.LIFT_M.value / 2, 4)]


def test_contact_moves_always_carry_their_own_floor_guard():
    # Oct 27 place-in: a contact-stop move from 10.6 to 7.4 cm passed no z_min, the Pi used its
    # stale default and refused it.
    sim, sk = skills_over(-0.12, -0.12)
    sent = []
    orig = sk.robot.trajectory
    sk.robot.trajectory = lambda wps, durs, z_min=None, contact_stop=False: (sent.append(z_min), orig(wps, durs, z_min, contact_stop))[1]
    assert sk.move("arm", height=0.10).ok
    assert sk.move("arm", height=0.07, until_contact=True).ok
    assert sent[-1] is not None and sent[-1] < -0.12 + 0.07


def test_a_robot_refusal_is_a_failed_step_not_a_crash():
    from robot import RobotError

    sim, sk = skills_over(-0.12, -0.12)

    def refuse(*a, **k):
        raise RobotError("trajectory refused")

    sk.robot.trajectory = refuse
    r = sk.move("arm", height=0.03, until_contact=True)
    assert not r.ok and "refused" in r.reason


def test_held_needs_the_hold_watch_not_just_a_high_reading():
    class R:
        def __init__(self, pos, state):
            self.st = {"joints": {"arm_gripper": pos}, "gripper_hold": {"state": state} if state else None}

        def status(self):
            return self.st

    sk = Skills2.__new__(Skills2)
    for pos, state, want in ((6.8, "holding", True), (22.9, None, False), (1.6, "holding", False), (6.0, "dropped", False)):
        sk.robot = R(pos, state)
        assert sk.held() is want, (pos, state)


def test_round_centre_ignores_where_the_hidden_near_side_ends():
    # two consecutive looks of the still cup (Oct 27), the near side ending at 0.953 and 0.753
    a = goals.round_centre([0.113, 0.025, 0.776, 0.953])
    b = goals.round_centre([0.116, 0.033, 0.784, 0.753])
    assert np.linalg.norm(a - b) < 0.02
    boxes_centre_v = ((0.025 + 0.953) / 2, (0.033 + 0.753) / 2)
    assert abs(boxes_centre_v[0] - boxes_centre_v[1]) > 0.09  # what the box centre did


def test_locate_from_two_heights_recovers_a_cup_placed_in_a_simulated_view():
    # A camera model consistent with geometry.py: a point at (dx, dy) from the TCP's vertical
    # and height r is seen at under_tcp(z) - J(h) (d + h) / z @ (dx, dy), z = d + h - r.
    from v2.geometry import locate_from_two_heights

    g = jaw_geom()
    g.grasp_box, g.grasp_target_height = [0.411, 0.613, 0.545, 0.927], 0.0015
    d, rim, true = g.cam_above_tips, 0.07, np.array([0.012, -0.020])
    taught = np.array([(0.411 + 0.545) / 2, (0.613 + 0.927) / 2])
    z_t = d + 0.0015 - prm.BRICK_HEIGHT_M.value

    def view(h, diameter=0.08):
        z = d + h - rim
        under = g.descent_p0 + (taught - g.descent_p0) * z_t / z
        c = under - (g.jacobian(h) * (d + h) / z) @ true
        w = diameter * np.linalg.norm(g.jacobian(h)[0]) * (d + h) / z
        return c, w, h

    (c1, w1, h1), (c2, w2, h2) = view(0.106), view(0.131)
    delta, height = locate_from_two_heights(g, c1, w1, h1, c2, w2, h2)
    assert abs(height - rim) < 1e-6 and np.allclose(delta, true, atol=1e-6)
