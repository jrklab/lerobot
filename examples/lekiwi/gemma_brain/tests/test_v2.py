
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

"""Offline checks for the v2 skills (brain/): geometry, and move/land/grip run by the Pi's
AutoController in the web_server simulation (as test_arm.py)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "web_server"))

from brain import actions, goals, params as prm  # noqa: E402
from brain.geometry import Geometry, fit_floor  # noqa: E402
from brain.goals import Detection  # noqa: E402
from brain.primitives import Skills  # noqa: E402
from common.arm import Arm, Calibration  # noqa: E402
from test_arm import SimRobot as _SimRobot  # noqa: E402
from test_auto_control import Sim  # noqa: E402

CAL = Calibration()


class SimRobot(_SimRobot):
    def trajectory(self, waypoints, durations, z_min=None, contact_stop=False):
        cmd = {"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min}
        if contact_stop:
            cmd["contact_stop"] = True
        return self.run(cmd)

    def gripper(self, pos, speed=None):
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed})

    def frame(self, cam, after=0.0):
        return b"", after


def geom(floor_z):
    return Geometry(floor_coef=np.array([floor_z, 0.0, 0.0]), floor_points=3, jac=np.array([[4.2, 0.5], [0.7, -5.3]]),
                    jac_height=0.094, cam_above_tips=0.1075, descent_p0=np.array([0.5, 0.15]),
                    fixed_jaw_inner_u=0.564, fixed_jaw_tip_v=0.754, sources={})


def skills_over(real_floor_tcp_z, predicted):
    # The sim's floor stops the TCP; ours is TCP z at contact, so the same number.
    sim = Sim(floor_z=real_floor_tcp_z)
    sk = Skills(SimRobot(sim), Arm(SimRobot(sim), CAL, body_check=False), geom(predicted))
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
    for name in ("primitives.py", "goals.py", "geometry.py", "actions.py", "chains.py"):
        path = HERE.parent / "brain" / name
        stray = _stray_numbers(path)
        assert not stray, f"numbers outside params.py: {stray} (move them there with their source, or mark a convention)"
    for name in ("actions.py", "chains.py"):
        assert "# convention" not in (HERE.parent / "brain" / name).read_text()


def test_the_stray_number_check_catches_one(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("A = 0.002\nB = round(0.5, 4)\nC = s[9]\nD = 90.0  # convention: straight down\n")
    assert _stray_numbers(p) == ["x.py:1: 0.002", "x.py:2: 0.5"]


# --- find / servo / goals ---


class FakeModel:
    def __init__(self, boxes):
        self.boxes, self.calls = list(boxes), 0

    def box(self, jpeg, what):
        from common.gemma import Answer

        self.calls += 1
        self.asked = what
        return Answer(text="", value=self.boxes.pop(0) if self.boxes else None)


def red_frame(box):
    img = np.full((480, 640, 3), 170, np.uint8)
    x0, y0, x1, y1 = (int(v * s) for v, s in zip(box, (640, 480, 640, 480), strict=True))
    cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 220), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_placed_beside_measures_the_gap_and_level():
    from brain.goals import placed_beside

    # Beside: the reference's right edge left of the object by about half its width.
    held = [0.45, 0.5, 0.55, 0.7]
    assert placed_beside(held, [0.30, 0.52, 0.40, 0.72])  # gap 0.05 = intended
    assert not placed_beside(held, [0.40, 0.52, 0.50, 0.72])  # overlapping
    assert not placed_beside(held, [0.10, 0.52, 0.20, 0.72])  # far off
    assert not placed_beside(held, [0.30, 0.80, 0.40, 1.0])  # not level


def test_place_in_check_is_the_front_camera_finding_the_object_on_the_floor():
    # Oct 28 miss tests: the brick lay on the floor by the slid cup. The check is one front look:
    # on the floor = missed; not there = in (a brick in the cup is behind its wall).
    asked = []

    class S:
        def __init__(self, on_floor):
            self.on_floor = on_floor

        def find(self, obj, cam, fast_ok=True, **k):
            asked.append((obj, cam, fast_ok))
            return Detection(box=[0.4, 0.62, 0.5, 0.7]) if self.on_floor else None

        def ev(self, *a, **k):
            pass

    r = actions.check_in(S(True), "red block", "paper cup")
    assert not r.ok and "on the floor" in r.reason
    assert actions.check_in(S(False), "red block", "paper cup").ok
    assert asked == [("red block", "front", False)] * 2  # the brick, front camera, the main model

class FrameRobot:
    def __init__(self, jpeg):
        self.jpeg = jpeg

    def frame(self, cam, after=0.0):
        return self.jpeg, after


def finder(jpeg, main, fast):
    sk = Skills.__new__(Skills)
    sk.robot, sk.log, sk.main, sk.fast, sk.seen = FrameRobot(jpeg), None, main, fast, {}
    return sk


def test_find_uses_the_fast_model_only_when_tracking_and_checks_its_colour(monkeypatch):
    box = [0.40, 0.40, 0.50, 0.55]
    jpeg = red_frame(box)
    main, fast = FakeModel([box, box]), FakeModel([[0.05, 0.05, 0.15, 0.15], box])
    sk = finder(jpeg, main, fast)
    import brain.primitives as primitives

    monkeypatch.setattr(primitives.time, "sleep", lambda s: None)
    assert sk.find("red block") is not None and (main.calls, fast.calls) == (1, 0)  # first sight: main
    d = sk.find("red block")  # tracking: fast boxed grey floor -> rejected by colour -> main
    assert d is not None and d.by == "main" and (main.calls, fast.calls) == (2, 1)
    d = sk.find("red block")  # tracking: fast boxed the brick
    assert d is not None and d.by == "fast"


def test_a_front_box_ending_above_the_horizon_is_not_on_the_floor(monkeypatch):
    # Oct 28: carried at the hover, the jaw tips show at the front frame's top.
    import brain.primitives as primitives

    monkeypatch.setattr(primitives.time, "sleep", lambda s: None)
    high, low = [0.50, 0.00, 0.60, 0.20], [0.40, 0.40, 0.50, 0.55]
    for box, found in ((high, False), (low, True)):
        sk = finder(red_frame(box), FakeModel([box]), None)
        sk.geom = geom(-0.11)
        sk.geom.front = {"range_horizon_v": 0.335}
        assert (sk.find("red block", "front") is not None) == found


def test_a_measured_move_takes_up_the_arms_play():
    # Oct 28 trials: a 2 mm reversal moved the encoders 0 mm, a forward step went 3.5 mm too far.
    from brain.primitives import Result

    class Play:
        """An arm whose position follows its command only once the command is more than `slack`
        away, from the side it was pushed (backlash)."""

        def __init__(self, slack):
            self.cmd, self.pos, self.slack, self.sent = np.zeros(2), np.zeros(2), slack, []

        def move(self, part, delta=None, **k):
            self.sent.append(delta[:2])
            self.cmd = self.cmd + np.asarray(delta[:2])
            gap = self.cmd - self.pos
            self.pos = np.where(np.abs(gap) > self.slack, self.cmd - np.sign(gap) * self.slack, self.pos)
            return Result(True)

    class Cal:
        resolution = 4096

    def skills(arm):
        sk = Skills.__new__(Skills)
        sk.log, sk.arm, sk.move, sk._settle = None, type("A", (), {"calib": Cal()})(), arm.move, lambda: None
        sk.tcp = lambda: np.array([*arm.pos, -0.08])
        return sk

    play = Play(0.003)
    sk = skills(play)
    play.move("arm", delta=(0.006, 0.0, 0.0))  # last pushed forward: the play is taken up behind
    for d in ((-0.004, 0.0), (0.004, 0.001)):
        before, sent = play.pos.copy(), len(play.sent)
        assert sk.move_measured(d).ok
        assert np.linalg.norm(play.pos - before - d) <= prm.SERVO_TOL_M.value
        assert len(play.sent) > sent + 1  # the shortfall was commanded again

    class Stuck(Play):
        """Moves 1 mm on the first command, then ignores the small corrections (Oct 28: 8
        corrections left 1.2 and 2.2 mm)."""

        def move(self, part, delta=None, **k):
            self.sent.append(delta[:2])
            if len(self.sent) == 1:
                self.pos = self.pos + np.sign(delta[:2]) * 0.001
            return Result(True)

    stuck = Stuck(0.0)
    assert skills(stuck).move_measured((0.005, 0.0)).ok
    assert len(stuck.sent) == 2  # one correction, which didn't move it: no more


def test_the_look_settle_ends_once_the_arm_is_still_and_waits_after_the_base():
    import time as _time

    class Cal:
        resolution = 4096

    sk = Skills.__new__(Skills)
    sk.log, sk.arm = None, type("A", (), {"calib": Cal()})()
    path = iter([0.25, 0.254] + [0.258] * 50)  # still from the 3rd read
    reads = []
    sk.tcp = lambda: (reads.append(_time.monotonic()), np.array([next(path), 0.0, -0.08]))[1]
    sk._last_moved = "arm"
    t = _time.monotonic()
    sk._settle()
    assert _time.monotonic() - t < prm.LOOK_SETTLE_S.value - 0.05 and len(reads) == 4
    # a control tick apart: closer reads returned the same sample (Oct 28)
    assert min(np.diff(reads)) >= prm.CONTROL_PERIOD_S.value * 0.95
    sk._last_moved = "base"
    t = _time.monotonic()
    sk._settle()
    assert _time.monotonic() - t >= prm.LOOK_SETTLE_S.value - 0.01


def test_carry_holds_at_the_hover_without_folding():
    from brain.primitives import Result

    calls = []

    class S:
        arm = type("A", (), {"reset": lambda self: calls.append("fold")})()

        def move(self, part, **k):
            calls.append((part, k.get("height")))
            return Result(True)

    assert actions.carry(S()).ok and calls == [("arm", prm.HOVER_HEIGHT_M.value)]


def test_container_looks_ask_for_its_rim(monkeypatch):
    # Oct 28 replay: "the rim of the paper cup" boxed it in 55 of 55 wrist frames, "the paper cup
    # on the floor" missed all 6 with the cup's middle under the held brick.
    import brain.primitives as primitives

    monkeypatch.setattr(primitives.time, "sleep", lambda s: None)
    box = [0.2, 0.3, 0.8, 0.95]
    main = FakeModel([box, box])
    sk = finder(red_frame(box), main, None)
    sk.find("paper cup", "wrist", fast_ok=False, rim=True)
    assert main.asked == "the rim of the paper cup"
    sk.find("red block", "wrist")
    assert main.asked == "the red block on the floor"


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

    def tcp(self):  # servo() logs the TCP at each look
        return np.zeros(3)

    cmd_tcp = tcp

    def move_measured(self, delta):
        return self.move("arm", delta=delta)

    def find(self, obj, cam, angle=False):
        u, v = self.point
        return Detection(box=[u - 0.08, v - 0.1, u, v + 0.1])

    def move(self, part, delta=None, roll=None, **kw):
        j = self.goal._jac(self, self.h) * self.true_scale
        self.point = self.point + j @ np.asarray(delta[:2], float)  # the object, seen from the moving camera
        self.moves += 1
        from brain.primitives import Result

        return Result(True)

    def ev(self, *a, **k):
        pass

    servo = Skills.servo


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
    from brain.geometry import MissingCalibrationError

    sim = SimServo.__new__(SimServo)
    sim.geom, sim.roll, sim.h = geom(-0.11), 0.0, 0.03
    with pytest.raises(MissingCalibrationError):
        goals.JawGoal().target(sim, 0.03)


def test_the_roll_ranges_ends_are_moves_the_planner_accepts():
    # Oct 28 full task: a brick at 15 deg, the range ending at ~14: the roll clamped to the end
    # was refused six times (the joint at 100.4, the planner's limit 99).
    # The robot's own calibration (the sim's uncalibrated one gives a different range):
    from common.arm import Calibration

    cal = Calibration(*_real_cal())
    sim = Sim(floor_z=-0.12)
    sk = Skills(SimRobot(sim), Arm(SimRobot(sim), cal, body_check=False), geom(-0.12))
    assert actions.hover(sk).ok
    for end in sk.roll_range:
        r = sk.move("arm", delta=(0.0, 0.0, 0.0), roll=end)
        assert r.ok, (end, r.reason)


def _real_cal():
    c = json.loads((Path(__file__).parent / "fixtures" / "arm_calibration.json").read_text())  # robot.calibration(), Oct 28
    return c["joints"], c["motor_resolution"]


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


def test_pick_chain_runs_its_steps_in_order_and_stops_at_the_first_failure(monkeypatch):
    from brain import chains
    from brain.primitives import Result

    calls = []

    def ok(name):
        return lambda *a, **k: (calls.append(name), Result(True))[1]

    for name in ("approach", "hover", "line_up", "align", "land", "lift"):
        monkeypatch.setattr(chains.actions, name, ok(name))
    sk = Skills.__new__(Skills)
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
    calls.clear()
    sk.grip = lambda how, **k: (calls.append(how), Result(True))[1]
    assert chains.pick(sk, "red block", start="hover").ok and calls[0] == "hover"  # a retry: no approach
    # A refused move's flag reaches the chain's result, for the task loop's retry choice.
    sk.servo = lambda obj, goal, part=None: Result(False, "servo move refused: can't move there", {"refused": True})
    r = chains.pick(sk, "red block")
    assert not r.ok and r.data["refused"] and r.data["failed"] == "centre_high"


def test_only_the_pis_check_refusals_count_as_refused():
    from brain.primitives import pi_refused
    from common.robot import RobotError

    assert pi_refused(RobotError("POST /auto/cmd: HTTP 409: trajectory: the gripper reaches z -0.091 m at waypoint 6, below z_min -0.085 m"))
    assert not pi_refused(RobotError("POST /auto/cmd: HTTP 409: busy: command 12 (trajectory) is still running"))
    assert not pi_refused(RobotError("POST /auto/cmd: HTTP 409: not in auto mode (mode is gamepad)"))
    assert not pi_refused(RobotError("trajectory: stopped: heartbeat lost"))


def _chain_sk(calls, held=True):
    from brain.primitives import Result

    sk = Skills.__new__(Skills)
    sk.servo = lambda obj, goal, part=None: (calls.append(f"{type(goal).__name__}:{part or goal.part}"), Result(True))[1]
    sk.move = lambda part, **k: (calls.append("move"), Result(True))[1]
    sk.grip = lambda how, **k: (calls.append(how), Result(True))[1]
    sk.held = lambda: held
    sk.log, sk.find, sk.seen = None, lambda *a, **k: None, {}  # the place check sees nothing: passes, unchecked
    return sk


def test_place_in_runs_over_lower_release(monkeypatch):
    from brain import chains

    calls = []
    monkeypatch.setattr(prm, "demo_place_view", lambda: ([0.15, 0.05, 0.75, 0.80], 0.106))
    monkeypatch.setattr(chains.actions, "approach", lambda sk, o, tol=None: (calls.append("approach"), chains.Result(True))[1])
    monkeypatch.setattr(chains.actions, "carry", lambda sk: (calls.append("carry"), chains.Result(True))[1])
    monkeypatch.setattr(chains.actions, "over_container", lambda sk, c: (calls.append("over"), chains.Result(True))[1])
    sk = _chain_sk(calls)
    seq = iter([True, True, False])  # held before, ... , empty after the release
    sk.held = lambda: next(seq)
    r = chains.place_in(sk, "red block", "paper cup")
    assert r.ok, r
    assert calls == ["carry", "approach", "move", "over", "move", "open", "move"]


def test_place_needs_something_held():
    from brain import chains

    r = chains.place_in(_chain_sk([], held=False), "red block", "paper cup", drive=False)
    assert not r.ok and r.data["failed"] == "holding"


def test_place_next_to_lines_up_twice_and_lands(monkeypatch):
    from brain import chains

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


def test_lift_right_after_landing_is_allowed():
    # Oct 27 teach: the lift from the grasp height was refused by the free-move floor guard.
    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    assert actions.land(sk).ok
    before = sk.tcp()[2]
    r = sk.move("arm", delta=(0.0, 0.0, prm.LIFT_M.value))
    assert r.ok, r
    assert sk.tcp()[2] > before + prm.LIFT_M.value - 0.01


def test_a_move_crossing_the_clearance_goes_at_the_near_speed_only_below_it():
    # Oct 28 trials: a 13 cm lift from the floor took 13 s at the near speed all the way.
    from common.arm import FLOOR_CLEARANCE_M, JAW_TIP_OFFSET_M

    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    assert actions.land(sk).ok
    # The fitted floor tilts: 5.6 mm lower at the reach point than at x 0, where the free-move
    # guard is taken (Oct 28 fit). Tilted the same way about where the arm is:
    x = sk.cmd_tcp()[0]
    sk.geom.floor_coef = np.array([-0.12 + 0.0224 * x, -0.0224, 0.0])
    sk.arm.floor_z = float(sk.geom.floor_coef[0]) - JAW_TIP_OFFSET_M  # as Skills.__init__ sets them
    sk.arm.z_min = sk.arm.floor_z + FLOOR_CLEARANCE_M + JAW_TIP_OFFSET_M
    sent = []
    orig = sk.arm.robot.trajectory

    def spy(wps, durs, z_min=None, contact_stop=False):
        # Oct 28: the free part, ending on the line, went with the free-move guard and the Pi
        # refused it (the jaw tips below it): every part's guard is under the tips at the line.
        x, y = sk.cmd_tcp()[:2]
        assert z_min <= sk.geom.floor(x, y) + FLOOR_CLEARANCE_M, z_min
        sent.append((sk.height(), sum(durs)))
        return orig(wps, durs, z_min, contact_stop)

    sk.arm.robot.trajectory = spy
    assert sk.move("arm", delta=(0.0, 0.0, prm.LIFT_M.value)).ok
    clear = FLOOR_CLEARANCE_M + JAW_TIP_OFFSET_M
    assert len(sent) == 2 and sent[0][0] < clear  # the near part first, from the floor
    near_s = (clear - prm.GRASP_LIFT_M.value) / prm.NEAR_SPEED_MPS.value
    free_s = (prm.GRASP_LIFT_M.value + prm.LIFT_M.value - clear) / prm.FREE_SPEED_MPS.value
    assert abs(sent[0][1] - near_s) < 0.5 and abs(sent[1][1] - free_s) < 0.5
    assert abs(sk.height() - (prm.GRASP_LIFT_M.value + prm.LIFT_M.value)) < 0.005
    sent.clear()
    h0 = sk.height()
    assert sk.move("arm", height=prm.FINAL_LOOK_HEIGHT_M.value).ok  # down: free first, then near
    assert abs(sent[0][1] - (h0 - clear) / prm.FREE_SPEED_MPS.value) < 0.5
    assert abs(sk.height() - prm.FINAL_LOOK_HEIGHT_M.value) < 0.002


def test_lift_halves_where_the_full_lift_is_out_of_reach():
    from brain.primitives import Result

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
    from common.robot import RobotError

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

    sk = Skills.__new__(Skills)
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


def test_locate_from_one_view_recovers_a_cup_placed_in_a_simulated_view():
    # A camera model consistent with geometry.py: a point at (dx, dy) from the TCP's vertical
    # and height r is seen at under_tcp(z) - J(h) (d + h) / z @ (dx, dy), z = d + h - r.
    from brain.geometry import locate_from_one_view

    g = jaw_geom()
    g.grasp_box, g.grasp_target_height = [0.411, 0.613, 0.545, 0.927], 0.0015
    d, rim, true, h = g.cam_above_tips, 0.044, np.array([0.012, -0.020]), 0.106
    taught = np.array([(0.411 + 0.545) / 2, (0.613 + 0.927) / 2])
    z_t, z = d + 0.0015 - prm.BRICK_HEIGHT_M.value, d + h - rim
    c = g.descent_p0 + (taught - g.descent_p0) * z_t / z - (g.jacobian(h) * (d + h) / z) @ true
    assert np.allclose(locate_from_one_view(g, c, h, rim), true, atol=1e-9)


def test_front_object_height_of_the_cup_in_two_logged_views():
    from brain.geometry import front_object_height

    g = geom(-0.11)
    g.front = {"u_per_deg": 0.0184, "range_k": 0.0583, "range_horizon_v": 0.335}
    near, far = front_object_height(g, [0.25, 0.321, 0.7, 0.803]), front_object_height(g, [0.5, 0.333, 0.88, 0.647])
    assert abs(near - far) < 0.003 and 0.04 < near < 0.05  # Oct 28: 4.3 and 4.2 cm

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


def test_a_failed_frame_request_resyncs_the_clock_and_retries(monkeypatch):
    import brain.primitives as primitives
    from common.robot import RobotError

    monkeypatch.setattr(primitives.time, "sleep", lambda s: None)

    class R:
        _clock_offset, calls = 36.0, 0

        def frame(self, cam, after=0.0):
            self.calls += 1
            if self._clock_offset is not None:
                raise RobotError("GET /auto/frame/front: HTTP 504")
            return b"jpeg", after

    sk = finder(b"", None, None)
    sk.robot = R()
    assert sk.frame("front") == b"jpeg" and sk.robot.calls == 2


def test_base_moves_ask_for_the_pis_stopping_tolerance_more():
    sent = []

    class R:
        def base(self, dx=0.0, dy=0.0, dtheta_deg=0.0):
            sent.append((round(dx, 4), round(dy, 4), round(dtheta_deg, 2)))
            return {"result": {"moved": [dx, dy, dtheta_deg]}}

    sk = Skills.__new__(Skills)
    sk.robot, sk.log = R(), None
    sk.move("base", delta=(0.006, 0.008))  # 1 cm asked: the Pi would end it at once
    sk.move("base", roll=-5.0)
    assert sent[0] == (0.012, 0.016, 0.0) and sent[1] == (0.0, 0.0, -7.0)


def test_look_around_turns_left_and_right_until_it_finds_the_object():
    from brain import params as prm_
    from brain.geometry import look_offsets
    from brain.primitives import Result

    class World:
        """The object is seen when it's within half the view of the base's heading."""

        def __init__(self, bearing, near=False):
            self.geom = geom(-0.11)
            self.geom.front = {"u_per_deg": 0.0184}
            self.heading, self.bearing, self.near, self.turns, self.backs = 0.0, bearing, near, [], 0

        def find(self, obj, cam):
            half_view = 0.5 / self.geom.front["u_per_deg"]
            visible = abs((self.bearing - self.heading + 180) % 360 - 180) <= half_view and not self.near
            return Detection(box=[0.4, 0.4, 0.5, 0.5]) if visible else None

        def move(self, part, delta=None, roll=None, **kw):
            if roll is not None:
                self.turns.append(roll)
                self.heading += roll
            else:
                self.backs += 1
                self.near = False  # backing up brings it into view
            return Result(True)

    w = World(bearing=-50.0)
    r = actions.look_around(w, "red block")
    assert r.ok and abs(w.heading - w.bearing) <= 0.5 / 0.0184
    w = World(bearing=0.0, near=True)
    assert actions.look_around(w, "red block").ok and w.backs == 1
    steps = look_offsets((1 - prm_.LOOK_OVERLAP.value) / 0.0184)
    assert steps[:2] == [steps[0], -steps[0]] and steps[-1] == 180.0
    # Nothing there (Oct 28: no brick placed): all the way round, each turn the short way, and
    # back to the start; the last one asked 343 deg and was refused.
    w = World(bearing=0.0, near=True)
    w.find = lambda obj, cam: None
    assert not actions.look_around(w, "red block").ok
    assert max(abs(t) for t in w.turns) <= 180.0 and abs((w.heading + 180) % 360 - 180) < 1e-6, w.turns


def test_a_base_turn_over_the_pis_limit_goes_in_two():
    sent = []

    class R:
        def base(self, dx=0.0, dy=0.0, dtheta_deg=0.0):
            sent.append(round(dtheta_deg, 2))
            return {"result": {"moved": [dx, dy, dtheta_deg]}}

    sk = Skills.__new__(Skills)
    sk.robot, sk.log = R(), None
    assert sk.move("base", roll=-180.0).ok
    stop = prm.BASE_TURN_STOP_SHORT_DEG.value
    assert sent == [-90.0 - stop, -90.0 - stop] and all(abs(t) <= prm.BASE_TURN_MAX_DEG.value for t in sent)


def test_over_container_uses_the_look_that_brought_it_into_view(monkeypatch):
    # Oct 28: the look right after bring_into_view's got no box for a cup filling the view.
    from brain.primitives import Result

    looks, moves = [Detection(box=[0.17, 0.14, 0.84, 0.97])], []

    class World:
        geom, roll, heights = jaw_geom(), 0.0, {"paper cup": 0.044}

        def height(self):
            return 0.106

        def find(self, obj, cam, angle=False, fast_ok=True, **kw):
            return looks.pop(0) if looks else None  # one box, then nothing

        def move(self, part, delta=None, **kw):
            moves.append((part, delta))
            return Result(True)

        def ev(self, *a, **k):
            pass

    monkeypatch.setattr(actions, "locate_from_one_view", lambda *a, **k: np.array([0.004, -0.003]))
    monkeypatch.setattr(actions, "view_half_width", lambda *a, **k: 0.1)
    r = actions.over_container(World(), "paper cup")
    assert r.ok, r.reason
    assert moves == [("arm", (0.004, -0.003, 0.0))], moves


def test_bring_into_view_moves_towards_the_container_and_stops_if_it_drifts_away():
    from brain.primitives import Result

    class World:
        geom, roll = jaw_geom(), 0.0

        def __init__(self, centre, sign=1.0):
            self.geom.grasp_box = [0.411, 0.613, 0.545, 0.927]
            self.c, self.sign, self.moves = np.array(centre, float), sign, 0

        def height(self):
            return 0.106

        def find(self, obj, cam, angle=False, fast_ok=True, **kw):
            u, v = self.c
            return Detection(box=[u - 0.3, v - 0.33, u + 0.3, v + 0.33])

        def move(self, part, delta=None, **kw):
            self.moves += 1
            self.c = self.c + self.sign * (self.geom.jacobian(0.106) @ np.asarray(delta[:2]))
            return Result(True)

    w = World([0.15, 0.55])  # off the left edge, as on Oct 28
    assert actions.bring_into_view(w, "paper cup").ok and w.c[0] - 0.3 > 0
    w = World([0.15, 0.55], sign=-1.0)  # a world where moves push it further out
    r = actions.bring_into_view(w, "paper cup")
    assert not r.ok and "further out" in r.reason and w.moves == 1


def test_servo_converges_at_the_gain_with_the_jacobian_30_percent_off_either_way():
    for scale in (0.7, 1.3):
        sim = SimServo(jaw_geom(), start_err=[0.12, -0.10], true_scale=scale)
        r = sim.servo("red block", sim.goal)
        assert r.ok and r.data["looks"] <= 6, (scale, r)
