
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

"""The primitives (brain/primitives.py): find, move (run by the Pi's AutoController in the
web_server simulation), measured moves, the settle, grip and the base."""

from pathlib import Path

import numpy as np
from brain import actions, params as prm
from brain.primitives import Skills
from common.arm import Arm, Calibration
from helpers import (
    FakeModel,
    SimRobot,
    _real_cal,
    finder,
    geom,
    red_frame,
    skills_over,
)
from test_auto_control import Sim

HERE = Path(__file__).resolve().parent


def test_move_out_of_vertical_reach_is_refused_without_moving():
    sim, sk = skills_over(-0.12, -0.12)
    before = dict(sim.pos)
    r = sk.move(xy=(0.6, 0.0), height=0.10)
    assert not r.ok and "can't move" in r.reason
    assert sim.pos == before

def test_base_moves_are_one_kind_at_a_time():
    sim, sk = skills_over(-0.12, -0.12)
    assert not sk.move("base", delta=(0.05, 0.0), roll=10.0).ok

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

def test_the_roll_ranges_ends_are_moves_the_planner_accepts():
    # Oct 28 full task: a brick at 15 deg, the range ending at ~14: the roll clamped to the end
    # was refused six times (the joint at 100.4, the planner's limit 99).
    # The robot's own calibration, on both sides (the planner's limits and the simulated Pi's).
    # Oct 29 trial 2: at the range's end, a centring step 1.4 cm further out was refused (the
    # same roll needs a slightly different joint there): both ends must also hold within the
    # line-up's tolerance of the reach point.
    from brain import params as prm

    cal = Calibration(*_real_cal())
    for end_i in (0, 1):
        sim = Sim(floor_z=-0.12, angles=cal)
        sk = Skills(SimRobot(sim), Arm(SimRobot(sim), cal, body_check=False), geom(-0.12))
        assert actions.hover(sk).ok
        end = sk.roll_range[end_i]
        r = sk.move("arm", delta=(0.0, 0.0, 0.0), roll=end)
        assert r.ok, (end, r.reason)
        tol = prm.REACH_TOL_M.value
        for dx in (tol, -2 * tol):  # out to the tolerance, then back in past the reach point by it
            r = sk.move("arm", delta=(dx, 0.0, 0.0))
            assert r.ok, (end, dx, r.reason)

def test_only_the_pis_check_refusals_count_as_refused():
    from brain.primitives import pi_refused
    from common.robot import RobotError

    assert pi_refused(RobotError("POST /auto/cmd: HTTP 409: trajectory: the gripper reaches z -0.091 m at waypoint 6, below z_min -0.085 m"))
    assert not pi_refused(RobotError("POST /auto/cmd: HTTP 409: busy: command 12 (trajectory) is still running"))
    assert not pi_refused(RobotError("POST /auto/cmd: HTTP 409: not in auto mode (mode is gamepad)"))
    assert not pi_refused(RobotError("trajectory: stopped: heartbeat lost"))

def test_a_move_crossing_the_clearance_goes_at_the_near_speed_only_below_it():
    # Oct 28 trials: a 13 cm lift from the floor took 13 s at the near speed all the way.
    from common.arm import FLOOR_CLEARANCE_M, JAW_TIP_OFFSET_M

    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    assert actions.land(sk).ok
    # The fitted floor tilts: 5.6 mm lower at the reach point than at x 0, where the free-move
    # guard is taken (Oct 28 fit). Tilted the same way about where the arm is:
    x = sk.cmd_tcp()[0]
    sk.geom.floor_coef = np.array([-0.12 + 0.0224 * x, -0.0224, 0.0])
    sk.arm.floor_z = sk.geom.floor(prm.REACH_POINT_X_M.value, 0.0) - JAW_TIP_OFFSET_M  # as Skills.__init__ sets them
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
    sk.geom = geom(-0.11)
    # degrees open: on the brick's 15.8 mm side (9.9), open and not holding, empty (1.6: the jaws
    # meet), closed further after a drop
    for pos, state, want in ((9.9, "holding", True), (32.0, None, False), (1.6, "holding", False), (9.0, "dropped", False)):
        sk.robot = R(pos, state)
        assert sk.held() is want, (pos, state)


def test_the_gripper_opening_in_mm_follows_the_measured_curve():
    g = geom(-0.11)
    assert g.gripper_mm(1.0) == 0.0  # the jaws meet at 1.41 deg
    assert abs(g.gripper_mm(9.93) - 15.8) < 1e-9 and abs(g.gripper_mm(16.0) - 23.6) < 0.1
    assert abs(g.gripper_mm(40.0) - (47.8 + 5.33 * 16 / 12.35)) < 1e-6  # past the last: its slope
    for mm in (0.0, 5.0, 15.8, 44.5, 60.0):
        assert abs(g.gripper_mm(g.gripper_deg(mm)) - mm) < 1e-9

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
