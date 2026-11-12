#!/usr/bin/env python

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

"""Offline checks for planner.py and the task loop (brain/task_loop.py): plan parsing and checking,
and the loop's steps, retries, replans and fold against fake chains. No robot, no model.

    uv run pytest tests/test_task_loop.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from brain import actions, chains, params as prm
from brain.primitives import Result
from brain.task_loop import SUPPORTED, PlanBrain
from common.arm import RESET_SAFE_Z_M, PlanError
from common.gemma import Answer
from common.planner import Step, check_plan, objects_in, parse_plan
from common.robot import OperatorTookOver, RobotError
from common.runlog import Log


class FakeModel:
    model = "fake"


class FakePlanner:
    """Returns the queued plans in order: the first for plan(), the rest for replan()."""

    def __init__(self, *plans: str):
        self.model = FakeModel()
        self.plans = list(plans)
        self.replans = []

    def plan(self, task, front, wrist, holding=None):
        return parse_plan(self.plans.pop(0)), Answer("", ms=1.0)

    def replan(self, task, done, failed, reason, holding, front, wrist):
        self.replans.append((list(done), failed, reason, holding))
        return parse_plan(self.plans.pop(0)), Answer("", ms=1.0)


class FakeArm:
    """At z (m, arm frame); move_to refuses when `stuck`. Records moves."""

    z_min = -0.065

    def __init__(self, z=0.15, stuck=False, at_rest=True):
        self.z, self.stuck, self.moves, self.at_rest = z, stuck, [], at_rest

    def at_pose(self, name):
        return self.at_rest


    def tcp(self):
        m = np.eye(4)
        m[:3, 3] = [0.2, 0.0, self.z]
        return m

    def move_to(self, xyz, **kw):
        if self.stuck:
            raise PlanError("arm_wrist_flex would hit its limit")
        self.moves.append(("move_to", round(xyz[2], 3)))
        self.z = xyz[2]

    def reset(self):
        self.moves.append(("reset",))
        self.at_rest = True


class FakeSkills:
    """What the task loop reads off Skills: what's held, the arm, the height."""

    def __init__(self, arm=None):
        self.holding, self.arm, self.roll, self.seen = None, arm or FakeArm(), 0.0, {}
        self.gripped = False
        self.pose, self.found, self.base_moves = [0.0, 0.0, 0.0], [], []

    def height(self):
        return self.arm.z + 0.13  # the floor at TCP z -0.13

    def held(self):
        return self.gripped

    def odom(self):
        return np.array(self.pose)

    def find(self, obj, cam, **kw):
        self.found.append(obj)
        return None


class FakeRobot:
    def frame(self, cam, after=0.0):
        return b"\xff\xd8jpeg", after


@pytest.fixture
def world(monkeypatch):
    """Fake chains recording their calls; `fail` maps a call like 'pick red block' to how many
    times it fails; release_then_fail: a place opens the jaws, then fails (Oct 26); the hover
    before a fold is refused when hover_ok is False."""

    class W:
        calls, fail, release_then_fail, hover_ok = [], {}, False, True
        fail_data: dict = {}  # what a failed chain's result carries, e.g. {"failed": "land"}

    w = W()
    w.calls, w.fail, w.fail_data = [], {}, {}

    def do(call, sk):
        w.calls.append(call)
        if w.fail.get(call, 0) > 0:
            w.fail[call] -= 1
            return Result(False, f"{call} failed", dict(w.fail_data))
        return Result(True)

    def pick(sk, what, drive=True, start=None):
        r = do(f"pick {what}", sk)
        if start is not None:
            w.calls[-1] += f" from {start}"
        sk.gripped = r.ok
        return r

    def place(rel):
        def run(sk, what, target, drive=True):
            r = do(f"place {what} {rel} {target}", sk)
            if r.ok:
                sk.gripped = False
            if w.release_then_fail:  # opened over the target, then the retreat was refused
                sk.gripped, sk.arm.z = False, 0.015
                return Result(False, "robot error: arm_wrist_flex would hit its limit")
            return r

        return run

    monkeypatch.setattr(chains, "pick", pick)
    monkeypatch.setattr(chains, "place_in", place("in"))
    monkeypatch.setattr(chains, "place_next_to", place("next_to"))
    monkeypatch.setattr(actions, "hover_at", lambda sk, h: (w.calls.append("hover"), Result(w.hover_ok))[1])
    return w


def brain(planner, sk, holding=None):
    return PlanBrain(sk, planner, Log(Path(tempfile.mkdtemp())), FakeRobot(), holding)


def test_parse_plan():
    steps = parse_plan('pick "red block"\nplace "red block" in "cup"\ndone')
    assert steps == [Step("pick", "red block"), Step("place", "red block", "in", "cup")]
    assert str(steps[1]) == 'place "red block" in "cup"'
    for bad in ('pick "a" in "b"', 'place "a" under "b"', 'push "a" in "b"', "hello", "done"):
        with pytest.raises(ValueError):
            parse_plan(bad)


def test_check_plan():
    ok = parse_plan('pick "red block"\nplace "red block" in "cup"\npick "blue block"\nplace "blue block" next_to "cup"')
    assert check_plan(ok, SUPPORTED, None) == []
    assert objects_in(ok) == ["red block", "cup", "blue block"]
    on = check_plan(parse_plan('pick "red block"\nplace "red block" on "blue block"'), SUPPORTED, None)
    assert len(on) == 1 and "place on" in on[0]
    twice = check_plan(parse_plan('pick "a"\npick "b"'), SUPPORTED, None)
    assert any("already holding" in p for p in twice)
    assert check_plan(parse_plan('place "a" in "cup"'), SUPPORTED, "a") == []
    assert any("not holding" in p for p in check_plan(parse_plan('place "a" in "cup"'), SUPPORTED, None))
    assert any("push" in p for p in check_plan(parse_plan('push "a" toward "b"'), SUPPORTED, None))


def test_runs_the_steps_on_the_chains_and_tracks_what_is_held(world):
    sk = FakeSkills()
    ok, _ = brain(FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone'), sk).run("red in cup")
    assert ok and world.calls == ["pick red block", "place red block in cup"], world.calls
    assert sk.holding is None


def test_unsupported_plan_moves_nothing(world):
    ok, why = brain(FakePlanner('pick "red block"\nplace "red block" on "blue block"\ndone'), FakeSkills()).run("stack")
    assert not ok and "can't do 'place on'" in why, why
    assert world.calls == [], world.calls


def test_a_place_relation_without_a_chain_runs_nothing(world):
    sk = FakeSkills()
    sk.holding = "red block"
    r = brain(FakePlanner(), sk).place("on", "blue block")
    assert not r.ok and "no chain" in r.reason and world.calls == [], world.calls


def test_pick_retries_from_the_hover_without_folding_then_succeeds(world):
    world.fail = {"pick red block": 2}
    sk = FakeSkills()
    ok, _ = brain(FakePlanner('pick "red block"\ndone'), sk).run("pick red")
    assert ok and world.calls == ["pick red block", "hover"] * 2 + ["pick red block"], world.calls
    assert sk.holding == "red block" and ("reset",) not in sk.arm.moves, sk.arm.moves  # no fold


def test_a_start_over_folds_where_the_hover_is_refused(world):
    world.fail, world.hover_ok = {"pick red block": 1}, False
    sk = FakeSkills()
    ok, _ = brain(FakePlanner('pick "red block"\ndone'), sk).run("pick red")
    assert ok and ("reset",) in sk.arm.moves, sk.arm.moves


def test_a_refused_pick_is_not_retried(world):
    # Oct 28: a wrist roll past its limit was refused six times over two plans (~140 s).
    world.fail = {"pick red block": 99}
    world.fail_data = {"failed": "align", "refused": True}
    ok, why = brain(FakePlanner('pick "red block"'), FakeSkills()).run("pick red")
    assert not ok and "refused again" in why, why
    assert world.calls.count("pick red block") == 2, world.calls  # tried, and once more


def test_a_pick_failing_after_the_hover_resumes_there_without_folding(world):
    world.fail = {"pick red block": 1}
    world.fail_data = {"failed": "centre_low"}
    ok, _ = brain(FakePlanner('pick "red block"\ndone'), FakeSkills()).run("pick red")
    assert ok and world.calls == ["pick red block", "pick red block from hover"], world.calls


def test_a_second_failure_after_resuming_starts_over_from_the_approach(world):
    world.fail = {"pick red block": 2}
    world.fail_data = {"failed": "land"}
    sk = FakeSkills()
    ok, _ = brain(FakePlanner('pick "red block"\ndone'), sk).run("pick red")
    assert ok and world.calls == ["pick red block", "pick red block from hover", "hover", "pick red block"], world.calls
    assert ("reset",) not in sk.arm.moves, sk.arm.moves  # parked at the hover, not folded


def test_a_failed_approach_starts_over_from_the_hover(world):
    world.fail = {"pick red block": 1}
    world.fail_data = {"failed": "approach"}
    ok, _ = brain(FakePlanner('pick "red block"\ndone'), FakeSkills()).run("pick red")
    assert ok and world.calls == ["pick red block", "hover", "pick red block"], world.calls


def test_a_pick_that_gripped_before_a_later_step_failed_keeps_it(world, monkeypatch):
    sk = FakeSkills()
    monkeypatch.setattr(chains, "pick", lambda sk, what, drive=True, start=None: (setattr(sk, "gripped", True), Result(False, "lift: refused"))[1])
    ok, why = brain(FakePlanner('pick "red block"\ndone'), sk).run("pick red")
    assert ok and sk.holding == "red block", why


def test_a_failed_step_runs_the_rest_of_the_plan_again_as_planned(world):
    # The place fails all its tries in code; still holding, the plan's place runs again, to the
    # same target: no planner (Nov 11: its replan after a failed pick sent the ball to the cup).
    world.fail = {"place red block in cup": int(prm.PLACE_TRIES.value)}
    planner = FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone')
    ok, _ = brain(planner, FakeSkills()).run("red in cup")
    assert ok and planner.replans == [], world.calls
    assert world.calls.count("pick red block") == 1 and world.calls[-1] == "place red block in cup"


def test_a_failed_pick_runs_again_and_the_destinations_stay(world):
    world.fail = {"pick tennis ball": int(prm.PICK_TRIES.value)}
    plan = 'pick "tennis ball"\nplace "tennis ball" in "box"\npick "red brick"\nplace "red brick" in "cup"\ndone'
    ok, _ = brain(FakePlanner(plan), FakeSkills()).run("x")
    assert ok
    assert [c for c in world.calls if c.startswith("place")] == ["place tennis ball in box", "place red brick in cup"]


def test_gives_up_after_max_retries_of_the_plan(world):
    world.fail = {"place red block in cup": 99}
    ok, why = brain(FakePlanner('pick "red block"\nplace "red block" in "cup"'), FakeSkills()).run("red in cup")
    assert not ok and f"after {int(prm.MAX_REPLANS.value)} retries" in why, why
    tries = (int(prm.MAX_REPLANS.value) + 1) * int(prm.PLACE_TRIES.value)
    assert world.calls.count("place red block in cup") == tries, world.calls


def test_released_then_failed_counts_as_placed(world):
    # Oct 26: the brick dropped into the cup, then the retreat was refused. No replan to pick it
    # again; the arm folds via the hover.
    world.release_then_fail = True
    planner = FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone')
    ok, why = brain(planner, FakeSkills()).run("red in cup")
    assert ok and "unverified" in why, why
    assert planner.replans == [] and world.calls.count("pick red block") == 1, world.calls


def test_a_place_the_check_finds_wrong_is_replanned_not_counted(world, monkeypatch):
    # Released, then the look found it outside the cup: a failed step, not "placed, unverified".
    def missed(sk, what, target, drive=True):
        world.calls.append(f"place {what} in {target}")
        sk.gripped = False
        return Result(False, f"check: the {what} is not inside the {target}", {"failed": "check"})

    monkeypatch.setattr(chains, "place_in", missed)
    planner = FakePlanner('pick "red block"\nplace "red block" in "cup"')
    ok, why = brain(planner, FakeSkills()).run("red in cup")
    # not held any more: picked again, then placed again, in the same cup
    assert world.calls[:4] == ["pick red block", "place red block in cup", "hover", "pick red block"], world.calls
    assert "unverified" not in why


def test_fold_goes_via_the_hover(world):
    sk = FakeSkills()
    assert brain(FakePlanner('pick "a"'), sk).fold() == ""
    assert world.calls == ["hover"] and sk.arm.moves == [("reset",)], sk.arm.moves


def test_fold_lifts_straight_up_where_the_hover_is_refused(world):
    world.hover_ok = False
    sk = FakeSkills(arm=FakeArm(z=-0.11))
    assert brain(FakePlanner('pick "a"'), sk).fold() == ""
    assert sk.arm.moves == [("move_to", RESET_SAFE_Z_M), ("reset",)], sk.arm.moves


def test_fold_stays_put_beside_a_place_target_when_it_cant_lift(world):
    world.hover_ok, world.release_then_fail = False, True
    sk = FakeSkills(arm=FakeArm(stuck=True))
    ok, why = brain(FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone'), sk).run("x")
    assert not ok and "stayed where it was" in why, why
    assert ("reset",) not in sk.arm.moves, sk.arm.moves  # no tilt-and-pull-in fold


def test_fold_away_from_a_place_target_folds_even_when_it_cant_lift(world):
    # Oct 26: after the lift the wrist can't go straight up; not beside a place target, so
    # the normal fold (with its tilt-and-pull-in fallback) is fine.
    world.hover_ok = False
    sk = FakeSkills(arm=FakeArm(z=-0.04, stuck=True))
    assert brain(FakePlanner('pick "a"'), sk).fold() == "" and sk.arm.moves == [("reset",)], sk.arm.moves


def test_leaving_auto_mode_stops_without_replanning(world, monkeypatch):
    def dropped(sk, what, drive=True, start=None):
        raise RobotError("POST /auto/cmd: HTTP 409: not in auto mode (mode is gamepad)")

    monkeypatch.setattr(chains, "pick", dropped)
    planner = FakePlanner('pick "red block"\ndone')
    with pytest.raises(OperatorTookOver, match="gamepad disconnecting"):
        brain(planner, FakeSkills()).run("pick red")
    assert planner.replans == []


def test_starts_holding(world):
    sk = FakeSkills()
    sk.gripped = True
    ok, _ = brain(FakePlanner('place "green block" next_to "red block"\ndone'), sk, holding="green block").run("x")
    assert ok and world.calls == ["place green block next_to red block"], world.calls


def test_jaws_closed_at_the_start_are_holding_what_the_plan_places(world):
    # Nov 6: the ball was still held from the run before; the plan picked it again.
    sk = FakeSkills()
    sk.gripped = True
    b = brain(FakePlanner('place "tennis ball" in "shoe box"\ndone'), sk)
    ok, _ = b.run("put the tennis ball in the shoe box")
    assert ok and world.calls == ["place tennis ball in shoe box"], world.calls


def test_jaws_closed_at_the_start_refuse_a_plan_that_picks(world):
    sk = FakeSkills()
    sk.gripped = True
    ok, why = brain(FakePlanner('pick "tennis ball"\nplace "tennis ball" in "shoe box"\ndone'), sk).run("x")
    assert not ok and "already holding" in why and world.calls == []


def test_a_run_starts_from_rest_folding_a_raised_arm_first(world):
    # Nov 4: raised by hand, the first move down to the hover passed the lift's limit.
    sk = FakeSkills(arm=FakeArm(at_rest=False))
    ok, _ = brain(FakePlanner('pick "a"'), sk).run("pick a")
    assert ok and world.calls[:2] == ["hover", "pick a"] and sk.arm.moves[0] == ("reset",)
    sk = FakeSkills()  # already at rest: nothing moves first
    brain(FakePlanner('pick "a"'), sk).run("pick a")
    assert sk.arm.moves == []


def test_a_place_failing_before_the_release_retries_in_code_while_holding(world):
    # Nov 4: the model boxed white sneakers as the cup, the approach lost it, and the replan
    # asked to pick the cup while holding the brick.
    world.fail = {"place red block in cup": 1}
    sk = FakeSkills()
    sk.gripped = True
    r = brain(FakePlanner(), sk, holding="red block").place("in", "cup")
    assert r.ok and world.calls == ["place red block in cup", "hover", "place red block in cup"], world.calls
    assert sk.holding is None


def test_a_place_keeps_failing_holding_and_gives_up_after_its_tries(world):
    world.fail = {"place red block in cup": 5}
    sk = FakeSkills()
    sk.gripped = True
    r = brain(FakePlanner(), sk, holding="red block").place("in", "cup")
    assert not r.ok and "failed 2 times" in r.reason and world.calls.count("place red block in cup") == 2
    assert sk.holding == "red block"


def test_the_plans_objects_are_located_from_where_the_task_starts(world):
    # Nov 11: after the ball went in the box the brick was out of view; seen at the start, a later
    # step turns straight to it (actions.look_around).
    sk = FakeSkills()
    plan = 'pick "tennis ball"\nplace "tennis ball" in "black box"\npick "red brick"\nplace "red brick" in "paper cup"\ndone'
    ok, _ = brain(FakePlanner(plan), sk).run("x")
    assert ok and sk.found == ["tennis ball", "black box", "red brick", "paper cup"]
    sk = FakeSkills()
    sk.gripped = True  # held from the start: not looked for on the floor
    brain(FakePlanner('place "green block" next_to "red block"\ndone'), sk, holding="green block").run("x")
    assert sk.found == ["red block"]


def test_park_folds_then_drives_and_turns_back_where_the_task_began(world):
    import math

    sk = FakeSkills()
    moves = []
    sk.move = lambda part, delta=None, roll=None, **k: (moves.append((part, delta, roll)), Result(True))[1]
    b = brain(FakePlanner('pick "red block"\ndone'), sk)
    b.run("x")  # the task began at the odometry's origin, facing 0
    sk.pose = [0.3, 0.4, math.radians(90.0)]  # ended 0.3 forward, 0.4 left, turned 90 left
    world.calls.clear()
    assert b.park() == ""
    assert world.calls[0] == "hover"  # folded first (fold goes via the hover)
    (_, drive, _), (_, _, turn) = moves
    assert drive == (pytest.approx(-0.4), pytest.approx(0.3))  # in the base's frame, facing +y
    assert turn == pytest.approx(-90.0)


def test_park_without_odometry_stays(world):
    sk = FakeSkills()
    b = brain(FakePlanner('pick "red block"\ndone'), sk)
    b.run("x")
    sk.odom = lambda: None
    assert b.park() == "no odometry"
