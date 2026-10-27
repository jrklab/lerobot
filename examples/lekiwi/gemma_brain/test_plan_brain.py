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

"""Offline checks for planner.py and plan_brain.py: plan parsing and checking, and the
executor's expansion, retries and replans against fake skills. No robot, no model.

    .venv/bin/python test_plan_brain.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

import plan_brain
from arm import PlanError
from gemma import Answer
from plan_brain import SUPPORTED, PlanBrain
from planner import Step, check_plan, objects_in, parse_plan
from robot import OperatorTookOver, RobotError
from skills import Log, Result


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

    def __init__(self, z=0.15, stuck=False):
        self.z, self.stuck, self.moves = z, stuck, []

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


class FakePlanArm(FakeArm):
    """Can't move straight up (wrist limit), like the arm just after a pick."""

    def __init__(self, z):
        super().__init__(z=z, stuck=True)


class FakeSkills:
    """Records the skill calls; `fail` maps a call like 'fetch:red block' to how many times it fails."""

    def __init__(self, fail=None, arm=None, release_then_fail=False):
        self.calls, self.fail = [], dict(fail or {})
        self.holding, self.base_ok, self.arm = None, True, arm or FakeArm()
        self.release_then_fail = release_then_fail

    def height(self):
        return self.arm.z + 0.13  # the floor at TCP z -0.13

    def _do(self, call: str) -> Result:
        self.calls.append(call)
        if self.fail.get(call, 0) > 0:
            self.fail[call] -= 1
            return Result(False, f"{call} failed")
        return Result(True)

    def scan(self, objects):
        return self._do("scan:" + ",".join(objects))

    def fetch(self, what):
        r = self._do("fetch:" + what)
        if r.ok:
            self.holding = what
        return r

    def go_to(self, what):
        return self._do("go_to:" + what)

    def place_in(self, what):
        r = self._do("place_in:" + what)
        if r.ok:
            self.holding = None
        if self.release_then_fail:  # opened over the target, then the retreat was refused
            self.holding = None
            self.arm.z = 0.015
            return Result(False, "robot error: arm_wrist_flex would hit its limit")
        return r

    def place_next_to(self, what):
        r = self._do("place_next_to:" + what)
        if r.ok:
            self.holding = None
        return r


class FakeRobot:
    def frame(self, cam, after=0.0):
        return b"\xff\xd8jpeg", after


def brain(planner, sk, holding=None):
    log = Log(Path(tempfile.mkdtemp()))
    return PlanBrain(sk, planner, log, FakeRobot(), holding)


def test_parse_plan():
    steps = parse_plan('pick "red block"\nplace "red block" in "cup"\ndone')
    assert steps == [Step("pick", "red block"), Step("place", "red block", "in", "cup")]
    assert str(steps[1]) == 'place "red block" in "cup"'
    for bad in ('pick "a" in "b"', 'place "a" under "b"', 'push "a" in "b"', "hello", "done"):
        try:
            parse_plan(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


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


def test_expansion():
    sk = FakeSkills()
    ok, _ = brain(FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone'), sk).run("red in cup")
    assert ok, sk.calls
    assert sk.calls == ["scan:red block,cup", "fetch:red block", "go_to:cup", "place_in:cup"], sk.calls


def test_unsupported_plan_moves_nothing():
    sk = FakeSkills()
    ok, why = brain(FakePlanner('pick "red block"\nplace "red block" on "blue block"\ndone'), sk).run("stack")
    assert not ok and "can't do 'place on'" in why, why
    assert sk.calls == [], sk.calls


def test_pick_retries_then_succeeds():
    sk = FakeSkills(fail={"fetch:red block": 2})
    ok, _ = brain(FakePlanner('pick "red block"\ndone'), sk).run("pick red")
    assert ok and sk.calls.count("fetch:red block") == 3, sk.calls


def test_replan_after_failure():
    # The place fails once (going to the cup); the replan continues from "holding the red block".
    sk = FakeSkills(fail={"go_to:cup": 1})
    planner = FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone', 'place "red block" in "cup"\ndone')
    ok, _ = brain(planner, sk).run("red in cup")
    assert ok, sk.calls
    done, failed, _, holding = planner.replans[0]
    assert done == [Step("pick", "red block")] and failed == Step("place", "red block", "in", "cup")
    assert holding == "red block"
    assert sk.calls.count("scan:red block,cup") == 1, sk.calls  # no rescan of known objects


def test_gives_up_after_max_replans():
    sk = FakeSkills(fail={"go_to:cup": 99, "go_to:paper cup": 99})
    plans = ['pick "red block"\nplace "red block" in "cup"', 'place "red block" in "paper cup"',
             'place "red block" in "cup"']
    ok, why = brain(FakePlanner(*plans), sk).run("red in cup")
    assert not ok and "replans" in why, why


def test_identical_replan_runs_once_more_only():
    sk = FakeSkills(fail={"go_to:cup": 99})
    plans = ['pick "red block"\nplace "red block" in "cup"'] + ['place "red block" in "cup"'] * 2
    ok, why = brain(FakePlanner(*plans), sk).run("red in cup")
    assert not ok and "didn't change" in why, why
    assert sk.calls.count("go_to:cup") == 2, sk.calls


def test_released_then_failed_counts_as_placed():
    # Oct 26: the brick dropped into the cup, then the retreat was refused. No replan to pick it
    # again; the arm lifts straight up before folding.
    sk = FakeSkills(release_then_fail=True)
    planner = FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone')
    ok, why = brain(planner, sk).run("red in cup")
    assert ok and "unverified" in why, why
    assert planner.replans == [] and sk.calls.count("fetch:red block") == 1, sk.calls
    assert sk.arm.moves == [("move_to", plan_brain.RESET_SAFE_Z_M), ("reset",)], sk.arm.moves


def test_fold_stays_put_when_it_cant_lift_straight_up():
    sk = FakeSkills(release_then_fail=True, arm=FakeArm(stuck=True))
    ok, why = brain(FakePlanner('pick "red block"\nplace "red block" in "cup"\ndone'), sk).run("x")
    assert not ok and "stayed where it was" in why, why
    assert ("reset",) not in sk.arm.moves, sk.arm.moves  # no tilt-and-pull-in fold


def test_fold_from_high_up_uses_the_normal_fold():
    # Oct 26: after a pick the arm is ~19 cm up and can't go straight up; folding is safe there.
    sk = FakeSkills(arm=FakePlanArm(z=0.06))
    assert brain(FakePlanner('pick "a"'), sk).fold() == ""
    assert sk.arm.moves == [("reset",)], sk.arm.moves


def test_fold_after_a_pick_folds_even_when_low():
    # Oct 26: after the 9 cm lift the wrist can't go straight up; not beside a place target,
    # so the normal fold (with its tilt-and-pull-in fallback) is fine.
    sk = FakeSkills(arm=FakePlanArm(z=-0.04))
    b = brain(FakePlanner('pick "red block"'), sk)
    ok, _ = b.run("pick red")
    assert ok and b.fold() == "" and sk.arm.moves == [("reset",)], sk.arm.moves  # main() folds after run()


def test_leaving_auto_mode_stops_without_replanning():
    class Dropped(FakeSkills):
        def fetch(self, what):
            raise RobotError("POST /auto/cmd: HTTP 409: not in auto mode (mode is gamepad)")

    planner = FakePlanner('pick "red block"\ndone')
    try:
        brain(planner, Dropped()).run("pick red")
    except OperatorTookOver as e:
        assert "gamepad disconnecting" in str(e)
    else:
        raise AssertionError("expected OperatorTookOver")
    assert planner.replans == []


def test_starts_holding():
    sk = FakeSkills()
    ok, _ = brain(FakePlanner('place "green block" next_to "red block"\ndone'), sk, holding="green block").run("x")
    assert ok and sk.calls == ["scan:red block", "go_to:red block", "place_next_to:red block"], sk.calls


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
    sys.exit(0)
