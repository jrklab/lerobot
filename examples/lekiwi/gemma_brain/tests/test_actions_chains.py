
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

"""Actions and chains (brain/actions.py, brain/chains.py): land, lift, carry, look around, over a
container, the place checks, and the chains' order of steps."""

from pathlib import Path

import numpy as np
import pytest
from brain import actions, chains, params as prm
from brain.goals import Detection
from brain.primitives import Result, Skills
from helpers import (
    _chain_sk,
    geom,
    jaw_geom,
    skills_over,
)

HERE = Path(__file__).resolve().parent


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

def test_lift_right_after_landing_is_allowed():
    # Oct 27 teach: the lift from the grasp height was refused by the free-move floor guard.
    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    assert actions.land(sk).ok
    before = sk.tcp()[2]
    r = sk.move("arm", delta=(0.0, 0.0, prm.LIFT_M.value))
    assert r.ok, r
    assert sk.tcp()[2] > before + prm.LIFT_M.value - 0.01

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

def test_carry_holds_at_the_hover_without_folding():
    from brain.primitives import Result

    calls = []

    class S:
        arm = type("A", (), {"reset": lambda self: calls.append("fold")})()

        def move(self, part, **k):
            calls.append((part, k.get("height")))
            return Result(True)

    assert actions.carry(S()).ok and calls == [("arm", prm.HOVER_HEIGHT_M.value)]

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

def test_pick_chain_runs_its_steps_in_order_and_stops_at_the_first_failure(monkeypatch):
    from brain import chains
    from brain.primitives import Result

    calls = []

    def ok(name):
        return lambda *a, **k: (calls.append(name), Result(True))[1]

    for name in ("approach", "hover", "line_up", "align", "land", "lift"):
        monkeypatch.setattr(chains.actions, name, ok(name))
    sk = Skills.__new__(Skills)
    sk.log, sk.heights, sk._height_fallbacks = None, {}, set()
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


def test_the_approachs_first_drive_uses_the_main_models_look():
    # Oct 29: the look-around's main-model box put the brick 0.47 m away, the next (fast-model)
    # look 0.56 m; the drive went by the second and lost the brick.
    from brain.primitives import Result

    looks, used = [], []

    def make(by):
        return Detection(box=[0.48, 0.40, 0.53, 0.44], by=by)

    class S:
        seen, heights = {}, {}

        def __init__(self, first_by):
            self.first_by = first_by

        def find(self, obj, cam, fast_ok=True, **k):
            looks.append(fast_ok)
            return make(self.first_by if len(looks) == 1 else "main")

        def servo(self, obj, goal, part=None, first=None):
            used.append(first.by)
            return Result(True)

        def ev(self, *a, **k):
            pass

    assert actions.approach(S("main"), "red block").ok and used == ["main"] and len(looks) == 1  # no second look
    looks.clear(), used.clear()
    assert actions.approach(S("fast"), "red block").ok and used == ["main"] and looks == [True, False]


def _height_sk(heights):
    sk = Skills.__new__(Skills)
    sk.log, sk.heights, sk._height_fallbacks = None, dict(heights), set()
    sk.events = []
    sk.ev = lambda name, **k: sk.events.append((name, k))
    return sk


def test_an_objects_measured_height_sets_its_last_look_and_servo_plane(monkeypatch):
    """Oct 31: the light green block measures ~23 mm, twice the brick the values were set for."""
    calls = []
    for name in ("approach", "hover", "line_up", "align", "land", "lift"):
        monkeypatch.setattr(chains.actions, name, lambda *a, **k: Result(True))
    sk = _height_sk({"light green block": 0.023})
    sk.servo = lambda obj, goal, part=None: (calls.append(("servo", goal.plane_h)), Result(True))[1]
    sk.move = lambda part, **k: (calls.append(("down", k["height"])), Result(True))[1]
    sk.grip = lambda how, **k: Result(True)
    assert chains.pick(sk, "light green block").ok
    look = 0.023 + prm.LOOK_CLEARANCE_M.value
    assert calls == [("servo", 0.023), ("down", pytest.approx(look)), ("servo", 0.023)]


def test_an_unmeasured_object_takes_the_taught_height_noted_once():
    sk = _height_sk({"red block": 0.0})  # 0.0: the front camera couldn't place its bottom edge
    for _ in range(2):
        assert sk.object_height("red block") == prm.TAUGHT_OBJECT_HEIGHT_M.value
    assert [e for e, _ in sk.events] == ["object_height_fallback"]
    assert sk.final_look_height(None) == pytest.approx(prm.FINAL_LOOK_HEIGHT_M.value)


@pytest.mark.parametrize("height, above, refused", [(0.0114, 0.008, True), (0.023, 0.008, False), (0.023, 0.014, True)])
def test_the_landing_refuses_a_contact_over_half_the_objects_height(height, above, refused):
    sk = _height_sk({"block": height})
    sk.height = lambda: 0.0
    sk.tcp = lambda: np.array([0.25, 0.0, -0.10])
    sk.frame = lambda *a, **k: None
    sk.move = lambda part, **k: Result(True, data={"contact": True, "contact_z": -0.10 + above, "floor": -0.10})
    r = actions.land(sk, "block")
    assert (not r.ok) == refused


def test_place_heights_follow_the_targets_measured_height(monkeypatch):
    """The demo's release and hover above its own target's top, over a target of another height."""
    monkeypatch.setattr(prm, "demo_place_offsets", lambda: {"release": 0.0064, "hover": 0.0381})
    sk = _height_sk({"paper cup": 0.044, "plate": 0.015})
    assert actions.place_height(sk, "paper cup", "release") == pytest.approx(0.0504)
    assert actions.place_height(sk, "plate", "hover") == pytest.approx(0.0531)
    assert sk.events == []


def test_without_offsets_or_a_measured_target_the_demos_absolute_heights_stand(monkeypatch):
    sk = _height_sk({})
    monkeypatch.setattr(prm, "demo_place_offsets", lambda: {"release": 0.0064, "hover": 0.0381})
    assert actions.place_height(sk, "bowl", "release") == prm.RELEASE_ABOVE_FLOOR_M.value  # not measured
    monkeypatch.setattr(prm, "demo_place_offsets", lambda: None)
    sk.heights["bowl"] = 0.05
    assert actions.place_height(sk, "bowl", "hover") == prm.PLACE_HOVER_M.value  # no offsets
    assert [k["why"] for _, k in sk.events] == ["the target's height wasn't measured", "no demo offsets"]
