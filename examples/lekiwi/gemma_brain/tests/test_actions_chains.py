
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
    r = sk.move("arm", delta=(0.0, 0.0, prm.HOVER_HEIGHT_M.value))
    assert r.ok, r
    assert sk.tcp()[2] > before + prm.HOVER_HEIGHT_M.value - 0.01

def test_lift_halves_where_the_full_lift_is_out_of_reach():
    from brain.primitives import Result

    tried = []

    class S:
        heights, held_below = {}, 0.004

        def height(self):
            return 0.004

        def move(self, part, delta=None, **k):
            tried.append(round(delta[2], 4))
            return Result(delta[2] < 0.05, "" if delta[2] < 0.05 else "can't move there")

        def held(self):
            return True

    r = actions.lift(S())
    up = prm.HOVER_HEIGHT_M.value - 0.004  # up to the carry height: nothing taller measured
    assert r.ok and tried == [round(up, 4), round(up / 2, 4)]


def test_the_carry_clears_the_tallest_measured_object():
    sk = _height_sk({"red block": 0.012, "tall box": 0.12})
    sk.held_below = 0.004
    assert actions.carry_height(sk) == pytest.approx(0.12 + 0.004 + prm.PLACE_CLEARANCE_M.value)
    sk.heights = {"cup": 0.068}  # lower than the hover: the hover height stands
    assert actions.carry_height(sk) == prm.HOVER_HEIGHT_M.value


def test_a_held_object_hiding_the_view_is_carried_its_own_height_higher(monkeypatch):
    # Nov 6: carried at the hover the ball hid the shoe box's marker from the front camera.
    sk = _height_sk({"tennis ball": 0.062})
    sk.held_below, sk.holding = 0.032, "tennis ball"
    low = actions.carry_height(sk)
    monkeypatch.setattr(actions, "held_blocks_view", lambda sk: True)
    assert actions.carry_height(sk) == pytest.approx(low + 0.062)


def test_the_close_notes_how_far_the_object_reaches_below_the_jaws():
    from brain.primitives import Result

    sk = _height_sk({})
    sk.height = lambda: 0.0042
    sk.grip = lambda how, **k: Result(True)
    assert actions.close(sk).ok and sk.held_below == 0.0042

def test_carry_holds_at_the_hover_without_folding():
    from brain.primitives import Result

    calls = []

    class S:
        arm = type("A", (), {"reset": lambda self: calls.append("fold")})()
        heights, held_below = {}, None

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

        def find(self, obj, cam, fast_ok=True):
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
    n = steps.index(0.0)
    assert steps[:2] == [steps[0], 2 * steps[0]] and steps[n + 1 :] == [-a for a in steps[:n]] + [180.0]
    # One way, then the other: every turn one step but the turn back (Nov 6: 175 deg back and forth).
    turns = [actions.wrap_deg(b - a) for a, b in zip([0.0] + steps, steps, strict=False)]
    assert max(abs(t) for i, t in enumerate(turns) if i != n) <= 2 * steps[0] + 1e-6
    # Nothing there (Oct 28: no brick placed): all the way round, each turn the short way, and
    # back to the start; the last one asked 343 deg and was refused.
    w = World(bearing=0.0, near=True)
    w.find = lambda obj, cam, fast_ok=True: None
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

    S.widths = {}  # the cup's width unmeasured: any red block on the floor is a miss
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
    sk.seen, sk.looks, sk.grasp_width = {}, {}, None
    sk.widest, sk.grasp_plan = {}, None
    sk.frame = lambda *a, **k: None
    sk.grasp_open_mm = None
    sk.height = lambda: 0.004
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
    monkeypatch.setattr(chains.actions, "approach", lambda sk, o, tol=None, fast_ok=True: (calls.append("approach" if not fast_ok else "approach fast"), chains.Result(True))[1])
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

        def servo(self, obj, goal, part=None, first=None, fast_ok=True):
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
    sk.widths, sk.held_below, sk.geom = {}, None, None
    sk.seen, sk.looks, sk.grasp_width = {}, {}, None
    sk.widest, sk.grasp_plan = {}, None
    sk.frame = lambda *a, **k: None
    sk.grasp_open_mm = None
    sk.height = lambda: 0.004
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


def test_the_release_puts_the_held_objects_bottom_just_over_the_targets_top(monkeypatch):
    """Nov 4: measured, not the demo's offsets: the cup's top + how far the brick reaches below
    the jaws + the clearance; the demo's height for the same cup logged beside it."""
    monkeypatch.setattr(prm, "demo_place_offsets", lambda: {"release": 0.0064, "hover": 0.0381})
    sk = _height_sk({"cup": 0.068})
    sk.held_below = 0.004
    release = 0.068 + 0.004 + prm.PLACE_CLEARANCE_M.value
    assert actions.place_height(sk, "cup", "release") == pytest.approx(release)
    assert sk.events[-1] == ("place_height", {"target": "cup", "which": "release", "height_m": round(release, 4),
                                              "demo_m": 0.0744})


def test_the_hover_is_the_lowest_the_whole_target_is_in_view(monkeypatch):
    sk = _height_sk({"cup": 0.068, "plate": 0.015})
    sk.held_below = 0.004
    release = 0.068 + 0.004 + prm.PLACE_CLEARANCE_M.value
    # no width measured: just over the release
    assert actions.place_height(sk, "cup", "hover") == pytest.approx(release + prm.LOOK_CLEARANCE_M.value)
    seen = []
    monkeypatch.setattr(actions, "lowest_view_height", lambda g, w, h, start: (seen.append((w, h, start)), 0.15)[1])
    sk.widths = {"cup": 0.08}
    assert actions.place_height(sk, "cup", "hover") == 0.15
    assert seen == [(0.08, 0.068, pytest.approx(release + prm.LOOK_CLEARANCE_M.value))]


def test_the_lowest_view_height_steps_up_until_the_width_fits(monkeypatch):
    from brain import geometry

    monkeypatch.setattr(geometry, "view_half_width", lambda g, h, plane_h: (h - plane_h) / 2)
    h = geometry.lowest_view_height(None, 0.08, 0.068, 0.097)  # fits from 0.148 up
    assert 0.148 <= h < 0.148 + geometry.VIEW_HEIGHT_STEP_M
    assert geometry.lowest_view_height(None, 10.0, 0.068, 0.097) == geometry.MAX_VIEW_HEIGHT_M


def test_without_a_measured_target_the_demos_absolute_heights_stand():
    sk = _height_sk({})
    assert actions.place_height(sk, "bowl", "release") == prm.RELEASE_ABOVE_FLOOR_M.value
    assert actions.place_height(sk, "bowl", "hover") == prm.PLACE_HOVER_M.value
    assert [k["why"] for _, k in sk.events] == ["the target's height wasn't measured"] * 2


class _Geom:
    """A wrist camera reading 1 image fraction as 0.1 m, the taught gap 2.3 mm, jaws to 47.8 mm."""

    def wrist_distance(self, a, b, height, plane_h, roll_deg=0.0):
        return 0.1 * float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def fixed_jaw_gap(self, object_height):
        return 0.0023

    def gripper_max_mm(self):
        return 47.8


def _width_sk(monkeypatch, extent, short):
    from brain.goals import Detection
    from brain.primitives import Result

    sk = _height_sk({"red block": 0.0144})
    sk.geom, sk.roll, sk.closed_mm, opened = _Geom(), 0.0, None, []
    sk.seen = {"wrist:red block": Detection(box=[0.4, 0.5, 0.55, 0.77], jpeg=b"x")}
    monkeypatch.setattr(actions, "brick_angle", lambda jpeg, box, look=None: {
        "extent_u": [[0.4, 0.6], [0.4 + extent / 0.1, 0.6]], "across": [[0.4, 0.6], [0.4 + short / 0.1, 0.6]],
        "along": [[0.4, 0.6], [0.4, 0.6 + 2 * short / 0.1]]})
    sk.grip = lambda how, width=None, **k: (opened.append((how, width)), Result(True, data={"mm": sk.closed_mm}))[1]
    return sk, opened


def test_the_jaws_open_to_the_measured_width_before_landing(monkeypatch):
    # Nov 4: the brick's 15.8 mm short side read 14.8-16.7 mm; at a residual angle it spans more.
    sk, opened = _width_sk(monkeypatch, extent=0.0195, short=0.0182)
    assert actions.open_to_fit(sk, "red block").ok
    want = 1000 * (0.0195 + 0.0023 + 2 * (prm.SERVO_TOL_M.value + prm.WIDTH_ERROR_M.value))
    assert opened == [("open", pytest.approx(want))] and sk.grasp_width == pytest.approx(0.0182)
    sk, opened = _width_sk(monkeypatch, extent=0.050, short=0.03)
    r = actions.open_to_fit(sk, "red block")
    assert not r.ok and "wider than they open" in r.reason and opened == []


def test_unmeasured_the_jaws_stay_open_as_they_are(monkeypatch):
    sk, opened = _width_sk(monkeypatch, 0.02, 0.018)
    monkeypatch.setattr(actions, "brick_angle", lambda *a, **k: None)
    assert actions.open_to_fit(sk, "red block").ok and opened == [] and sk.grasp_width is None


def test_a_close_well_past_the_measured_width_is_a_miss(monkeypatch):
    sk, _ = _width_sk(monkeypatch, 0.0195, 0.0182)
    sk.grasp_width = 0.0182
    sk.closed_mm = 16.7  # the brick, squeezed: Nov 4
    assert actions.close(sk).ok
    sk.closed_mm = 9.0  # on an edge of it, or something smaller
    r = actions.close(sk)
    assert not r.ok and "missed it" in r.reason


def test_a_red_thing_far_from_the_cup_is_not_the_missed_brick():
    # Nov 4: the check boxed a basketball 2 m off as the red block; the brick was in the cup.
    from brain.goals import Detection

    class G:
        def front_point(self, uv):
            return np.array([2.2, -0.9, 0.0]) if uv[1] < 0.5 else np.array([0.30, 0.04, 0.0])

    class S:
        geom, widths, events = G(), {"paper cup": 0.095}, []

        def __init__(self, box):
            self.box = box

        def find(self, obj, cam, fast_ok=True, **k):
            return Detection(box=self.box)

        def ev(self, name, **k):
            self.events.append(k)

    assert actions.check_in(S([0.941, 0.303, 0.993, 0.373]), "red block", "paper cup").ok  # far: not it
    r = actions.check_in(S([0.4, 0.62, 0.5, 0.7]), "red block", "paper cup")  # by the cup: missed
    assert not r.ok and "on the floor" in r.reason


def test_a_box_on_the_containers_face_is_not_the_object_on_the_floor():
    # Nov 4: the cup's green band boxed as the light green block (in the cup).
    from types import SimpleNamespace

    from brain.goals import Detection

    class G:
        def front_point(self, uv):
            return np.array([0.26, 0.0, 0.0])  # by the cup: near enough to be a miss

    class S:
        geom, widths, events = G(), {"paper cup": 0.095}, []

        def __init__(self, box):
            self.box = box
            self.main = SimpleNamespace(box=lambda jpeg, prompt: SimpleNamespace(value=[0.31, 0.30, 0.84, 0.84]))

        def find(self, obj, cam, fast_ok=True, **k):
            return Detection(box=self.box, jpeg=b"frame")

        def ev(self, name, **k):
            self.events.append(k)

    assert actions.check_in(S([0.354, 0.583, 0.742, 0.812]), "light green block", "paper cup").ok  # on its face
    r = actions.check_in(S([0.45, 0.80, 0.52, 0.90]), "light green block", "paper cup")  # below its bottom: fallen
    assert not r.ok and "on the floor" in r.reason


def test_the_widest_band_is_low_on_a_brick_and_at_a_balls_middle():
    # A brick's side rows (Nov 4, run 20261104T102537): narrow at the floor's shadow and the top.
    brick = [(-0.0018, 0.009), (0.0012, 0.022), (0.0042, 0.0313), (0.0057, 0.0312), (0.0086, 0.0311),
             (0.0100, 0.0307), (0.0114, 0.0282), (0.0141, 0.012)]
    lo, hi = actions.widest_band(brick, 0.0152)
    assert lo == pytest.approx(0.0042) and hi == pytest.approx(0.0100)
    r = 0.0335  # a tennis ball
    ball = [(h, 2 * np.sqrt(max(r * r - (h - r) ** 2, 0.0))) for h in np.linspace(0.0, 2 * r, 68)]
    lo, hi = actions.widest_band(ball, 2 * r)
    assert 0.020 < lo < r < hi < 0.047
    assert actions.widest_band([], 0.01) is None


def _plan_sk(aspect, widest, height):
    from brain.goals import Detection

    sk = _height_sk({"thing": height})
    sk.seen = {"wrist:thing": Detection(box=[0.4, 0.4, 0.6, 0.6], jpeg=b"x")}
    sk.widest = {} if widest is None else {"thing": widest}
    return sk


def test_the_grasp_plan_follows_the_measured_shape(monkeypatch):
    monkeypatch.setattr(actions, "brick_angle", lambda jpeg, box, look=None: {"aspect": 2.0})
    sk = _plan_sk(2.0, (0.0042, 0.0100), 0.0152)  # a brick: as before
    p = actions.plan_grasp(sk, "thing").data
    assert not p["round"] and p["close_m"] == pytest.approx(0.0042) and p["outline_m"] == pytest.approx(0.0152)
    monkeypatch.setattr(actions, "brick_angle", lambda jpeg, box, look=None: {"aspect": 1.05})
    sk = _plan_sk(1.05, (0.024, 0.043), 0.067)  # a tennis ball: no roll, close below its middle
    p = actions.plan_grasp(sk, "thing").data
    assert p["round"] and p["outline_m"] == pytest.approx(0.043)
    assert p["close_m"] == pytest.approx(max((0.024 + 0.043) / 2, 0.067 - prm.JAW_DEPTH_M.value))  # its middle
    assert p["width_m"] is None  # no side width measured in this stub
    assert actions.align(sk, "thing").reason == "round: any roll"
    assert actions.outline_height(sk, "thing") == pytest.approx(0.043)
    sk = _plan_sk(2.0, None, 0.0152)  # no profile: the old grasp height
    monkeypatch.setattr(actions, "brick_angle", lambda jpeg, box, look=None: None)
    p = actions.plan_grasp(sk, "thing").data
    assert p["close_m"] == prm.GRASP_LIFT_M.value and not p["round"]


def test_a_high_close_lands_there_and_refuses_any_contact_on_the_way(monkeypatch):
    moves = []
    sk = _height_sk({"thing": 0.067})
    sk.grasp_plan = {"obj": "thing", "round": True, "close_m": 0.027, "outline_m": 0.043}
    sk.height = lambda: 0.2
    sk.final_look_height = lambda obj: 0.087
    contact = {"v": False}
    sk.move = lambda part, **k: (moves.append(k.get("height")),
                                 Result(True, data={"contact": contact["v"], "contact_z": -0.06, "floor": -0.10}))[1]
    assert actions.land(sk, "thing").ok and moves == [0.087, 0.027]
    contact["v"] = True
    r = actions.land(sk, "thing")
    assert not r.ok and "over the planned close" in r.reason


def test_a_soft_round_object_squashed_in_the_jaws_is_not_a_miss(monkeypatch):
    # Nov 5: the 62.4 mm tennis ball (width from the side) closed at 56.5 mm, held.
    sk, _ = _width_sk(monkeypatch, 0.0624, 0.0624)
    sk.grasp_width = 0.0624
    sk.grasp_plan = {"obj": "red block", "round": True, "close_m": 0.034, "outline_m": 0.04, "width_m": 0.0624}
    sk.closed_mm = 56.5
    assert actions.close(sk).ok
    sk.closed_mm = 40.0  # far inside even the side camera's allowance: something else
    assert not actions.close(sk).ok


def test_the_release_opens_at_least_as_wide_as_the_grasp(monkeypatch):
    from brain.primitives import Result

    opened = []
    sk = _height_sk({})
    sk.grip = lambda how, width=None, **k: (opened.append(width), Result(True, data={}))[1]
    sk.held = lambda: False
    sk.grasp_open_mm = 70.4  # the tennis ball's grasp (Nov 5)
    assert actions.release(sk).ok and opened[-1] == 70.4
    sk.grasp_open_mm = 23.4  # a brick's: the demo's wider opening stands
    actions.release(sk)
    assert opened[-1] == prm.OPEN_WIDTH.value


def test_a_hover_that_sags_is_raised_by_the_difference():
    # Nov 5: holding the tennis ball the arm settled ~2 cm low and the ball pushed the cup.
    heights = []
    sk = _height_sk({})
    sk.move = lambda part, **k: (heights.append(k["height"]), Result(True))[1]
    sk.height = lambda: heights[-1] - 0.015 if len(heights) == 1 else heights[-1] - 0.015 + 0.015
    assert actions.hover_at(sk, 0.12).ok
    assert heights == [0.12, pytest.approx(0.135)]
    heights.clear()
    sk.height = lambda: heights[-1] - 0.002  # within HEIGHT_TOL_M: left alone
    actions.hover_at(sk, 0.12)
    assert heights == [0.12]


def test_a_hover_out_of_reach_straight_down_tilts_the_gripper_and_the_descent_keeps_it():
    # Nov 6: over the 11.5 cm shoe box the hover was 17.6 cm; straight down reaches 16 cm.
    tried = []

    def move(part, **k):
        tried.append(k.get("pitch"))
        ok = k["pitch"] <= 80.0
        return Result(ok, "" if ok else "can't move there", {} if ok else {"refused": True})

    sk = _height_sk({})
    sk.move = move
    sk.height = lambda: 0.176
    assert actions.hover_at(sk, 0.176).ok
    assert tried == [90.0, 80.0] and sk.hover_pitch == 80.0
    sk.heights, sk.held_below = {"box": 0.115}, 0.031
    actions.lower_to_release(sk, "box")
    assert tried[-1] == 80.0
    tried.clear()
    sk.move = lambda part, **k: (tried.append(k["pitch"]), Result(False, "can't move there", {"refused": True}))[1]
    assert not actions.hover_at(sk, 0.5).ok  # never past the shallowest tilt
    assert tried[-1] >= prm.MIN_HOVER_PITCH_DEG.value > tried[-1] - 10.0  # EE_ROT_STEP_DEG


def test_a_sag_raise_out_of_reach_tilts_too():
    # Nov 6: straight down reached 16.9 cm, the ball sagged it 1.75 cm, the raise was refused.
    tried = []

    def move(part, **k):
        tried.append((round(k["height"], 4), k["pitch"]))
        ok = k["height"] < 0.18 or k["pitch"] <= 80.0
        return Result(ok, "" if ok else "can't move there", {} if ok else {"refused": True})

    sk = _height_sk({})
    sk.move = move
    sk.height = lambda: tried[-1][0] - 0.0175 if len(tried) == 1 else tried[-1][0]
    assert actions.hover_at(sk, 0.1687).ok
    assert tried == [(0.1687, 90.0), (0.1862, 90.0), (0.1862, 80.0)] and sk.hover_pitch == 80.0


def test_a_hover_refused_by_the_robot_does_not_tilt():
    tried = []
    sk = _height_sk({})
    sk.move = lambda part, **k: (tried.append(k["pitch"]), Result(False, "the robot refused", {"refused": False}))[1]
    assert not actions.hover_at(sk, 0.12).ok and tried == [90.0]


def test_the_hover_over_a_wide_container_skips_the_view_while_the_held_object_hides_it(monkeypatch):
    sk = _height_sk({"shoe box": 0.115})
    sk.widths = {"shoe box": 0.27}
    sk.held_below = 0.031
    monkeypatch.setattr(actions, "held_blocks_view", lambda sk: True)
    clear = 0.115 + 0.031 + prm.PLACE_CLEARANCE_M.value + prm.LOOK_CLEARANCE_M.value
    assert actions.place_height(sk, "shoe box", "hover") == pytest.approx(clear)
    monkeypatch.setattr(actions, "held_blocks_view", lambda sk: False)
    monkeypatch.setattr(actions, "lowest_view_height", lambda geom, width, h, start: 0.26)
    assert actions.place_height(sk, "shoe box", "hover") > clear + 0.05


def test_a_held_object_wider_than_half_the_wrist_view_hides_the_container(monkeypatch):
    monkeypatch.setattr(actions, "view_half_width", lambda geom, h, plane_h: 0.0546)  # Nov 5, at the jaws
    sk = _height_sk({})
    sk.grasp_width = 0.063  # the tennis ball
    assert actions.held_blocks_view(sk)
    sk.grasp_width = 0.0158  # a brick
    assert not actions.held_blocks_view(sk)
    assert actions.place_standoff(sk) == pytest.approx(0.0079 + prm.PLACE_CLEARANCE_M.value)


def test_the_front_camera_aims_the_containers_middle_under_the_gripper():
    from brain.goals import Detection

    moves = []

    class G:
        cams = type("C", (), {"T_base_cam": np.eye(4)})()  # the camera at the arm's origin

        def front_point(self, uv):
            return np.array([0.30, 0.0, -0.1])  # the cup's near edge 30 cm straight ahead

    sk = _height_sk({"paper cup": 0.065})
    sk.geom, sk.widths = G(), {"paper cup": 0.094}
    sk.seen = {"front:paper cup": Detection(box=[0.4, 0.3, 0.6, 0.6])}
    sk.cmd_tcp = lambda: np.array([0.25, 0.0, 0.0])
    sk.move = lambda part, **k: (moves.append((part, k["delta"])), Result(True))[1]
    assert actions.over_by_front(sk, "paper cup").ok
    (part, (dx, dy)), = moves
    assert part == "base" and dx == pytest.approx(0.30 + 0.047 - 0.25) and dy == pytest.approx(0.0)


def test_the_approach_can_stop_short_of_its_target():
    from types import SimpleNamespace

    from brain import goals
    from brain.goals import Detection

    geom = SimpleNamespace(front_point=lambda uv: np.array([0.4, 0.0, -0.1]), front_turn_and_distance=lambda p: (0.0, 0.10))
    sk = SimpleNamespace(geom=geom)
    det = Detection(box=[0.4, 0.3, 0.6, 0.6])
    assert goals.FrontApproachGoal().error(det, sk)[1] == pytest.approx(0.10)
    assert goals.FrontApproachGoal(stop_short_m=0.04).error(det, sk)[1] == pytest.approx(0.06)


def test_a_large_held_object_is_placed_by_the_front_camera(monkeypatch):
    from brain import chains

    calls = []
    monkeypatch.setattr(chains.actions, "held_blocks_view", lambda sk: True)
    monkeypatch.setattr(chains.actions, "approach",
                        lambda sk, o, tol=None, stop_short=0.0, fast_ok=True: (calls.append(f"approach short {stop_short > 0}" + (" fast" if fast_ok else "")), chains.Result(True))[1])
    monkeypatch.setattr(chains.actions, "carry", lambda sk: (calls.append("carry"), chains.Result(True))[1])
    monkeypatch.setattr(chains.actions, "over_by_front", lambda sk, c: (calls.append("over by front"), chains.Result(True))[1])
    sk = _chain_sk(calls)
    sk.grasp_width = 0.063
    seq = iter([True, True, False])
    sk.held = lambda: next(seq)
    assert chains.place_in(sk, "tennis ball", "paper cup").ok
    assert calls == ["carry", "approach short True", "move", "over by front", "move", "open", "move"]


class _MarkedBoxWorld:
    """A box face with a marker, seen by the front camera: base turns and drives move it in the
    arm frame (the base turning about the arm's origin, for simplicity)."""

    def __init__(self, centre, heading_deg):
        a = np.radians(heading_deg)
        self.centre, self.normal = np.array(centre, float), np.array([-np.cos(a), -np.sin(a), 0.0])
        self.moves, self.heights, self.events, self.grasp_width = [], {}, [], 0.065
        self.marker_img = None

        class Cams:
            turn_centre = np.zeros(2)
            front = T_base_cam = None

        class G:
            floor_seen_z = -0.1
            cams = Cams

            @staticmethod
            def front_turn_and_distance(p):
                return float(np.degrees(np.arctan2(p[1], p[0]))), 0.0

        self.geom = G()

    def ev(self, name, **k):
        self.events.append((name, k))

    def move(self, part, delta=None, roll=None, **k):
        from brain.primitives import Result

        self.moves.append(("turn", round(roll, 1)) if roll is not None else ("drive", round(delta[0], 3)))
        if roll is not None:
            c, s = np.cos(np.radians(-roll)), np.sin(np.radians(-roll))
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.centre, self.normal = rot @ self.centre, rot @ self.normal
        else:
            self.centre = self.centre - [delta[0], delta[1], 0.0]
        return Result(True)

    def marker(self):
        from brain.markers import MarkerPose

        return MarkerPose(91, self.centre.copy(), self.normal.copy(), np.zeros((4, 2)), float(np.linalg.norm(self.centre)))


def test_a_marked_box_is_approached_by_its_marker_and_measured_by_its_face(monkeypatch):
    w = _MarkedBoxWorld([0.50, 0.25, -0.05], 20.0)
    monkeypatch.setattr(actions, "find_marker", lambda sk, c: w.marker())
    monkeypatch.setattr(actions, "face_top_z", lambda img, m, lens, t: 0.017)
    stop = 2 * actions.release_inside(w)
    assert actions.approach_marker(w, "shoe box", stop).ok
    p = w.centre - w.normal * actions.release_inside(w)
    assert abs(np.degrees(np.arctan2(p[1], p[0]))) <= prm.MIN_BASE_TURN_DEG.value
    assert np.hypot(p[0], p[1]) == pytest.approx(prm.REACH_POINT_X_M.value + stop, abs=prm.REACH_TOL_M.value)
    assert w.moves[0][0] == "turn" and w.heights["shoe box"] == pytest.approx(0.117)
    assert w.marked == {"shoe box": actions.release_inside(w)}


def test_a_marked_box_without_its_marker_in_view_says_so(monkeypatch):
    w = _MarkedBoxWorld([0.50, 0.0, -0.05], 0.0)
    monkeypatch.setattr(actions, "find_marker", lambda sk, c: None)
    monkeypatch.setattr(actions, "look_around", lambda sk, obj, fast_ok=True, look=None: chains.Result(False, "none"))
    r = actions.approach_marker(w, "shoe box", 0.0)
    assert not r.ok and r.data["no_marker"]


def test_a_place_in_a_marked_box_falls_back_to_the_box_models_only_without_its_marker(monkeypatch):
    calls = []
    monkeypatch.setattr(chains.actions, "held_blocks_view", lambda sk: True)
    monkeypatch.setattr(chains, "place_in_large", lambda sk, o, c, d: (calls.append("models"), chains.Result(True))[1])
    sk = _height_sk({})
    monkeypatch.setattr(chains, "place_in_marked", lambda sk, o, c: (calls.append("marker"), chains.Result(False, "x", {"no_marker": True}))[1])
    assert chains.place_in(sk, "tennis ball", "shoe box").ok and calls == ["marker", "models"]
    calls.clear()
    monkeypatch.setattr(chains, "place_in_marked", lambda sk, o, c: (calls.append("marker"), chains.Result(False, "lower: x"))[1])
    assert not chains.place_in(sk, "tennis ball", "shoe box").ok and calls == ["marker"]
    calls.clear()
    assert chains.place_in(sk, "tennis ball", "paper cup").ok and calls == ["models"]
