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

"""Layer 3, chains: what the planner's verbs expand to, as sequences of actions and
primitives. No numbers here (tests/test_geometry.py); a chain stops at the first step that fails and says
which, so the executor can retry or replan.

    pick "x":  approach -> hover -> line_up -> align -> servo(JawGoal) at the hover height ->
               down to the final look height -> servo(JawGoal) -> land -> grip(close) -> lift
    place "x" in "y":       carry (held at the hover height) -> approach y -> hover at the demo's place height -> over y (base,
               then arm) -> lower to the demo's release height -> release -> back up -> check it's inside
    place "x" next_to "y":  approach y -> hover -> beside y (base, then arm) -> down to the
               final look height -> beside y again -> land -> release -> back up -> check it's beside
"""

from __future__ import annotations

from brain import actions, params as prm
from brain.goals import BesideGoal, JawGoal
from brain.primitives import Result, Skills


def run(steps) -> Result:
    """Run (name, thunk) steps in order; the first failure ends the chain, named."""
    done = []
    for name, step in steps:
        r = step()
        if not r.ok:
            return Result(False, f"{name}: {r.reason}", {"done": done, "failed": name, **r.data})
        done.append(name)
    return Result(True, data={"done": done})


def pick(sk: Skills, obj: str, drive: bool = True, start: str | None = None) -> Result:
    """Pick `obj` from the floor. drive=False: the object is already near the reach point (the
    base stays put for the approach; the line-up still moves it). start: begin at this step (a
    retry resumes at "hover" rather than driving again)."""
    steps = [("approach", lambda: actions.approach(sk, obj))] if drive else []
    steps += [
        ("hover", lambda: actions.hover(sk)),
        ("line_up", lambda: actions.line_up(sk, obj)),
        ("align", lambda: actions.align(sk, obj)),
        # coarse at 10 cm (the line-up's tolerance), precise at 3 cm: the 10 cm stage spent 4-5
        # looks on precision the 3 cm stage redoes in 1-2 (Oct 28 speed check)
        ("centre_high", lambda: sk.servo(obj, JawGoal(tol_m=prm.REACH_TOL_M, plane_h=sk.object_height(obj)))),
        ("down", lambda: sk.move("arm", height=sk.final_look_height(obj))),
        ("centre_low", lambda: sk.servo(obj, JawGoal(plane_h=sk.object_height(obj)))),
        ("land", lambda: actions.land(sk, obj)),
        ("close", lambda: sk.grip("close")),
        ("lift", lambda: actions.lift(sk)),
    ]
    if start is not None:
        steps = steps[[name for name, _ in steps].index(start):]
    return run(steps)


def place_in(sk: Skills, obj: str, container: str, drive: bool = True) -> Result:
    """Put the held `obj` into `container` the way the owner's demo did: from above it, lowered
    to the demo's release height, then opened."""
    steps = [("holding", lambda: Result(sk.held(), "" if sk.held() else f"not holding the {obj}"))]
    steps += [("carry", lambda: actions.carry(sk)),
              ("approach", lambda: actions.approach(sk, container, prm.PLACE_APPROACH_TOL_M))] if drive else []
    steps += [
        ("hover", lambda: actions.hover_at(sk, prm.PLACE_HOVER_M)),
        ("over", lambda: actions.over_container(sk, container)),
        ("lower", lambda: actions.lower_to_release(sk)),
        ("release", lambda: actions.release(sk)),
        ("up", lambda: actions.hover_at(sk, prm.PLACE_HOVER_M)),
        ("check", lambda: actions.check_in(sk, obj, container)),
    ]
    return run(steps)


def place_next_to(sk: Skills, obj: str, ref: str, drive: bool = True) -> Result:
    """Set the held `obj` down on the floor beside `ref`: lined up from the hover height, checked
    again low, landed like a grasp, then opened."""
    steps = [("holding", lambda: Result(sk.held(), "" if sk.held() else f"not holding the {obj}"))]
    steps += [("carry", lambda: actions.carry(sk)), ("approach", lambda: actions.approach(sk, ref))] if drive else []
    steps += [
        ("hover", lambda: actions.hover_at(sk, prm.HOVER_HEIGHT_M)),
        ("beside_high", lambda: actions.beside(sk, ref)),
        ("down", lambda: sk.move("arm", height=prm.FINAL_LOOK_HEIGHT_M.value)),
        ("beside_low", lambda: sk.servo(ref, BesideGoal())),
        ("land", lambda: actions.land(sk)),
        ("release", lambda: actions.release(sk)),
        ("up", lambda: actions.hover_at(sk, prm.HOVER_HEIGHT_M)),
        ("check", lambda: actions.check_beside(sk, obj, ref)),
    ]
    return run(steps)
