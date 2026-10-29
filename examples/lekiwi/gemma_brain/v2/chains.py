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

"""v2 layer 3, chains: what the planner's verbs expand to, as sequences of actions and
primitives. No numbers here (test_v2); a chain stops at the first step that fails and says
which, so the executor can retry or replan.

    pick "x":  approach -> hover -> line_up -> align -> servo(JawGoal) at the hover height ->
               down to the final look height -> servo(JawGoal) -> land -> grip(close) -> lift
    place "x" in "y":       carry (arm folded, holding) -> approach y -> hover at the demo's place height -> over y (base,
               then arm) -> lower to the demo's release height -> release -> back up
    place "x" next_to "y":  approach y -> hover -> beside y (base, then arm) -> down to the
               final look height -> beside y again -> land -> release -> back up
"""

from __future__ import annotations

from v2 import actions, params as prm
from v2.goals import BesideGoal, JawGoal
from v2.skills import Result, Skills2


def run(steps) -> Result:
    """Run (name, thunk) steps in order; the first failure ends the chain, named."""
    done = []
    for name, step in steps:
        r = step()
        if not r.ok:
            return Result(False, f"{name}: {r.reason}", {"done": done, "failed": name, **r.data})
        done.append(name)
    return Result(True, data={"done": done})


def pick(sk: Skills2, obj: str, drive: bool = True) -> Result:
    """Pick `obj` from the floor. drive=False: the object is already near the reach point (the
    base stays put for the approach; the line-up still moves it)."""
    steps = [("approach", lambda: actions.approach(sk, obj))] if drive else []
    steps += [
        ("hover", lambda: actions.hover(sk)),
        ("line_up", lambda: actions.line_up(sk, obj)),
        ("align", lambda: actions.align(sk, obj)),
        # coarse at 10 cm (the line-up's tolerance), precise at 3 cm: the 10 cm stage spent 4-5
        # looks on precision the 3 cm stage redoes in 1-2 (Oct 28 speed check)
        ("centre_high", lambda: sk.servo(obj, JawGoal(tol_m=prm.REACH_TOL_M))),
        ("down", lambda: sk.move("arm", height=prm.FINAL_LOOK_HEIGHT_M.value)),
        ("centre_low", lambda: sk.servo(obj, JawGoal())),
        ("land", lambda: actions.land(sk)),
        ("close", lambda: sk.grip("close")),
        ("lift", lambda: actions.lift(sk)),
    ]
    return run(steps)


def place_in(sk: Skills2, obj: str, container: str, drive: bool = True) -> Result:
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
    ]
    return run(steps)


def place_next_to(sk: Skills2, obj: str, ref: str, drive: bool = True) -> Result:
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
    ]
    return run(steps)
