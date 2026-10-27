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
"""

from __future__ import annotations

from v2 import actions, params as prm
from v2.goals import JawGoal
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
        ("centre_high", lambda: sk.servo(obj, JawGoal())),
        ("down", lambda: sk.move("arm", height=prm.FINAL_LOOK_HEIGHT_M.value)),
        ("centre_low", lambda: sk.servo(obj, JawGoal())),
        ("land", lambda: actions.land(sk)),
        ("close", lambda: sk.grip("close")),
        ("lift", lambda: actions.lift(sk)),
    ]
    return run(steps)
