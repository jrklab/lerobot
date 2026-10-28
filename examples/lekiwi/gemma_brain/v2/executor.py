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

"""Run the planner's plans on the v2 chains (design doc "v2 skills: three layers"). The planner,
its plan checks, the replanning and the rest of the task loop are v1's PlanBrain (plan_brain.py);
only the steps run differently:

    pick "x"                pick chain, approach included (retried PICK_TRIES times)
    place "x" in "y"        place_in chain (approach to y, located from two heights, lowered)
    place "x" next_to "y"   place_next_to chain

v2's approach finds an object itself, so there's no separate scan. Between tries and before a
replan the arm folds (v1's fold, which lifts straight up first when low) and the tracked
detections are dropped: the base may have moved.
"""

from __future__ import annotations

from arm import PlanError
from brain import PICK_TRIES
from plan_brain import PlanBrain
from robot import OperatorTookOver, RobotError

from v2 import actions, chains, params as prm
from v2.skills import Result, Skills2


class PlanBrainV2(PlanBrain):
    def __init__(self, sk: Skills2, planner, log, robot=None, holding: str | None = None):
        super().__init__(sk, planner, log, robot, holding)
        sk.holding = holding  # PlanBrain.run reads and writes it

    def ensure_scanned(self, objects: list[str]) -> Result:
        return Result(True)  # the approach looks for the object itself

    def _settle(self) -> None:
        self.sk.roll = 0.0
        self.sk.seen.clear()

    def pick(self, what: str) -> Result:
        reasons = []
        for attempt in range(1, PICK_TRIES + 1):
            r = chains.pick(self.sk, what)
            if r.ok:
                self.sk.holding = what
                return r
            if self.sk.held():  # gripped, then a later step (the lift) failed: keep it
                self.sk.holding = what
                return Result(True, f"holding the {what} (after: {r.reason})")
            reasons.append(r.reason)
            self.log.ev("pick_retry", attempt=attempt, reason=r.reason)
            print(f"    pick attempt {attempt} failed: {r.reason}")
            self.fold()
            self._settle()
        return Result(False, f"failed {PICK_TRIES} times ({'; '.join(reasons)})")

    def place(self, rel: str, target: str) -> Result:
        what = self.sk.holding or "object"
        chain = chains.place_in if rel == "in" else chains.place_next_to
        r = chain(self.sk, what, target)
        if not self.sk.held():  # released, whether or not the steps after it went well
            self.sk.holding = None
        return r

    def fold(self) -> str:
        """Over the reach point at the hover height first, then fold. v1's fold lifts straight
        up to RESET_SAFE_Z_M (0.10 m in the arm frame, ~21 cm over the floor), which the
        vertical gripper can't reach at 25 cm out, so it stayed put after a failed place (Oct
        28). At the hover a held brick clears a cup's rim (5.2 cm measured)."""
        try:
            r = actions.hover_at(self.sk, prm.HOVER_HEIGHT_M)
            if r.ok:
                self.sk.arm.reset()
                why = ""
            else:
                why = super().fold()
        except (RobotError, PlanError) as e:
            if isinstance(e, OperatorTookOver):
                raise
            why = str(e)
        self._settle()
        return why
