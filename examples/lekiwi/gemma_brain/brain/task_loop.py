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

"""The task loop: a spoken task in, the planner's steps run on the chains (design doc
"v2 skills: three layers").

    plan      the planner (26B) writes the whole plan once from one front and one wrist frame;
              the plan is grammar-limited and checked before anything moves
    run       each step on its chain:
                  pick "x"                pick chain, approach included (PICK_TRIES tries)
                  place "x" in "y"        place_in chain
                  place "x" next_to "y"   place_next_to chain
    recover   a failed step: fold, then the planner replans from what's done and why it
              failed (MAX_REPLANS); the same plan back twice ends it. A place whose jaws
              opened before a later step failed counts as placed (unverified): a replan would
              pick it up again.

The approach finds an object itself, so there's no separate scan. Between tries and before a
replan the arm folds and the tracked detections are dropped: the base may have moved. Moved
out of v1's plan_brain.PlanBrain on Oct 28 when v1 was retired (tag lekiwi-v1-final).
"""

from __future__ import annotations

import time

from common.arm import JAW_TIP_OFFSET_M, RESET_SAFE_Z_M, PlanError
from common.planner import Planner, Step, check_plan
from common.robot import OperatorTookOver, RobotError
from common.runlog import Log

from brain import actions, chains, params as prm
from brain.primitives import Result, Skills

# What the chains can run: (verb, relation).
SUPPORTED = {("pick", None), ("place", "in"), ("place", "next_to")}


class PlanBrain:
    def __init__(self, sk: Skills | None, planner: Planner, log: Log, robot=None, holding: str | None = None):
        self.sk, self.planner, self.log, self.robot = sk, planner, log, robot
        self.holding = holding
        if sk is not None:
            sk.holding = holding
        self.near_target = False  # the gripper is low beside a place target (see fold())

    def frames(self) -> tuple[bytes, bytes]:
        now = time.time()
        front, _ = self.robot.frame("front", after=now)
        wrist, _ = self.robot.frame("wrist", after=now)
        self.log.frame("plan_front", front)
        self.log.frame("plan_wrist", wrist)
        return front, wrist

    def make_plan(self, goal: str, front: bytes, wrist: bytes) -> tuple[list[Step] | None, str]:
        try:
            steps, a = self.planner.plan(goal, front, wrist, self.holding)
        except ValueError as e:
            self.log.ev("plan", ok=False, error=str(e))
            return None, f"the planner's answer was not a plan: {e}"
        self.log.ev("plan", steps=[str(s) for s in steps], ms=round(a.ms))
        print(f"plan ({a.ms / 1000:.1f} s):\n  " + "\n  ".join(map(str, steps)))
        problems = check_plan(steps, SUPPORTED, self.holding)
        if problems:
            self.log.ev("plan_rejected", problems=problems)
            return None, "plan not runnable: " + "; ".join(problems)
        return steps, ""

    # --- one step on the chains ---

    def _forget(self) -> None:
        self.sk.roll = 0.0
        self.sk.seen.clear()

    def _park(self) -> None:
        """The arm at the hover over the reach point, detections forgotten: the base may move.
        Folds instead where the hover is refused."""
        if actions.hover_at(self.sk, prm.HOVER_HEIGHT_M).ok:
            self._forget()
        else:
            self.fold()

    def pick(self, what: str) -> Result:
        """The pick chain, retried PICK_TRIES times where a retry can help. A refused move
        (the planner or the Pi turned it down) is refused again from the same pose: no retry,
        the planner hears why (Oct 28: a wrist roll past its limit failed six tries in two
        plans, ~140 s). A failure after the hover resumes at the hover where the base already
        is (the line-up finds the object again) instead of folding and driving again; a second
        one, or a failed approach, starts over from the approach with the arm kept at the hover
        (it folded and unfolded again, ~9 s: Oct 28 push test; at the hover it is out of the
        front camera's floor view, as when carrying)."""
        reasons = []
        tries = int(prm.PICK_TRIES.value)
        start = None
        for attempt in range(1, tries + 1):
            r = chains.pick(self.sk, what, start=start)
            if r.ok:
                self.sk.holding = what
                return r
            if self.sk.held():  # gripped, then a later step (the lift) failed: keep it
                self.sk.holding = what
                return Result(True, f"holding the {what} (after: {r.reason})")
            reasons.append(r.reason)
            if r.data.get("refused"):
                self.log.ev("pick_refused", attempt=attempt, reason=r.reason)
                print(f"    pick attempt {attempt} refused, no retry: {r.reason}")
                return Result(False, f"refused ({r.reason})", r.data)
            after_hover = r.data.get("failed") not in (None, "approach", "hover")
            start = "hover" if after_hover and start != "hover" else None
            self.log.ev("pick_retry", attempt=attempt, reason=r.reason, resume=start or "approach")
            print(f"    pick attempt {attempt} failed: {r.reason}")
            if start is None:
                self._park()
        return Result(False, f"failed {tries} times ({'; '.join(reasons)})")

    def place(self, rel: str, target: str) -> Result:
        what = self.sk.holding or "object"
        chain = {"in": chains.place_in, "next_to": chains.place_next_to}.get(rel)
        if chain is None:  # check_plan stops these before anything moves; never run another chain
            return Result(False, f"no chain for 'place {rel}'")
        r = chain(self.sk, what, target)
        if not self.sk.held():  # released, whether or not the steps after it went well
            self.sk.holding = None
        return r

    def run_step(self, s: Step) -> Result:
        if s.verb == "pick":
            return self.pick(s.obj)
        return self.place(s.rel, s.target)

    def fold(self) -> str:
        """Fold the arm to rest, via the hover over the reach point: from there a held brick
        clears a cup's rim (5.2 cm measured). Where the hover is refused, straight up to
        RESET_SAFE_Z_M first; when that is refused too right after a place (near_target) and
        the jaws are below FOLD_CLEAR_M, the arm stays where it is: Arm.reset's fallback tilts
        the gripper and pulls it in while still low, and on Oct 26 that dragged the cup 10 cm
        toward the robot. Returns why it stayed, or "" when folded."""
        sk, arm = self.sk, self.sk.arm
        try:
            if not actions.hover_at(sk, prm.HOVER_HEIGHT_M).ok:
                p = arm.tcp()[:3, 3]
                if p[2] < RESET_SAFE_Z_M:
                    try:
                        arm.move_to([p[0], p[1], RESET_SAFE_Z_M], speed=prm.FREE_SPEED_MPS.value,
                                    z_min=min(arm.z_min, p[2] - JAW_TIP_OFFSET_M - prm.LOW_MOVE_GUARD_M.value))
                    except PlanError as e:
                        if self.near_target and sk.height() < prm.FOLD_CLEAR_M.value:
                            self.log.ev("fold", ok=False, reason=str(e), height=round(sk.height(), 3))
                            return f"can't lift straight up from here ({e})"
            arm.reset()
        except (RobotError, PlanError) as e:
            if isinstance(e, OperatorTookOver):
                raise
            self.log.ev("fold", ok=False, reason=str(e))
            return str(e)
        finally:
            self._forget()
        return ""

    # --- the task ---

    def run(self, goal: str) -> tuple[bool, str]:
        self.log.ev("goal", goal=goal, planner=self.planner.model.model, holding=self.holding)
        self.sk.holding = self.holding
        front, wrist = self.frames()
        steps, why = self.make_plan(goal, front, wrist)
        if steps is None:
            return False, why
        done: list[Step] = []
        unverified: list[str] = []
        replans = repeats = 0
        while steps:
            s = steps[0]
            print(f"\n>>> {s}")
            held_before = self.holding
            try:
                r = self.run_step(s)
            except (RobotError, PlanError) as e:
                if isinstance(e, OperatorTookOver):
                    raise
                if "not in auto mode" in str(e):
                    # The Pi ends auto mode on gamepad input, the e-stop, or the gamepad
                    # disconnecting (it powers off when idle: Oct 26). Nothing can move now.
                    raise OperatorTookOver("robot left auto mode (gamepad input, e-stop, or the gamepad "
                                           "disconnecting, e.g. asleep after idling)") from e
                r = Result(False, f"robot error: {e}")
            self.holding = self.sk.holding
            self.near_target = s.verb == "place"
            self.log.ev("step_result", step=str(s), ok=r.ok, reason=r.reason, holding=self.holding)
            print(f"<<< {'OK' if r.ok else 'FAILED'}: {r.reason or 'done'}")
            checked_wrong = r.data.get("failed") == "check"  # released, and a look found it misplaced
            if not r.ok and s.verb == "place" and held_before == s.obj and self.holding is None and not checked_wrong:
                # The gripper opened over the target, then something after it failed (Oct 26:
                # the retreat from the cup). Placed, not verified: replanning would pick it again.
                self.log.ev("place_unverified", step=str(s), reason=r.reason)
                print(f"    released over the {s.target}: counting it as placed, unverified")
                unverified.append(str(s))
                r = Result(True)
                why = self.fold()
                if why:
                    return False, f"placed ({s}, unverified), then the arm stayed where it was: {why}"
            if r.ok:
                done.append(s)
                steps = steps[1:]
                continue
            max_replans = int(prm.MAX_REPLANS.value)
            if replans >= max_replans:
                return False, f"{s} failed after {max_replans} replans: {r.reason}"
            replans += 1
            why = self.fold()
            if why:
                return False, f"{s} failed ({r.reason}); the arm stayed where it was to replan: {why}"
            front, wrist = self.frames()
            try:
                new, a = self.planner.replan(goal, done, s, r.reason, self.holding, front, wrist)
            except ValueError as e:
                return False, f"{s} failed ({r.reason}) and the replan was not a plan: {e}"
            # The same steps again: one more try is a retry; a second time won't go differently.
            same = new == steps
            if same and repeats:
                return False, f"{s} failed twice ({r.reason}) and the replan didn't change it"
            repeats += same
            steps = new
            self.log.ev("replan", n=replans, steps=[str(x) for x in steps], ms=round(a.ms))
            print(f"replan {replans} ({a.ms / 1000:.1f} s):\n  " + "\n  ".join(map(str, steps)))
            problems = check_plan(steps, SUPPORTED, self.holding)
            if problems:
                self.log.ev("plan_rejected", problems=problems)
                return False, f"{s} failed ({r.reason}); replan not runnable: " + "; ".join(problems)
        summary = "done: " + "; ".join(map(str, done))
        return True, summary + (f" (unverified: {'; '.join(unverified)})" if unverified else "")
