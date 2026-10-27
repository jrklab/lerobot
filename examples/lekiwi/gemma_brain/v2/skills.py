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

"""v2 layer 1, the primitives: the only code that drives motors and cameras (design doc
"v2 skills: three layers"). Every value comes from geometry.py (CALIB) or params.py (DEMO,
SPEC, RULE); the actions and chains above only compose these.

    move(part, ...)   arm: vertical straight-line move to a height over x, y, or by a delta,
                      optionally stopping at contact; base: a plain turn or a plain drive
    grip(open|close)  returns whether something is held
    find, servo       next increment

Arm moves keep the gripper vertical (pitch 90 in the arm model with the homing-tick joint
zero). The Arm planner refuses joint targets outside the recorded range and paths below the
floor clearance before anything moves, and the Pi enforces z_min on every tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from arm import FLOOR_CLEARANCE_M, JAW_TIP_OFFSET_M, Arm, PlanError, plan_ee_line

from v2 import params as prm
from v2.geometry import Geometry

VERTICAL = 90.0  # the arm model's pitch for a gripper pointing straight down


@dataclass
class Result:
    ok: bool
    reason: str = ""
    data: dict = field(default_factory=dict)


class Skills2:
    def __init__(self, robot, arm: Arm, geom: Geometry, log=None):
        self.robot, self.arm, self.geom, self.log = robot, arm, geom, log
        self.roll = 0.0
        # The Arm's whole-arm floor clearance from the measured floor (arm.DEFAULT_FLOOR_Z_M,
        # -0.100, predates the joint-zero fix; the measured floor is ~-0.126 at the jaw tips).
        self.arm.floor_z = float(geom.floor_coef[0]) - JAW_TIP_OFFSET_M
        self.arm.z_min = self.arm.floor_z + FLOOR_CLEARANCE_M + JAW_TIP_OFFSET_M

    def ev(self, name: str, **kw) -> None:
        if self.log is not None:
            self.log.ev(name, **kw)

    def tcp(self) -> np.ndarray:
        return self.arm.tcp()[:3, 3]

    def height(self) -> float:
        """Jaw tips above the floor under the gripper (vertical gripper: TCP above floor TCP)."""
        p = self.tcp()
        return float(p[2] - self.geom.floor(p[0], p[1]))

    # --- move ---

    def move(self, part: str = "arm", *, xy=None, height: float | None = None, delta=None,
             roll: float | None = None, until_contact: bool = False, speed: float | None = None) -> Result:
        """One plain move.

        arm: to `height` (jaw tips above the floor) over TCP `xy` (default: where it is), or by
        `delta` (dx, dy, dz in metres); `roll` turns the wrist; until_contact: the Pi stops the
        move where the TCP meets something (data["contact_z"]), searching at most
        LAND_SEARCH_M past the goal. Refused, with nothing sent, out of reach or too low.
        base: `delta` = (dx, dy) in metres or `roll` = a turn in degrees, never both: a
        combined turn and strafe left the body unturned (Oct 27).
        """
        if part == "base":
            return self._move_base(delta, roll)
        p = self.tcp()
        roll = self.roll if roll is None else roll
        if delta is not None:
            goal = p + np.asarray(delta, float)
        else:
            x, y = (p[0], p[1]) if xy is None else xy
            goal = np.array([x, y, self.geom.floor(x, y) + (p[2] - self.geom.floor(p[0], p[1]) if height is None else height)])
        floor = self.geom.floor(goal[0], goal[1])
        low = goal[2] - floor < FLOOR_CLEARANCE_M
        speed = speed or (prm.NEAR_SPEED_MPS.value if low else prm.FREE_SPEED_MPS.value)
        search = prm.LAND_SEARCH_M.value if until_contact else 0.0
        # Low moves get a guard just under them; the Pi checks the TCP, so the jaw tips' offset too.
        z_min = (min(goal[2], p[2]) - search - JAW_TIP_OFFSET_M - 0.002) if low else None
        try:
            if until_contact:
                wps, durs = plan_ee_line(self.arm.calib, self.arm.targets(), goal.tolist(), VERTICAL, roll, speed,
                                         z_min=goal[2] - search)
                res = self.robot.trajectory(wps, durs, z_min=z_min, contact_stop=True).get("result", {})
            else:
                self.arm.move_to(goal.tolist(), pitch_deg=VERTICAL, roll_deg=roll, speed=speed, z_min=z_min)
                res = {}
        except PlanError as e:
            return Result(False, f"can't move there: {e}", {"goal": goal.round(4).tolist()})
        self.roll = roll
        data = {"goal": goal.round(4).tolist(), "floor": round(floor, 4), "contact": False}
        if res.get("contact"):
            data.update(contact=True, contact_z=float(res["z_meas"]), detector=res.get("how"))
        self.ev("move", part="arm", **data)
        return Result(True, data=data)

    def _move_base(self, delta, turn) -> Result:
        if (delta is None) == (turn is None):
            return Result(False, "base: give a drive (dx, dy) or a turn, one at a time")
        if turn is not None:
            res = self.robot.base(dtheta_deg=float(turn))
        else:
            res = self.robot.base(dx=float(delta[0]), dy=float(delta[1]))
        moved = res.get("result", {}).get("moved")
        self.ev("move", part="base", delta=None if delta is None else list(delta), turn=turn, moved=moved)
        return Result(True, data={"moved": moved})

    # --- grip ---

    def grip(self, how: str, width: float | None = None) -> Result:
        """open: to `width` (default the demo's OPEN_WIDTH). close: until the jaws stall;
        data["held"]: they stopped short of closed-empty and the Pi's hold watch saw no drop."""
        if how == "open":
            res = self.robot.gripper(prm.OPEN_WIDTH.value if width is None else width).get("result", {})
            return Result(True, data=res)
        if how != "close":
            return Result(False, f"grip: 'open' or 'close', not {how!r}")
        res = self.robot.gripper(0, speed=30).get("result", {})
        held = self.held()
        self.ev("grip", how="close", held=held, **res)
        if not held:
            return Result(False, f"closed to {res.get('pos')} without holding anything", {**res, "held": False})
        return Result(True, data={**res, "held": True})

    def held(self) -> bool:
        st = self.robot.status()
        hold = st.get("gripper_hold") or {}
        return st["joints"]["arm_gripper"] >= prm.GRIPPER_EMPTY_MAX.value and hold.get("state") != "dropped"
