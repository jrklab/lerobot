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

"""v2 skills, first increment: move, land and grip. find and servo come next; pick and place
are chains of these (design doc "Step 5 design").

Every move keeps the gripper vertical (pitch 90 in the arm model with the homing-tick joint
zero) unless told otherwise; the Arm planner refuses joint targets outside the recorded range
and paths below the floor clearance, and the Pi enforces z_min on every tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from arm import FLOOR_CLEARANCE_M, JAW_TIP_OFFSET_M, Arm, PlanError, plan_ee_line

from v2 import params as prm
from v2.geometry import Geometry

VERTICAL = 90.0
# The Pi stops a contact_stop trajectory this far below its goal at the latest (its z_min),
# so a floor higher than predicted is met, not pushed through: the floor fit's worst error.
LAND_SEARCH_M = 0.008


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

    def move(self, xy, height: float, roll: float | None = None, speed: float | None = None) -> Result:
        """Straight-line move of the vertical gripper to `height` (jaw tips above the floor) over
        TCP x, y. Refused, with nothing sent, when the path would leave the joints' recorded
        range or come within the arm's floor clearance."""
        roll = self.roll if roll is None else roll
        z = self.geom.floor(xy[0], xy[1]) + height
        # Low moves (servo heights, under the free-move clearance) get a guard just under them.
        z_min = None if height >= FLOOR_CLEARANCE_M else min(z, self.tcp()[2]) - 0.003
        try:
            self.arm.move_to([xy[0], xy[1], z], pitch_deg=VERTICAL, roll_deg=roll,
                             speed=speed or prm.FREE_SPEED_MPS.value, z_min=z_min)
        except PlanError as e:
            return Result(False, f"can't move there: {e}", {"xy": list(xy), "height": height})
        self.roll = roll
        self.ev("move", xy=[round(float(v), 4) for v in xy], height=round(height, 4), roll=round(roll, 1))
        return Result(True)

    # --- land ---

    def land(self, lift: float | None = None) -> Result:
        """From above the target, go straight down to `lift` (default GRASP_LIFT_M) above the
        floor the fit predicts, without pressing into it: v1's touch searched for the floor by
        pushing into it and the gripper slid 8-10 mm forward (Oct 27). The Pi's contact stop
        stays on as the safety: met early, the result says how high, and a touch at floor
        level is added to the floor data."""
        lift = prm.GRASP_LIFT_M.value if lift is None else lift
        p = self.tcp()
        floor = self.geom.floor(p[0], p[1])
        goal = [float(p[0]), float(p[1]), floor + lift]
        z_min = goal[2] - LAND_SEARCH_M - JAW_TIP_OFFSET_M - 0.002
        try:
            wps, durs = plan_ee_line(self.arm.calib, self.arm.targets(), goal, VERTICAL, self.roll,
                                     prm.NEAR_SPEED_MPS.value, z_min=goal[2] - LAND_SEARCH_M)
        except PlanError as e:
            return Result(False, f"can't land here: {e}")
        res = self.robot.trajectory(wps, durs, z_min=z_min, contact_stop=True).get("result", {})
        if not res.get("contact"):
            self.ev("land", how="reached", floor=round(floor, 4), lift=lift)
            return Result(True, data={"contact": False, "floor": floor})
        z = float(res["z_meas"])
        above = z - floor
        self.ev("land", how="contact", z=round(z, 4), above_floor_mm=round(1000 * above, 1), detector=res.get("how"))
        if above > prm.BRICK_HEIGHT_M.value / 2:
            return Result(False, f"met something {1000 * above:.0f} mm above the floor", {"contact_z": z})
        # The floor, a little higher than predicted: keep the touch, come back up to `lift`.
        self.arm.move_to([goal[0], goal[1], z + lift], pitch_deg=VERTICAL, roll_deg=self.roll,
                         speed=prm.NEAR_SPEED_MPS.value, z_min=z - 0.002)
        return Result(True, data={"contact": True, "floor": z, "predicted": floor})

    # --- grip ---

    def open(self, width: float | None = None) -> Result:
        r = self.robot.gripper(prm.OPEN_WIDTH.value if width is None else width)
        return Result(True, data=r.get("result", {}))

    def close(self) -> Result:
        res = self.robot.gripper(0, speed=30).get("result", {})
        self.ev("close", **res)
        if not res.get("stalled"):
            return Result(False, f"closed fully ({res.get('pos')}) without stalling: nothing between the jaws", res)
        return Result(True, data=res)

    def held(self) -> bool:
        """Something is between the jaws: they stopped short of closed-empty and the Pi's hold
        watch saw no drop."""
        st = self.robot.status()
        hold = st.get("gripper_hold") or {}
        return st["joints"]["arm_gripper"] >= prm.GRIPPER_EMPTY_MAX.value and hold.get("state") != "dropped"

    def lift(self, by: float | None = None) -> Result:
        p = self.tcp()
        try:
            self.arm.move_to([p[0], p[1], p[2] + (prm.LIFT_M.value if by is None else by)], pitch_deg=VERTICAL,
                             roll_deg=self.roll, speed=prm.FREE_SPEED_MPS.value)
        except PlanError as e:
            return Result(False, f"can't lift: {e}")
        return Result(True, data={"held": self.held()})
