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

"""Measure the floor height in the arm base frame by touching it, gripper closed and
pointing straight down.

    .venv/bin/python calibrate_floor.py [--x 0.22 0.28] [--write]

The TCP is lowered in small steps. After each step, the measured TCP height (forward
kinematics of the measured joints) is compared with the commanded one. In free air the arm
sags, so the measured TCP sits a little *below* the command; once the jaws touch, the floor
holds the TCP up while the command keeps going down, so measured minus commanded rises.
Contact is when it rises more than CONTACT_GAP_M above its free-air value (the median of the
first BASELINE_STEPS steps, fixed: the floor lifts the TCP gradually, so a running baseline
would follow it down); the floor TCP height is the measured z at the last free step. The Pi's own tracking/stall abort stays
armed as the backstop.

--write stores the result in calib.json (value, method, date, per-probe numbers).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from pathlib import Path

import numpy as np

from arm import Arm, PlanError
from robot import Robot, RobotError

STEP_M = 0.003
CONTACT_GAP_M = 0.0015
BASELINE_STEPS = 3
FAST_TO_Z_M = -0.08  # descend in one move to here (the floor is ~-0.113 for the TCP), then step
START_Z_M = 0.03  # TCP height the descent starts from
NEAR_X_M = 0.22  # a start point reachable straight down from "ready"
LOWEST_Z_M = -0.20  # never command below this, contact or not
CALIB = Path(__file__).with_name("calib.json")


def gap(arm: Arm) -> tuple[float, float, float]:
    """(commanded TCP z, measured TCP z, measured - commanded): negative in free air (sag),
    rising once something holds the TCP up."""
    time.sleep(0.35)  # let the servos settle on the last target
    st = arm.robot.status()
    z_cmd = float(arm.calib.fk(st["targets"])[2, 3])
    z_meas = float(arm.calib.fk(st["joints"])[2, 3])
    return z_cmd, z_meas, z_meas - z_cmd


def probe(arm: Arm, x: float) -> dict:
    print(f"\n--- probe at x={x:.3f} m: above the floor, pointing down")
    try:
        arm.move_to([x, 0.0, START_Z_M], pitch_deg=90, roll_deg=0, speed=0.03, z_min=LOWEST_Z_M)
    except PlanError:
        # The direct path out of "ready" can swing the wrist through its limit: go to the
        # nearest start point first, then slide out at the start height.
        arm.move_to([NEAR_X_M, 0.0, START_Z_M], pitch_deg=90, roll_deg=0, speed=0.03, z_min=LOWEST_Z_M)
        arm.move_to([x, 0.0, START_Z_M], pitch_deg=90, roll_deg=0, speed=0.02, z_min=LOWEST_Z_M)
    arm.move_to([x, 0.0, FAST_TO_Z_M], pitch_deg=90, roll_deg=0, speed=0.015, z_min=LOWEST_Z_M)
    _, z0, g0 = gap(arm)
    print(f"start: measured z {z0:.4f}, measured - commanded {1000 * g0:+.1f} mm")
    free_gaps = [g0]
    z = FAST_TO_Z_M
    last_free = z0
    contact = False
    while z > LOWEST_Z_M + STEP_M:
        z -= STEP_M
        try:
            arm.move_to([x, 0.0, z], pitch_deg=90, roll_deg=0, speed=0.01, z_min=LOWEST_Z_M)
        except RobotError as e:
            # The Pi's backstop fired (stall / tracking error): that's contact too.
            print(f"Pi stopped the step at z {z:.4f}: {e}")
            break
        z_cmd, z_meas, g = gap(arm)
        base = float(np.median(free_gaps[: BASELINE_STEPS + 1]))
        print(f"cmd z {z_cmd:+.4f}  measured {z_meas:+.4f}  meas-cmd {1000 * g:+5.1f} mm (free air {1000 * base:+.1f})")
        if len(free_gaps) > BASELINE_STEPS and g - base > CONTACT_GAP_M:
            print(f"contact: the floor holds the TCP {1000 * (g - base):.1f} mm above its free-air sag")
            contact = True
            break
        free_gaps.append(g)
        last_free = z_meas
    if not contact:
        raise RuntimeError("reached the lowest allowed z without touching anything")
    # The step before contact was free: the jaws touched between it and the contact step.
    floor = last_free
    # Back off: up 4 cm.
    st = arm.robot.status()
    p = arm.calib.fk(st["joints"])[:3, 3]
    arm.move_to([p[0], p[1], p[2] + 0.04], pitch_deg=90, roll_deg=0, speed=0.02, z_min=LOWEST_Z_M)
    print(f"floor TCP z at x={x:.3f}: {floor:.4f} m")
    return {"x": x, "floor_tcp_z": round(floor, 4), "free_air_sag_mm": round(-1000 * float(np.median(free_gaps)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=float, nargs="+", default=[0.22, 0.28])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    robot = Robot()
    probes = []
    with robot:
        arm = Arm(robot)
        try:
            arm.pose("ready", speed=20)
            for x in args.x:
                probes.append(probe(arm, x))
        except (RobotError, PlanError, RuntimeError) as e:
            print(f"\nstopped: {e}")
        finally:
            try:
                arm.reset()
            except (RobotError, PlanError) as e:
                print(f"reset failed: {e} -- the arm is holding where it is")

    if not probes:
        return
    zs = [p["floor_tcp_z"] for p in probes]
    print("probes:", probes)
    print(f"\nfloor TCP z: {zs} -> mean {np.mean(zs):.4f} m, spread {1000 * (max(zs) - min(zs)):.1f} mm")
    if args.write:
        calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
        calib["floor_tcp_z_m"] = {
            "value": round(float(np.mean(zs)), 4),
            "unit": "m",
            "frame": "arm base (so101_kinematics), TCP pointing straight down, gripper closed",
            "method": "calibrate_floor.py: touch the floor in 4 mm steps; contact = measured-vs-commanded gap growth",
            "date": datetime.date.today().isoformat(),
            "probes": probes,
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")


if __name__ == "__main__":
    main()
