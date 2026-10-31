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

# ruff: noqa: E402

"""The arm calibrates itself: no one moves it by hand. The coarse calibration, from the URDF
and the joints' own ends; the camera calibration (solve_cameras.py --fit-offsets) then refines
the zeros it can see.

    .venv/bin/python calibration/self_calibrate.py bootstrap          # the arm at rest
    .venv/bin/python calibration/self_calibrate.py probe gripper      # one joint per run
    .venv/bin/python calibration/self_calibrate.py probe wrist_roll ... shoulder_lift
    .venv/bin/python calibration/self_calibrate.py write [--dry-run]  # zeros and ranges, saved
    .venv/bin/python calibration/self_calibrate.py restore            # the calibration from before

The Pi's web service must run with --allow-uncalibrated (it starts on a provisional calibration
when the motors don't match a file) and writes each calibration itself (POST /auto/calibration):
no service restart, the gamepad's e-stop live throughout.

bootstrap   from the rest pose, placed by eye (the SO-101's folded pose, common/arm.py
            NAMED_POSES["rest"]): each motor's homing offset is set so that the pose reads its
            rest angle, and each range to the whole turn. The rest angles are within a few
            degrees of the half turn's middle, so every joint's travel lies away from the
            encoder's 4095 -> 0 wrap; a probe that crosses it fails rather than read a jump.
probe       one joint, one end at a time, at a reduced torque limit (the Pi's "probe"
            command, auto_control.py): drive slowly until the joint stops following, back off.
            Each end is probed from a pose (probe_pose) where the whole sweep, a margin past the
            URDF's limit, stays clear of the floor and the robot body in the arm model. An end
            found well short of the URDF's limit is probed again at twice the torque, once: the
            joint may have stalled under its own weight; a short end that stays is kept (a
            cable, as on the wrist roll).
write       zeros: shoulder lift, elbow and wrist flex from their two ends, the ends' midpoint
            taken as the URDF limits' midpoint; shoulder pan and wrist roll keep the rest
            placement (their ends aren't symmetric about the URDF's zero, and the cameras
            can't see their zeros). Ranges: the ends less the probe's back-off (the URDF clips
            what a command may reach, joint_angles.limits). The gripper's range: closed end to
            open end. Then saved to the robot's calibration file, the old one kept in
            data/self_calibration.json for restore.

Order (one joint moving at a time): gripper, wrist roll, shoulder pan (the arm folded: nothing
it can reach), then wrist flex, elbow, shoulder lift (the joints carrying weight, each from a
pose chosen on the URDF model).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from common.arm import (
    BODY_FRONT_X_M,
    BODY_TOP_Z_M,
    DEFAULT_FLOOR_Z_M,
    IK_ARM_JOINTS,
    NAMED_POSES,
    Arm,
    PlanError,
    arm_points,
)
from common.robot import Robot, RobotError
from joint_angles import GRIPPER, URDF_LIMITS_DEG

STATE = ROOT / "data" / "self_calibration.json"
RESOLUTION = 4096
HOMING_TICK = RESOLUTION // 2 - 1  # 2047
HOMING_OFFSET_MAX = RESOLUTION // 2 - 1  # the register's sign-magnitude limit
TICKS_PER_DEG = RESOLUTION / 360.0

ORDER = [
    "arm_gripper",
    "arm_wrist_roll",
    "arm_shoulder_pan",
    "arm_wrist_flex",
    "arm_elbow_flex",
    "arm_shoulder_lift",
]
# Zeros from the two ends (the others keep the rest placement: see the module docstring).
ZERO_FROM_ENDS = ("arm_shoulder_lift", "arm_elbow_flex", "arm_wrist_flex")
REST = {**NAMED_POSES["rest"], GRIPPER: 0.0}  # the gripper rests closed
# The gripper closes towards "-": on the SO-101 its motor's ticks fall as it closes (its degrees
# count up from closed, joint_angles.py, with drive_mode 0).

REST_TOLERANCE_DEG = 5.0  # the owner's: the rest pose is placed by eye to within this
PAST_URDF_DEG = 15.0  # how far past the URDF's limit a sweep is checked clear: the ends found
#                       on this arm lay up to 11 deg past it (Oct 29), plus REST_TOLERANCE_DEG
SHORT_DEG = 2 * REST_TOLERANCE_DEG  # an end further inside the URDF's limit than this is suspect
SWEEP_CLEARANCE_M = 0.05  # convention: floor clearance of a probe's sweep, for a joint sagging
#                           at a low torque limit and the rest placement's error
GRIPPER_TRAVEL_DEG = 200.0  # convention: more than a gripper opens
MOVE_SPEED = 20.0  # deg/s between poses, as the other calibration scripts
PROBE_SPEED = 10.0  # deg/s
PROBE_TORQUE = 0.3  # of the motor's maximum
PROBE_BACKOFF_DEG = 3.0  # web_server/auto_control.py: where a probe parks after an end
POSE_GRID_DEG = 5.0  # convention: the step of the pose search (15 misses the shoulder lift's forward end)


# --- the provisional calibration ---


def _wrap_offset(ticks: int) -> int:
    """An offset matters modulo a turn; the register holds +/-2047."""
    ticks = (ticks + RESOLUTION // 2) % RESOLUTION - RESOLUTION // 2
    return max(-HOMING_OFFSET_MAX, min(HOMING_OFFSET_MAX, ticks))


def present_ticks(joint: str, c: dict, deg: float) -> float:
    """The motor's Present_Position for a reading in degrees (joint_angles.JointAngles)."""
    if c.get("drive_mode"):
        raise ValueError(f"{joint}: a reversed drive_mode isn't supported")
    if joint == GRIPPER:
        return c["range_min"] + deg * TICKS_PER_DEG
    return HOMING_TICK + deg * TICKS_PER_DEG


def provisional_calibration(
    calib: dict[str, dict], measured: dict[str, float], rest: dict[str, float] = REST
) -> dict:
    """Homing offsets for each joint, at rest, to read its rest angle (the gripper: the half
    turn, room to both sides), and whole-turn ranges. Feetech: Present = Actual - Homing_Offset
    (mod a turn)."""
    out = {}
    for joint, c in calib.items():
        actual = round(present_ticks(joint, c, measured[joint])) + c["homing_offset"]
        want = HOMING_TICK if joint == GRIPPER else HOMING_TICK + rest[joint] * TICKS_PER_DEG
        out[joint] = {
            "homing_offset": _wrap_offset(round(actual - want)),
            "range_min": 0,
            "range_max": RESOLUTION - 1,
        }
    return out


# --- where each end is probed from ---


def _sweep_clear(pose: dict[str, float], joint: str, end: float, floor_z: float) -> float | None:
    """The sweep's lowest clearance above the floor, None when it passes low over the body."""
    lowest = math.inf
    for a in np.linspace(pose[joint], end, 40):
        pts = arm_points(np.radians([{**pose, joint: a}[j] for j in IK_ARM_JOINTS]))[2:]
        if np.any((pts[:, 0] < BODY_FRONT_X_M) & (pts[:, 2] < BODY_TOP_Z_M)):
            return None
        lowest = min(lowest, float(pts[:, 2].min()) - floor_z)
    return lowest


def sweep_end(joint: str, direction: int) -> float:
    lo, hi = URDF_LIMITS_DEG[joint]
    return hi + PAST_URDF_DEG if direction > 0 else lo - PAST_URDF_DEG


def probe_pose(joint: str, direction: int, floor_z: float = DEFAULT_FLOOR_Z_M) -> dict[str, float]:
    """The pose (degrees) to probe this end from: rest when the sweep from it is clear, else
    the clear pose on a POSE_GRID_DEG grid of the other weight-bearing joints nearest rest.
    The gripper's ends are probed at rest."""
    rest = {j: REST[j] for j in IK_ARM_JOINTS}
    if joint == GRIPPER:
        return rest
    end = sweep_end(joint, direction)
    clear = _sweep_clear(rest, joint, end, floor_z)
    if clear is not None and clear >= SWEEP_CLEARANCE_M:
        return rest
    others = [j for j in ("arm_shoulder_lift", "arm_elbow_flex", "arm_wrist_flex") if j != joint]
    best = None
    grid = {
        j: np.arange(
            math.ceil(URDF_LIMITS_DEG[j][0] / POSE_GRID_DEG),
            math.floor(URDF_LIMITS_DEG[j][1] / POSE_GRID_DEG) + 1,
        )
        * POSE_GRID_DEG
        for j in others
    }
    for a in grid[others[0]]:
        for b in grid[others[1]]:
            pose = {**rest, others[0]: float(a), others[1]: float(b)}
            clear = _sweep_clear(pose, joint, end, floor_z)
            if clear is None or clear < SWEEP_CLEARANCE_M:
                continue
            dist = max(abs(pose[j] - rest[j]) for j in others)
            if best is None or dist < best[0]:
                best = (dist, pose)
    if best is None:
        raise PlanError(
            f"no pose to probe {joint}'s {'+' if direction > 0 else '-'} end from clear of the floor"
        )
    return best[1]


# --- the final calibration ---


def short_of_urdf(joint: str, direction: int, stop_deg: float) -> bool:
    if joint == GRIPPER:
        return False
    lo, hi = URDF_LIMITS_DEG[joint]
    return stop_deg < hi - SHORT_DEG if direction > 0 else stop_deg > lo + SHORT_DEG


def final_calibration(calib: dict[str, dict], ends: dict[str, dict[str, float]]) -> tuple[dict, dict]:
    """From the provisional calibration and each joint's ends ({joint: {"-": deg, "+": deg}},
    as read in it): the motors' new offsets and ranges, and the zero corrections made (deg)."""
    zeros = {}
    for joint in ZERO_FROM_ENDS:
        if joint in ends:
            lo, hi = URDF_LIMITS_DEG[joint]
            zeros[joint] = (lo + hi) / 2 - (ends[joint]["-"] + ends[joint]["+"]) / 2
    out = {}
    for joint, e in ends.items():
        c = out[joint] = dict(calib[joint])
        d = zeros.get(joint, 0.0)
        # Read d degrees higher: Present = Actual - Homing_Offset (joint_angles.shifted_calibration,
        # which also shifts the range: here the range is new).
        c["homing_offset"] = _wrap_offset(c["homing_offset"] - round(d * TICKS_PER_DEG))
        if joint == GRIPPER:
            # Closed exactly at its end, as a calibration by hand records it: a gripper short of
            # closing can't hold anything thin.
            ticks = (
                present_ticks(joint, calib[joint], e["-"]),
                present_ticks(joint, calib[joint], e["+"] - PROBE_BACKOFF_DEG),
            )
        else:
            ticks = (
                HOMING_TICK + (e["-"] + d + PROBE_BACKOFF_DEG) * TICKS_PER_DEG,
                HOMING_TICK + (e["+"] + d - PROBE_BACKOFF_DEG) * TICKS_PER_DEG,
            )
        c["range_min"], c["range_max"] = round(ticks[0]), round(ticks[1])
        if not 0 <= c["range_min"] < c["range_max"] <= RESOLUTION - 1:
            raise ValueError(f"{joint}: range {c['range_min']}..{c['range_max']} outside the turn")
    keys = ("homing_offset", "range_min", "range_max")
    return {j: {k: int(c[k]) for k in keys} for j, c in out.items()}, zeros


# --- on the robot ---


def _load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(state: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def _at(robot: Robot) -> dict[str, float]:
    return robot.status()["joints"]


def bootstrap(robot: Robot) -> dict:
    """The arm is at rest. Keeps the calibration the motors had (for restore), then writes the
    provisional one to the motors only."""
    cal = robot.calibration()
    state = _load()
    if "before" not in state:
        state["before"] = {
            "joints": cal["joints"],
            "state": cal.get("state"),
            "date": datetime.datetime.now().isoformat(),
        }
    new = provisional_calibration(cal["joints"], _at(robot))
    reply = robot.write_calibration(new, save=False)
    state.update(provisional=new, ends={}, bootstrapped=datetime.datetime.now().isoformat())
    _save(state)
    print("provisional calibration written (motors only); the arm now reads:")
    for j, v in _at(robot).items():
        print(f"  {j:18s} {v:8.2f} deg (rest {REST[j]:.2f})")
    return reply


def _move(arm: Arm, pose: dict[str, float], what: str) -> None:
    cur = arm.targets()
    goal = {**cur, **{j: pose[j] for j in IK_ARM_JOINTS}}
    arm.check_clear(cur, goal, what)
    arm.robot.joints(goal, speed=MOVE_SPEED)


def probe_joint(robot: Robot, joint: str) -> dict[str, float]:
    """Both ends of one joint, each from its probe_pose, back to rest after."""
    if robot.calibration().get("state") != "provisional":
        raise RobotError("bootstrap first: the arm isn't on a provisional calibration")
    arm = Arm(robot)
    found = {}
    try:
        for d in (-1, +1):
            _move(arm, probe_pose(joint, d, arm.floor_z), f"to probe {joint}")
            start = _at(robot)[joint]
            travel = GRIPPER_TRAVEL_DEG if joint == GRIPPER else abs(sweep_end(joint, d) - start)
            torque = PROBE_TORQUE / 2 if joint == "arm_wrist_roll" else PROBE_TORQUE
            for attempt in range(2):
                r = robot.run(
                    {
                        "cmd": "probe",
                        "joint": joint,
                        "direction": d,
                        "speed": PROBE_SPEED,
                        "torque": torque,
                        "max_travel": min(travel, 360.0),
                        "units": "deg",
                    },
                    timeout=60.0,
                )
                stop = r["result"]["stop_deg"]
                print(
                    f"{joint} {'+' if d > 0 else '-'}: end at {stop:.1f} deg (torque {torque:.2f}, "
                    f"load {r['result']['load']:.0f})"
                )
                if attempt or not short_of_urdf(joint, d, stop):
                    break
                torque = min(2 * torque, 0.6)
                print(f"  {SHORT_DEG:.0f}+ deg short of the URDF's limit: again at torque {torque:.2f}")
            found["+" if d > 0 else "-"] = stop
    finally:
        try:
            _move(arm, REST, "back to rest")
        except (RobotError, PlanError) as e:
            print(f"back to rest failed: {e} -- the arm is holding where it is")
    state = _load()
    state.setdefault("ends", {})[joint] = found
    _save(state)
    return found


def write(robot: Robot, dry_run: bool) -> None:
    state = _load()
    missing = [j for j in ORDER if j not in state.get("ends", {})]
    if missing:
        raise SystemExit(f"probe these first: {', '.join(missing)}")
    cal = robot.calibration()["joints"]
    new, zeros = final_calibration(cal, state["ends"])
    for j in ORDER:
        print(
            f"  {j:18s} ends {state['ends'][j]['-']:7.1f} {state['ends'][j]['+']:7.1f}  zero {zeros.get(j, 0.0):+5.1f} deg"
            f"  -> offset {new[j]['homing_offset']}, range {new[j]['range_min']}..{new[j]['range_max']}"
        )
    if dry_run:
        print("(dry run)")
        return
    _move(Arm(robot), REST, "to rest for the write")
    robot.write_calibration(new, save=True)
    state.update(final=new, zeros=zeros, written=datetime.datetime.now().isoformat())
    _save(state)
    print("saved. At rest the arm now reads:")
    for j, v in _at(robot).items():
        print(f"  {j:18s} {v:8.2f} deg (rest {REST[j]:.2f})")
    print(
        "next: the floor touches (calibrate_floor.py --write) and the camera calibration "
        "(capture_cameras.py scene, solve_cameras.py --fit-offsets --write)"
    )


def restore(robot: Robot) -> None:
    before = _load().get("before")
    if not before:
        raise SystemExit(f"nothing to restore in {STATE}")
    keys = ("homing_offset", "range_min", "range_max")
    _move(Arm(robot), REST, "to rest for the write")
    robot.write_calibration({j: {k: c[k] for k in keys} for j, c in before["joints"].items()}, save=True)
    print(f"restored the calibration from {before['date']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("bootstrap")
    p = sub.add_parser("probe")
    p.add_argument("joints", nargs="+", help="e.g. gripper wrist_roll (in ORDER)")
    w = sub.add_parser("write")
    w.add_argument("--dry-run", action="store_true")
    sub.add_parser("restore")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    robot = Robot(base_enabled=False)
    with robot:
        if args.step == "bootstrap":
            bootstrap(robot)
        elif args.step == "probe":
            for name in args.joints:
                probe_joint(robot, name if name.startswith("arm_") else f"arm_{name}")
        elif args.step == "write":
            write(robot, args.dry_run)
        else:
            restore(robot)


if __name__ == "__main__":
    main()
