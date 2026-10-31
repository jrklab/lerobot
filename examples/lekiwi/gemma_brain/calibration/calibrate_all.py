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

"""The whole robot calibration in one run, no one touching the robot: the arm's joints from
their ends and the URDF, the cameras, the joint zeros the cameras can see, the floor.

    .venv/bin/python calibration/calibrate_all.py --board-thickness 0.00125 [--check-only]

Before: the boards placed (CAMERA_CALIBRATION.md, "The calibration scene"; the floor board
stays), the arm folded at rest, nothing within its reach, ~30 cm clear behind the robot and
the floor clear to the side board; the Pi's service with --allow-uncalibrated. Then:

  0. checks, no motion: the Pi answers in degrees; the front camera sees the front board, the
     wrist camera (at rest) the floor board
  1. coarse (self_calibrate.py): bootstrap from rest, each joint's two ends probed, the
     calibration written and saved
  2. fine (capture_cameras.py, solve_cameras.py --fit-offsets): a scene captured and solved;
     zero offsets larger than their spread are applied (self_calibrate.py correct) and the
     round starts again; with none, the other pose order is captured and the two solved
     together: the verdict always rests on two scenes under the same calibration (at most
     MAX_CORRECTIONS)
  3. the cameras written from those two scenes
  4. the floor: touches on the bare floor between the robot and the floor board, and one on
     the board within reach (where the cameras placed it), less its thickness, for the floor
     plane. The board's top as touched and as the cameras saw it is the same surface: their
     difference is the run's check
  5. a summary, runs/<time>-calibrate_all/summary.json

Everything goes to the run's folder; the calibration from before the run is kept there
(self_calibration.json "before": self_calibrate.py --state <it> restore puts it back).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "web_server")]

from auto_control import JAW_TIP_OFFSET_M
from brain import params as prm
from common.arm import Arm, Calibration
from common.robot import Robot, RobotError

from calibration import boards, calibrate_floor, capture_cameras, self_calibrate as sc, solve_cameras

CALIB = ROOT / "data" / "calib.json"
TOUCHES = ROOT / "data" / "floor_touches.json"
MAX_CORRECTIONS = 3  # convention: the zeros settle in one or two (Oct 30: one)
SCENES_PER_VERDICT = 2  # both pose orders under one calibration (CAMERA_CALIBRATION.md)
TOUCH_REACH_M = (0.18, 0.30)  # where the floor touches have run, gripper vertical (Oct 29-30)
FLOOR_TOUCHES = 3  # convention: a plane needs 3
TOUCH_CLEAR_M = 0.03  # convention: off the board's edge and its backing, for the jaws
FLOOR_BOARD = "wrist"


def _run_dir() -> Path:
    out = ROOT / "runs" / f"{datetime.datetime.now():%Y%m%dT%H%M%S}-calibrate_all"
    out.mkdir(parents=True)
    return out


def check(robot: Robot) -> dict:
    """No motion: what the procedure needs to see before it moves anything."""
    st = robot.status()
    seen = {
        cam: boards.detect(boards.decode(robot.frame(cam, after=time.time())[0]))
        for cam in ("front", "wrist")
    }
    found = {cam: {b: len(d["ids"]) for b, d in v.items()} for cam, v in seen.items()}
    problems = []
    if "front" not in seen["front"]:
        problems.append("the front camera doesn't see the front board")
    if FLOOR_BOARD not in seen["wrist"]:
        problems.append("the wrist camera (the arm at rest) doesn't see the floor board")
    if st.get("arm_calibration") == "calibrated":
        # The coarse stage takes its zeros from where the arm lies. Calibrated, its readings
        # can tell whether it does: held in the planner's rest pose instead, the elbow sits 13 deg
        # less folded and the bootstrap's elbow zero comes out that far off (Oct 30).
        off = {
            j: round(st["joints"][j] - sc.REST[j], 1)
            for j in sc.IK_ARM_JOINTS
            if abs(st["joints"][j] - sc.REST[j]) > sc.REST_TOLERANCE_DEG
        }
        if off:
            problems.append(f"the arm doesn't lie at rest (deg from it: {off}): fold it down onto the base")
    return {"boards_seen": found, "problems": problems, "arm_calibration": st.get("arm_calibration")}


def touch_points(board_pose: np.ndarray) -> tuple[list[float], float | None]:
    """Where to touch along the arm's x axis (y = 0): FLOOR_TOUCHES on the bare floor between
    the robot and the floor board, from the near end of TOUCH_REACH_M to TOUCH_CLEAR_M short of
    the board's edge; and one on the board's printed area if it's within reach (None if not).
    The board as the cameras placed it (Oct 30: its printed area 29-42 cm out, the reach 30)."""
    board = boards.BOARDS[FLOOR_BOARD]
    corners = boards.object_points(FLOOR_BOARD, np.arange(len(board.getChessboardCorners())))
    pts = (board_pose[:3, :3] @ corners.T).T + board_pose[:3, 3]
    if not pts[:, 1].min() < 0 < pts[:, 1].max():
        raise RuntimeError(
            f"the floor board isn't across the arm's centre line (y {pts[:, 1].min():.2f}-{pts[:, 1].max():.2f} m)"
        )
    edge = pts[:, 0].min() - board.getSquareLength()  # the outer edge: one square past the inner corners
    hi = min(edge - TOUCH_CLEAR_M, TOUCH_REACH_M[1])
    if hi - TOUCH_REACH_M[0] < 0.03:  # convention: a slope needs a few cm between touches
        raise RuntimeError(f"too little bare floor before the floor board (its edge {edge:.2f} m out)")
    floor_xs = [round(float(x), 3) for x in np.linspace(TOUCH_REACH_M[0], hi, FLOOR_TOUCHES)]
    on = max(pts[:, 0].min(), TOUCH_REACH_M[0])
    return floor_xs, (round(float(on), 3) if on <= TOUCH_REACH_M[1] else None)


def coarse(robot: Robot, log: dict) -> None:
    sc.bootstrap(robot)
    for joint in sc.ORDER:
        log["ends"][joint] = sc.probe_joint(robot, joint)
    sc.write(robot, dry_run=False)


def fine(robot: Robot, thickness: float, log: dict) -> dict:
    """Capture, solve, correct until no offset is larger than its spread over two scenes."""
    scenes: list[Path] = []
    reverse = False
    while True:
        poses = capture_cameras.plan(
            Calibration.from_robot(robot), prm.REACH_POINT_X_M.value, reverse=reverse
        )
        refused = [p for p in poses if "joints" not in p]
        if refused:
            raise RuntimeError(f"scene poses refused: {[p['name'] for p in refused]}")
        scenes.append(capture_cameras.run_scene(robot, poses, capture_cameras.sweep_plan()))
        reverse = not reverse
        result = solve_cameras.solve(scenes, fit_offsets=True, floor_board_thickness=thickness)
        log["solves"].append(
            {
                "scenes": [s.name for s in scenes],
                **{k: result[k] for k in ("joint_offsets_deg", "spread_deg", "to_apply_deg", "rms_px")},
            }
        )
        apply = result["to_apply_deg"]
        if apply and len(log["corrections"]) < MAX_CORRECTIONS:
            sc.correct(robot, apply)
            log["corrections"].append(apply)
            scenes = []  # earlier scenes were under the old zeros
            continue
        if len(scenes) >= SCENES_PER_VERDICT:
            if apply:
                print(f"still to apply after {MAX_CORRECTIONS} corrections: {apply} -- kept as they are")
            return solve_cameras.solve(scenes, floor_board_thickness=thickness, write=True) | {
                "scenes": scenes
            }


def floor(robot: Robot, cams: dict, thickness: float, log: dict) -> dict:
    """The floor plane from touches on the bare floor and on the board (less its thickness); the
    board's top as touched against as seen by the cameras is the run's check."""
    floor_xs, board_x = touch_points(np.asarray(cams["floor_board_pose"][-1]))
    arm = Arm(robot)
    probes = []
    try:
        arm.pose("ready", speed=20)
        for x in floor_xs + ([board_x] if board_x is not None else []):
            probes.append({**calibrate_floor.probe(arm, x), "on": "floor" if x in floor_xs else "board"})
    finally:
        arm.reset()
    points = [
        {
            "x": p["x"],
            "y": 0.0,
            "z": round(p["floor_tcp_z"] - (thickness if p["on"] == "board" else 0.0), 4),
            "how": f"probe on the {p['on']}",
            "run": f"calibrate_all.py {datetime.date.today().isoformat()}",
        }
        for p in probes
    ]
    out = {"touches": probes, "floor_tcp_z": round(float(np.mean([q["z"] for q in points])), 4)}
    on_board = [p for p in probes if p["on"] == "board"]
    if on_board:
        touched_top = on_board[0]["floor_tcp_z"] - JAW_TIP_OFFSET_M  # the jaw tips on the board
        out.update(
            board_top_touched_z=touched_top,
            board_top_seen_z=cams["board_top_z"],
            touched_minus_seen_mm=round(1000 * (touched_top - cams["board_top_z"]), 1),
        )
    calib = json.loads(CALIB.read_text())
    calib["floor_tcp_z_m"] = {
        "value": out["floor_tcp_z"],
        "unit": "m",
        "frame": "arm base (so101_kinematics), TCP pointing straight down, gripper closed",
        "method": f"calibrate_all.py: touches on the floor (and on the floor board less its {1000 * thickness:.2f} mm)",
        "date": datetime.date.today().isoformat(),
        "probes": probes,
    }
    CALIB.write_text(json.dumps(calib, indent=2) + "\n")
    touches = json.loads(TOUCHES.read_text()) if TOUCHES.exists() else {}
    touches["previous"] = touches.get("points", [])
    touches["points"] = points
    TOUCHES.write_text(json.dumps(touches, indent=1) + "\n")
    log["floor"] = out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-thickness", type=float, required=True, help="the floor board's (m, a ruler's)")
    ap.add_argument("--check-only", action="store_true", help="step 0 only: no motion")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    robot = Robot()
    with robot:
        checked = check(robot)
        print("checks:", checked)
        if checked["problems"] or args.check_only:
            if checked["problems"]:
                raise SystemExit("not started: " + "; ".join(checked["problems"]))
            return
        out = _run_dir()
        sc.STATE = out / "self_calibration.json"
        log = {
            "started": datetime.datetime.now().isoformat(),
            "checks": checked,
            "ends": {},
            "solves": [],
            "corrections": [],
            "board_thickness_m": args.board_thickness,
        }
        t0 = time.monotonic()
        try:
            print("\n=== 1. coarse: the joints' ends ===")
            coarse(robot, log)
            print("\n=== 2-3. fine: the cameras and the joint zeros ===")
            cams = fine(robot, args.board_thickness, log)
            log["cameras"] = {k: cams[k] for k in ("joint_offsets_deg", "board_top_z", "floor_z", "rms_px")}
            log["cameras"]["scenes"] = [s.name for s in cams["scenes"]]
            print("\n=== 4. the floor ===")
            fl = floor(robot, cams, args.board_thickness, log)
            log["state"] = "done"
        except (RobotError, RuntimeError, sc.PlanError) as e:
            log["state"], log["reason"] = "stopped", str(e)
            print(f"\nstopped: {e}")
        finally:
            log["minutes"] = round((time.monotonic() - t0) / 60, 1)
            (out / "summary.json").write_text(json.dumps(log, indent=1, default=str) + "\n")
    print(f"\n{log['state']} in {log['minutes']} min; summary: {out / 'summary.json'}")
    if log["state"] == "done":
        print(f"  corrections applied: {log['corrections'] or 'none'}")
        print(f"  final offsets (within their spread): {log['solves'][-1]['joint_offsets_deg']}")
        if "touched_minus_seen_mm" in fl:
            print(f"  the board's top touched vs seen by the cameras: {fl['touched_minus_seen_mm']:+.1f} mm")
    print(f"  to put back the calibration from before: self_calibrate.py --state {sc.STATE} restore")


if __name__ == "__main__":
    main()
