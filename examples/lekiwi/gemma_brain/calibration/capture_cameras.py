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

# ruff: noqa: E402, N803, N806
# (N8: transforms and camera matrices keep their textbook capitals, as in solve_cameras.py)

"""Capture the ChArUco views that solve_cameras.py solves the cameras from: the boards stay put,
the robot moves, nobody holds anything.

    .venv/bin/python calibration/capture_cameras.py scene --plan-only
    .venv/bin/python calibration/capture_cameras.py scene [--no-sweep] [--no-closeup] [--reverse]
    .venv/bin/python calibration/capture_cameras.py home --scene runs/<t>-scene
    .venv/bin/python calibration/capture_cameras.py closeup

The wrist board lies flat on the floor centred on the reach point (its long side across the
robot), the front board stands upright in front where both cameras see it, the side board
upright beside the robot, either side (calibration/CAMERA_CALIBRATION.md). First the front
camera (arm folded, out of its view) takes a few frames; then the arm visits plan()'s poses,
each a joint move checked for floor and body clearance, and at each one the wrist camera takes
a frame and the measured joint angles are recorded; the arm folds. Then the base sweep (not
with --no-sweep): the base turns and backs up with the arm folded, and the front camera takes
a frame at each stop, for its lens; then it drives back to the start, closed loop on the front
camera's view of the boards (once calib.json has the measured front camera). Then the side board
close-up (not with --no-closeup): the robot turns 90 deg right, or left if the board isn't there,
drives in to 30 cm from it, takes frames at a few headings, backs out and returns to the start.
--reverse visits the arm poses in the opposite order. --plan-only prints the poses'
joint targets (or why one is refused) and the sweep's moves, and moves nothing.
runs/<t>-scene/scene.json plus the frames.

    .venv/bin/python calibration/capture_cameras.py lens --cam wrist|front [--views 25]

The fallback for a camera the robot can't point at the boards: no robot motion, a board held
in front of the camera, moved around (30 cm and more away, every part of the image, tilted).
A view is kept when the board has been still for two frames and is somewhere new (position or
size); runs/<t>-lens-<cam>/views.json plus the frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "web_server"))  # the Pi side: auto_control.py

from auto_control import BASE_POS_TOL_M, BASE_THETA_TOL_DEG
from brain import params as prm
from common.arm import (
    DEFAULT_FLOOR_Z_M,
    FREE_JOINT_SPEED,
    JOINT_LIMIT_MARGIN_DEG,
    NAMED_POSES,
    Arm,
    Calibration,
    PlanError,
    plan_ee_line,
)

from calibration import boards
from calibration.solve_cameras import board_in_camera, inv

# A lens view is new when its board's centre moved this share of the image width from every
# kept view of that board, or its size changed by this share; still when no corner moved more
# than STILL_PX between two frames. Convention: enough to spread the views over the image.
NEW_SHARE = 0.08
STILL_PX = 1.5
SETTLE_S = 0.5  # after a move, before the frame: the arm's sway dies down (convention)
FRONT_FRAMES = 3


# The scene's arm poses, in joint space: anchors (a TCP pose: height above the floor, pitch),
# each with joint changes (deg) around it. Over the floor board the gripper points straight down
# from two heights; the forward anchor faces the upright board 30 deg down from 18 cm up (lower
# poses see only the floor board; "ready" is too high and too close to it). Each change turns one
# joint (or two, to change the height): views that differ in rotation about several axes, each
# joint's zero seen on its own. Heights and pitches from the Oct 29 plan check: inside the reach,
# the joint limits and the boards' views.
FLOOR_CHANGES = (
    {},
    {"arm_wrist_flex": 12},
    {"arm_wrist_flex": -12},
    {"arm_wrist_roll": 45},
    {"arm_wrist_roll": -45},
    {"arm_shoulder_pan": 10},
    {"arm_shoulder_pan": -10},
    {"arm_shoulder_lift": -6, "arm_wrist_flex": 6},
)
FORWARD_CHANGES = (
    {},
    {"arm_wrist_flex": 6},
    {"arm_wrist_flex": -6},
    {"arm_shoulder_pan": 8},
    {"arm_shoulder_pan": -8},
    {"arm_wrist_roll": 30},
    {"arm_wrist_roll": -30},
)
FORWARD_X_M = 0.26
# Three more arm shapes looking at the floor board (x past its centre, height, pitch): lift and
# elbow move differently from the down anchors, so their zero offsets come apart. The real error
# isn't noise but pose-dependent (backlash, sag, link lengths: two subsets of the first Oct 29
# scene gave offsets 3 deg apart while noise alone allows 0.1), and the offsets soak up some of
# it, so the shapes are spread wide. Picked by the predicted offset uncertainty from that scene
# (lift-elbow correlation -0.75 -> -0.24).
SHAPE_ANCHORS = (("low", 0.03, 0.06, 50.0), ("far", 0.11, 0.09, 70.0), ("near", -0.09, 0.16, 70.0))
SHAPE_CHANGES = (
    {},
    {"arm_wrist_flex": 8},
    {"arm_wrist_flex": -8},
    {"arm_shoulder_pan": 10},
    {"arm_shoulder_pan": -10},
)
# The front lens: the base turns and backs up with the arm folded, so the fixed boards sweep the
# front camera's image (it sees about +/-30 deg) and change size. Never forward: the floor board
# lies ~6 cm ahead of the robot.
SWEEP_BACK_M = (0.0, 0.10, 0.20)
SWEEP_HEADINGS_DEG = (-20.0, -10.0, 0.0, 10.0, 20.0)
# Back to the start after the sweep, closed loop: the front camera's view of the boards against
# the run's first frame says where the base is; turn, then drive, until within the base's own
# tolerances (the Pi's: a move smaller than these counts as done before it starts, Oct 29: a
# 1.2-1.9 deg turn didn't move the base). A single drive never goes further forward than
# MAX_RETURN_FORWARD_M.
RETURN_TOL_M = BASE_POS_TOL_M
RETURN_TOL_DEG = BASE_THETA_TOL_DEG
RETURN_TRIES = 4
# The front lens's close-up: the side board (beside the robot, found by turning 90 deg right, then
# left) is approached to CLOSEUP_STANDOFF_M, where it fills ~70% of the image width (24.5 cm at
# f ~550 px) and, the camera being low and tilted down, reaches the top of the image: the rows the
# sweep leaves empty. Closed loop on the board's measured distance, one drive of at most
# CLOSEUP_STEP_M at a time; then a few headings there, and back out the same way.
CLOSEUP_BOARD = "side"
SIDE_TURNS_DEG = (-90.0, 90.0)  # right first
CLOSEUP_STANDOFF_M = 0.30
CLOSEUP_STEP_M = 0.15
CLOSEUP_TRIES = 6
CLOSEUP_HEADINGS_DEG = (-15.0, 0.0, 15.0)
# The 90 and 180 deg turns to find the side board go at half the Pi's default turn speed (25
# deg/s): the motor bus dropped right after the first full-speed 90 deg turn (Oct 29, cause not
# known; a supply sagging under the turn's current would fit).
SIDE_TURN_SPEED_DEG_S = 12.0
MAX_RETURN_FORWARD_M = 0.25


def _in_limits(calib: Calibration, joint: str, deg: float) -> bool:
    lo, hi = calib.limits(joint)
    return lo + JOINT_LIMIT_MARGIN_DEG <= deg <= hi - JOINT_LIMIT_MARGIN_DEG


def plan(
    calib: Calibration,
    board_x: float = prm.REACH_POINT_X_M.value,
    floor_z: float = DEFAULT_FLOOR_Z_M,
    reverse: bool = False,
) -> list[dict]:
    """[{"name", "anchor", "joints"} or {..., "refused"}]: every pose in the joint limits (less the
    planner's margin) and every move from the previous pose clear of the floor and the body.
    reverse: the same poses in the opposite order, so most are reached from the other side (a
    joint's play past its encoder shows as the difference)."""
    arm = Arm(None, calib=calib, floor_z=floor_z)
    ready = dict(NAMED_POSES["ready"])
    anchors = []
    for name, x, h, pitch, changes in (
        ("down 11 cm", board_x, 0.11, 90.0, FLOOR_CHANGES),
        ("down 14 cm", board_x, 0.14, 90.0, FLOOR_CHANGES),
        ("forward", FORWARD_X_M, 0.18, 30.0, FORWARD_CHANGES),
        *((n, board_x + dx, h, pitch, SHAPE_CHANGES) for n, dx, h, pitch in SHAPE_ANCHORS),
    ):
        try:
            wps = plan_ee_line(calib, ready, [x, 0.0, floor_z + h], pitch, 0.0, 0.04, arm.z_min)[0]
            anchors.append((name, {**ready, **wps[-1]}, changes))
        except PlanError as e:
            anchors.append((name, None, str(e)))
    poses = []
    for name, base, changes in anchors:
        if base is None:
            poses.append({"name": name, "anchor": name, "refused": changes})
            continue
        for ch in changes:
            label = name + "".join(f" {j[4:]}{d:+d}" for j, d in ch.items())
            goal = {j: v + ch.get(j, 0) for j, v in base.items()}
            bad = [j for j in ch if not _in_limits(calib, j, goal[j])]
            if bad:
                poses.append(
                    {
                        "name": label,
                        "anchor": name,
                        "refused": f"{bad[0]} past its limit ({goal[bad[0]]:.1f} deg)",
                    }
                )
            else:
                poses.append({"name": label, "anchor": name, "joints": goal})
    out, prev = [], ready
    for p in reversed(poses) if reverse else poses:
        if "joints" in p:
            try:
                arm.check_clear(prev, p["joints"], p["name"])
                prev = p["joints"]
            except PlanError as e:
                p = {"name": p["name"], "anchor": p["anchor"], "refused": str(e)}
        out.append(p)
    return out


def _cameras() -> dict | None:
    """calib.json "cameras" (solve_cameras.py --write), or None before the first calibration."""
    calib = json.loads((ROOT / "data" / "calib.json").read_text())
    return calib.get("cameras", {}).get("value")


def _out(kind: str) -> Path:
    d = ROOT / "runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-{kind}"
    d.mkdir(parents=True)
    return d


# --- lens ---


def _shape(d: dict, width: int) -> np.ndarray:
    px = np.asarray(d["px"])
    return np.array([*(px.mean(axis=0) / width), np.log(np.ptp(px, axis=0).max() / width)])


def _is_new(d: dict, kept: list[dict], width: int) -> bool:
    s = _shape(d, width)
    return all(np.max(np.abs(s - _shape(k, width))) > NEW_SHARE for k in kept)


def _still(a: dict, b: dict) -> bool:
    common = set(a["ids"]) & set(b["ids"])
    if len(common) < boards.MIN_CORNERS:
        return False
    pa, pb = (dict(zip(d["ids"], map(tuple, d["px"]), strict=True)) for d in (a, b))
    return max(np.hypot(pa[i][0] - pb[i][0], pa[i][1] - pb[i][1]) for i in common) < STILL_PX


def _write_views(out: Path, cam: str, size, views: list[dict]) -> None:
    (out / "views.json").write_text(
        json.dumps({"camera": cam, "image_size": size, "views": views}, indent=1) + "\n"
    )


def capture_lens(robot, cam: str, n_views: int) -> Path:
    out = _out(f"lens-{cam}")
    views, kept, prev, size = [], {}, {}, None
    print(f"Move a board around in front of the {cam} camera; {n_views} views to go (Ctrl-C ends early).")
    try:
        while len(views) < n_views:
            jpeg, _ = robot.frame(cam, after=time.time())
            img = boards.decode(jpeg)
            size = [img.shape[1], img.shape[0]]
            seen = boards.detect(img)
            new = {
                n: d
                for n, d in seen.items()
                if n in prev and _still(d, prev[n]) and _is_new(d, kept.get(n, []), size[0])
            }
            prev = seen
            if not new:
                continue
            k = len(views)
            (out / f"{k:03d}.jpg").write_bytes(jpeg)
            views.append({"frame": f"{k:03d}.jpg", "boards": new})
            for n, d in new.items():
                kept.setdefault(n, []).append(d)
            _write_views(out, cam, size, views)  # after every view: however the capture ends, they're kept
            print(
                f"  view {k + 1}/{n_views}: "
                + ", ".join(f"{n} ({len(d['ids'])} corners)" for n, d in new.items())
            )
    except KeyboardInterrupt:
        print("\nended early")
    _write_views(out, cam, size, views)
    print(f"{len(views)} views in {out}")
    return out


# --- scene ---


def _grab(robot, cam: str, out: Path, tag: str) -> dict:
    jpeg, _ = robot.frame(cam, after=time.time())
    (out / f"{tag}.jpg").write_bytes(jpeg)
    seen = boards.detect(jpeg)
    print(f"  {tag}: " + (", ".join(f"{n} ({len(d['ids'])} corners)" for n, d in seen.items()) or "no board"))
    return {"frame": f"{tag}.jpg", "boards": seen}


def capture_scene(robot, poses: list[dict]) -> Path:
    out = _out("scene")
    arm = Arm(robot)
    scene = {"front": [], "wrist": [], "plan": poses}
    try:
        arm.reset()
        time.sleep(SETTLE_S)
        scene["front"] = [_grab(robot, "front", out, f"front-{k}") for k in range(FRONT_FRAMES)]
        for k, p in enumerate(poses):
            if "joints" not in p:
                continue
            cur = arm.targets()
            arm.check_clear(cur, {**cur, **p["joints"]}, f"capture pose {k}")
            robot.joints(p["joints"], speed=FREE_JOINT_SPEED, z_min=arm.joint_z_min())
            time.sleep(SETTLE_S)
            rec = _grab(robot, "wrist", out, f"wrist-{k:02d}")
            rec["joints"] = {j: float(v) for j, v in robot.status()["joints"].items()}
            rec["anchor"] = p["anchor"]
            if rec["boards"]:
                scene["wrist"].append(rec)
    finally:
        (out / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
        arm.reset()
    print(f"{len(scene['wrist'])} wrist poses with a board, {len(scene['front'])} front frames in {out}")
    return out


def sweep_plan() -> list[dict]:
    """The base moves, each relative to the last: at each distance back, the headings in turn,
    then back to heading 0 before backing up further."""
    moves, heading, back = [], 0.0, 0.0
    for b in SWEEP_BACK_M:
        if b != back:
            moves.append({"turn_deg": 0.0, "back_m": b - back, "at": None})
            back = b
        for h in SWEEP_HEADINGS_DEG:
            moves.append({"turn_deg": h - heading, "back_m": 0.0, "at": {"back_m": b, "heading_deg": h}})
            heading = h
        moves.append({"turn_deg": -heading, "back_m": 0.0, "at": None})
        heading = 0.0
    return moves


def capture_sweep(robot, out: Path, moves: list[dict]) -> list[dict]:
    """The front camera's frames from each base position of the sweep (the arm folded)."""
    frames = []
    for m in moves:
        robot.base(dx=-m["back_m"], dtheta_deg=m["turn_deg"])
        if m["at"] is None:
            continue
        time.sleep(SETTLE_S)
        at = m["at"]
        rec = _grab(robot, "front", out, f"sweep-{100 * at['back_m']:02.0f}cm-{at['heading_deg']:+03.0f}deg")
        rec["base"] = at
        if rec["boards"]:
            frames.append(rec)
    return frames


def base_offset(T_cam_board_ref, T_cam_board_now, T_base_cam) -> tuple[float, float, float]:
    """Where the base is now against where it was in the reference frame, in the reference
    base frame: (x m, y m, heading deg). The board is fixed and the camera rides the base."""
    move = T_base_cam @ T_cam_board_ref @ inv(T_cam_board_now) @ inv(T_base_cam)
    return float(move[0, 3]), float(move[1, 3]), float(np.degrees(np.arctan2(move[1, 0], move[0, 0])))


def front_measure(robot, ref_boards: dict, cams: dict):
    """A function giving the base's offset from where the reference frame was taken, from a fresh
    front frame: one estimate per board seen in both (any fixed board will do; from 20 cm back
    the floor board is too foreshortened to read, the upright ones aren't), their median."""
    K, dist = np.asarray(cams["front"]["K"]), np.asarray(cams["front"]["dist"])
    T_base_cam = np.asarray(cams["front"]["T_base_cam"])
    ref = {n: board_in_camera(K, dist, n, d) for n, d in ref_boards.items()}

    def measure() -> tuple[float, float, float]:
        time.sleep(SETTLE_S)
        seen = boards.detect(robot.frame("front", after=time.time())[0])
        est = [
            base_offset(ref[n], board_in_camera(K, dist, n, d), T_base_cam)
            for n, d in seen.items()
            if n in ref
        ]
        if not est:
            raise RuntimeError(
                "none of the start's boards is in the front camera's view: can't find the start"
            )
        x, y, th = np.median(np.asarray(est), axis=0)
        return float(x), float(y), float(th)

    return measure


def board_ahead(robot, cams: dict, name: str = CLOSEUP_BOARD) -> float | None:
    """How far the named board's centre is ahead of the front camera, along the base's forward
    axis (m), from a fresh front frame; None when the board isn't in view."""
    time.sleep(SETTLE_S)
    seen = boards.detect(robot.frame("front", after=time.time())[0])
    if name not in seen:
        return None
    K, dist = np.asarray(cams["front"]["K"]), np.asarray(cams["front"]["dist"])
    T_base_cam = np.asarray(cams["front"]["T_base_cam"])
    T = T_base_cam @ board_in_camera(K, dist, name, seen[name])
    centre = np.asarray(boards.BOARDS[name].getChessboardCorners(), float).mean(axis=0)
    return float((T[:3, :3] @ centre + T[:3, 3])[0] - T_base_cam[0, 3])


def drive_to(robot, cams: dict, target_m: float) -> float:
    """Drive straight until the close-up board is target_m ahead (within the base's tolerance),
    re-measuring after every drive of at most CLOSEUP_STEP_M; returns the last distance."""
    for _ in range(CLOSEUP_TRIES):
        d = board_ahead(robot, cams)
        if d is None:
            raise RuntimeError(f"lost the {CLOSEUP_BOARD} board while driving to it")
        if abs(d - target_m) < BASE_POS_TOL_M:
            return d
        robot.base(dx=float(np.clip(d - target_m, -CLOSEUP_STEP_M, CLOSEUP_STEP_M)))
    raise RuntimeError(
        f"not {100 * target_m:.0f} cm from the {CLOSEUP_BOARD} board after {CLOSEUP_TRIES} drives"
    )


def capture_closeup(robot, out: Path, cams: dict) -> tuple[list[dict], float | None]:
    """Find the side board (a 90 deg turn right, then left), drive in to CLOSEUP_STANDOFF_M, take
    a front frame at each of CLOSEUP_HEADINGS_DEG, drive back out and turn back. Returns the
    frames and the turn that found the board (None: on neither side, nothing taken)."""
    found, heading = None, 0.0
    for turn in SIDE_TURNS_DEG:
        robot.base(dtheta_deg=turn - heading, theta_speed=SIDE_TURN_SPEED_DEG_S)
        heading = turn
        if board_ahead(robot, cams) is not None:
            found = turn
            break
    if found is None:
        robot.base(dtheta_deg=-heading, theta_speed=SIDE_TURN_SPEED_DEG_S)
        print(f"  no {CLOSEUP_BOARD} board on either side: no close-up")
        return [], None
    print(f"  the {CLOSEUP_BOARD} board is on the {'right' if found < 0 else 'left'}")
    start = board_ahead(robot, cams)  # the distance to come back out to
    drive_to(robot, cams, CLOSEUP_STANDOFF_M)
    frames, h = [], 0.0
    for target in CLOSEUP_HEADINGS_DEG:
        robot.base(dtheta_deg=target - h)
        h = target
        time.sleep(SETTLE_S)
        rec = _grab(robot, "front", out, f"closeup-{target:+03.0f}deg")
        rec["base"] = {"side_turn_deg": found, "heading_deg": target}
        if rec["boards"]:
            frames.append(rec)
    robot.base(dtheta_deg=-h)
    drive_to(robot, cams, start)
    robot.base(dtheta_deg=-found, theta_speed=SIDE_TURN_SPEED_DEG_S)
    return frames, found


def return_to_start(robot, measure) -> tuple[float, float, float]:
    """Turn, then drive, until the base is back where measure() counts from (within RETURN_TOL);
    returns the last offset. Raises if it doesn't get there in RETURN_TRIES moves."""
    for k in range(RETURN_TRIES + 1):
        x, y, th = measure()
        print(f"  base {100 * x:+.1f} cm forward, {100 * y:+.1f} cm left, {th:+.1f} deg from the start")
        if np.hypot(x, y) < RETURN_TOL_M and abs(th) < RETURN_TOL_DEG:
            return x, y, th
        if k == RETURN_TRIES:
            break
        if abs(th) >= RETURN_TOL_DEG:
            robot.base(dtheta_deg=-th)  # the heading first: then the offset is in the start's axes
        else:
            if -x > MAX_RETURN_FORWARD_M:
                raise RuntimeError(f"the start is {-100 * x:.0f} cm ahead: more than a return drives blind")
            robot.base(dx=-x, dy=-y)
    raise RuntimeError(f"not back at the start after {RETURN_TRIES} moves")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    lp = sub.add_parser("lens")
    lp.add_argument("--cam", choices=["wrist", "front"], required=True)
    lp.add_argument("--views", type=int, default=25)
    sp = sub.add_parser("scene")
    sp.add_argument("--plan-only", action="store_true")
    sp.add_argument("--no-sweep", action="store_true", help="skip the base sweep for the front lens")
    sp.add_argument("--reverse", action="store_true", help="visit the poses in the opposite order")
    sp.add_argument(
        "--no-closeup", action="store_true", help="skip the side board close-up for the front lens"
    )
    sub.add_parser("closeup", help="only the side board close-up, from where the robot is, and back")
    hp = sub.add_parser(
        "home", help="drive the base back to where a scene started (front camera, floor board)"
    )
    hp.add_argument("--scene", type=Path, required=True)
    sp.add_argument(
        "--board-x", type=float, default=prm.REACH_POINT_X_M.value, help="the floor board's centre (arm x, m)"
    )
    args = ap.parse_args()

    from common.robot import Robot

    if args.what == "lens":
        capture_lens(Robot(), args.cam, args.views)  # frames only: the robot isn't taken over
        return
    if args.what == "closeup":
        out, cams = _out("closeup"), _cameras()
        with Robot() as robot:
            ref = boards.detect(robot.frame("front", after=time.time())[0])
            frames, turn = capture_closeup(robot, out, cams)
            (out / "closeup.json").write_text(
                json.dumps({"closeup": frames, "closeup_turn_deg": turn}, indent=1) + "\n"
            )
            print(f"{len(frames)} close-up frames in {out}; back to the start:")
            return_to_start(robot, front_measure(robot, ref, cams))
        return
    if args.what == "home":
        ref = json.loads((args.scene / "scene.json").read_text())["front"][0]["boards"]
        with Robot() as robot:
            return_to_start(robot, front_measure(robot, ref, _cameras()))
        return
    robot = Robot()
    poses = plan(Calibration.from_robot(robot), args.board_x, reverse=args.reverse)
    for k, p in enumerate(poses):
        print(
            f"{k:2d} {p['name']:28s}",
            {j[4:]: round(v, 1) for j, v in p["joints"].items()}
            if "joints" in p
            else f"REFUSED: {p['refused']}",
        )
    print(f"{sum('joints' in p for p in poses)}/{len(poses)} poses planned")
    moves = [] if args.no_sweep else sweep_plan()
    if moves:
        print("then, the arm folded, the base sweep (each move relative to the last):")
        for m in moves:
            what = f"back {100 * m['back_m']:.0f} cm" if m["back_m"] else f"turn {m['turn_deg']:+.0f} deg"
            print(f"  {what:14s}", f"-> front frame at {m['at']}" if m["at"] else "")
    if moves and not args.no_closeup:
        print(
            f"then back to the start, and the {CLOSEUP_BOARD} board close-up: turn {SIDE_TURNS_DEG[0]:+.0f} deg "
            f"(then {SIDE_TURNS_DEG[1]:+.0f} if it isn't there), drive in to {100 * CLOSEUP_STANDOFF_M:.0f} cm from "
            f"it (at most {100 * CLOSEUP_STEP_M:.0f} cm a drive, re-measured), frames at "
            f"{', '.join(f'{h:+.0f}' for h in CLOSEUP_HEADINGS_DEG)} deg, back out, turn back, back to the start"
        )
    if args.plan_only:
        return
    with Robot(base_enabled=bool(moves)) as robot:
        out = capture_scene(robot, poses)
        if moves:
            scene = json.loads((out / "scene.json").read_text())
            try:
                scene["sweep"] = capture_sweep(robot, out, moves)
            finally:
                (out / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
            print(f"{len(scene['sweep'])} sweep frames with a board")
            cams = _cameras()
            if cams and scene["front"] and scene["front"][0]["boards"]:
                measure = front_measure(robot, scene["front"][0]["boards"], cams)
                print("back to the start:")
                scene["returned"] = return_to_start(robot, measure)
                if not args.no_closeup:
                    print(f"the {CLOSEUP_BOARD} board close-up:")
                    try:
                        scene["closeup"], scene["closeup_turn_deg"] = capture_closeup(robot, out, cams)
                    finally:
                        (out / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
                    print("back to the start:")
                    scene["returned"] = return_to_start(robot, measure)
                (out / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
            else:
                print("no measured front camera in calib.json yet: the robot stays 20 cm back, no close-up")


if __name__ == "__main__":
    main()
