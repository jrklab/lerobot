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

"""Capture the ChArUco views that solve_cameras.py solves the cameras from: the boards stay put,
the robot moves, nobody holds anything.

    .venv/bin/python calibration/capture_cameras.py scene --plan-only
    .venv/bin/python calibration/capture_cameras.py scene [--no-sweep]

The wrist board lies flat on the floor centred on the reach point (its long side across the
robot), the front board stands upright in front where both cameras see it, the side board
upright where the front camera sees it (calibration/CAMERA_CALIBRATION.md). First the front
camera (arm folded, out of its view) takes a few frames; then the arm visits plan()'s poses,
each a joint move checked for floor and body clearance, and at each one the wrist camera takes
a frame and the measured joint angles are recorded; the arm folds. Then the base sweep (not
with --no-sweep): the base turns and backs up with the arm folded, and the front camera takes
a frame at each stop, for its lens; the robot ends 20 cm back. --plan-only prints the poses'
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


def _in_limits(calib: Calibration, joint: str, deg: float) -> bool:
    lo, hi = calib.limits(joint)
    return lo + JOINT_LIMIT_MARGIN_DEG <= deg <= hi - JOINT_LIMIT_MARGIN_DEG


def plan(
    calib: Calibration, board_x: float = prm.REACH_POINT_X_M.value, floor_z: float = DEFAULT_FLOOR_Z_M
) -> list[dict]:
    """[{"name", "joints"} or {"name", "refused"}]: every pose in the joint limits (less the
    planner's margin) and every move from the previous pose clear of the floor and the body."""
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
    out, prev = [], ready
    for name, base, changes in anchors:
        if base is None:
            out.append({"name": name, "anchor": name, "refused": changes})
            continue
        for ch in changes:
            label = name + "".join(f" {j[4:]}{d:+d}" for j, d in ch.items())
            goal = {j: v + ch.get(j, 0) for j, v in base.items()}
            bad = [j for j in ch if not _in_limits(calib, j, goal[j])]
            try:
                if bad:
                    raise PlanError(f"{bad[0]} past its limit ({goal[bad[0]]:.1f} deg)")
                arm.check_clear(prev, goal, label)
            except PlanError as e:
                out.append({"name": label, "anchor": name, "refused": str(e)})
                continue
            out.append({"name": label, "anchor": name, "joints": goal})
            prev = goal
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    lp = sub.add_parser("lens")
    lp.add_argument("--cam", choices=["wrist", "front"], required=True)
    lp.add_argument("--views", type=int, default=25)
    sp = sub.add_parser("scene")
    sp.add_argument("--plan-only", action="store_true")
    sp.add_argument("--no-sweep", action="store_true", help="skip the base sweep for the front lens")
    sp.add_argument(
        "--board-x", type=float, default=prm.REACH_POINT_X_M.value, help="the floor board's centre (arm x, m)"
    )
    args = ap.parse_args()

    from common.robot import Robot

    if args.what == "lens":
        capture_lens(Robot(), args.cam, args.views)  # frames only: the robot isn't taken over
        return
    robot = Robot()
    poses = plan(Calibration.from_robot(robot), args.board_x)
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


if __name__ == "__main__":
    main()
