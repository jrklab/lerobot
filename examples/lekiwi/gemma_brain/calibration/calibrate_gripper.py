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

"""Measure the gripper's opening in millimetres against its angle (calib.json "gripper_opening"),
by closing it on objects of known width: nothing (0 mm), a 2x4 brick across its short side
(15.8 mm) and its long side (31.8 mm), a 2x6 brick's long side (47.8 mm), twice each.
Interactive: run it in a terminal. calibrate_all.py's gripper step does the same with no one
there (auto(): a 2x4 brick lying on the floor, grasped as a pick does).

    .venv/bin/python calibration/calibrate_gripper.py [--objects nothing 2x4-short ...] [--add] [--write]

--add keeps the points already in calib.json and adds these (e.g. --objects 2x6-long --add).

The arm goes to the hover over the reach point (gripper vertical, 10 cm up) and stays there.
For each object the gripper opens fully; place the object between the jaws, at the jaw tips
where a pick holds it, held lightly by its ends, and press ENTER: the gripper closes at the
pick's closing speed until it stalls on the object, the same stall a pick ends on, and the
angle it stopped at is recorded. The opening is interpolated straight between the measured
points (each width's mean angle), not fitted with one line: the jaw swings on a pivot and
opens less per degree as it opens wider (Oct 29: ~1.9 mm/deg up to 15.8 mm, ~1.3 to 31.8; a
line was 1.9 mm off). The arm folds at the end.

Why: the gripper's angle depends on where its closed end was recorded at calibration (the
Oct 29 recalibration moved a brick's reading from 8.9 to 9.8-10.0 deg); its opening in
millimetres doesn't, and it compares directly with an object's width.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from brain import actions, params as prm
from brain.geometry import Geometry
from brain.goals import AngleGoal, JawGoal
from brain.params import Param
from brain.primitives import Skills
from common import vision
from common.arm import Arm
from common.robot import JOINT_UNITS, Robot
from common.runlog import Log

CALIB = ROOT / "data" / "calib.json"
# Width (mm) of what the jaws close on: nothing, LEGO bricks (8 mm per stud less the 0.2 mm play).
OBJECTS = {
    "nothing": ("nothing between the jaws", 0.0),
    "2x4-short": ("the 2x4 brick across its short side (2 studs)", 15.8),
    "2x4-long": ("the 2x4 brick across its long side (4 studs)", 31.8),
    "2x6-long": ("the 2x6 brick across its long side (6 studs)", 47.8),
}


def close_measured(robot) -> dict:
    """Close at the pick's speed until the stall (no squeeze: the stall itself is measured) and
    read where the jaws stopped."""
    cmd = {"cmd": "gripper", "pos": 0.0, "speed": prm.CLOSE_SPEED.value, "units": JOINT_UNITS}
    res = robot.run(cmd).get("result", {})
    deg = float(robot.status()["joints"]["arm_gripper"])
    return {"deg": round(deg, 2), "stalled": bool(res.get("stalled"))}


def curve(points: list[dict]) -> list[list[float]]:
    """[[deg, mm], ...]: each width's mean angle, by width."""
    widths = sorted({p["mm"] for p in points})
    return [[round(float(np.mean([p["deg"] for p in points if p["mm"] == w])), 2), w] for w in widths]


# --- calibrate_all.py's gripper step: a 2x4 brick on the floor, no one there ---

BRICK = "brick"  # what ColourBoxes stands for
AUTO_REPEATS = 2  # each width closed on this many times (as the interactive run)
LEGO_PITCH_MM, LEGO_PLAY_MM = 8.0, 0.2  # a brick is 8 mm per stud less the 0.2 mm play (OBJECTS)
# The brick lies beside the floor board on its right (the side board's side, owner Nov 4): its
# middle about this far out past the board's edge; the line-up takes the rest.
BRICK_PAST_EDGE_M = 0.05  # convention
# Each grasp squares the brick to the jaws first, tighter than a pick's 12 deg: a long-side close on a
# brick a few degrees off closed 2.2 deg (~3 mm) wider than the one before (Nov 4).
ALIGN_TOL = Param(3.0, "RULE", "the gripper calibration's grasps: the brick within this of square to the jaws (deg)")


class ColourBoxes:
    """The box model's stand-in for the calibration: the largest strongly coloured blob in the
    frame (vision.coloured_blobs), the only coloured thing beside the black-and-white boards.
    No model server needed."""

    def box(self, jpeg: bytes, prompt: str) -> SimpleNamespace:
        found = vision.coloured_blobs(jpeg)
        return SimpleNamespace(value=found[0] if found else None)


def lego_width_mm(measured_m: float) -> float:
    """A brick side's width from the camera's measure (+/-1 mm): the nearest whole number of
    studs, its exact width (15.8 mm cannot be taken for 23.8)."""
    studs = max(1, round((1000.0 * measured_m + LEGO_PLAY_MM) / LEGO_PITCH_MM))
    return round(studs * LEGO_PITCH_MM - LEGO_PLAY_MM, 1)


def brick_side_y(board_pose) -> float:
    """Where across (arm y, m) the brick lies: past the floor board's right-hand edge (its inner
    corners' least y, less a square to the edge) by BRICK_PAST_EDGE_M."""
    from calibration import boards

    board = boards.BOARDS["wrist"]
    corners = boards.object_points("wrist", np.arange(len(board.getChessboardCorners())))
    pose = np.asarray(board_pose)
    pts = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
    return float(pts[:, 1].min() - board.getSquareLength() - BRICK_PAST_EDGE_M)


def _turned(sk: Skills, roll: float) -> float:
    """The roll a quarter turn from `roll`, either way, within the wrist's range."""
    lo, hi = sk.roll_range
    for r in (roll + 90.0, roll - 90.0):
        if lo <= r <= hi:
            return r
    raise RuntimeError(f"no quarter turn from roll {roll:.0f} deg within {lo:.0f}..{hi:.0f}")


def _ok(r, what: str):
    if not r.ok:
        raise RuntimeError(f"{what}: {r.reason}")
    return r


def _grasp_once(sk: Skills, robot, wide: float, side: str) -> dict:
    """From the hover: square the jaws to the brick (ALIGN_TOL), a quarter turn more for its long
    side, centre high and low as a pick does, measure the side, open to fit, land, close until
    the stall; then open and back up to the hover."""
    _ok(sk.servo(BRICK, AngleGoal(ALIGN_TOL)), "align")
    if side == "long":
        _ok(sk.move("arm", delta=(0.0, 0.0, 0.0), roll=_turned(sk, sk.roll)), "quarter turn")
    _ok(sk.servo(BRICK, JawGoal(tol_m=prm.REACH_TOL_M, plane_h=sk.object_height(BRICK))), "centre high")
    _ok(sk.move("arm", height=sk.final_look_height(BRICK)), "down")
    _ok(sk.servo(BRICK, JawGoal(plane_h=sk.object_height(BRICK))), "centre low")
    w = actions.measure_width(sk, BRICK)
    if w is None:
        raise RuntimeError("the brick's sides weren't measured at the last look")
    measured = w["short_m"] if side == "short" else w["long_m"]
    _ok(actions.open_to_fit(sk, BRICK), "open")
    _ok(actions.land(sk, BRICK), "land")
    c = close_measured(robot)
    robot.gripper(wide)
    _ok(actions.hover_at(sk, prm.HOVER_HEIGHT_M), "up")
    mm = lego_width_mm(measured)
    print(f"    {side} side: measured {1000 * measured:.1f} mm -> {mm} mm, closed at {c['deg']} deg")
    return {"mm": mm, "what": f"2x4 brick, {side} side", "measured_mm": round(1000 * measured, 1), **c}


def auto(robot, board_pose, out: Path, repeats: int = AUTO_REPEATS, moves: list | None = None) -> list[dict]:
    """The gripper's points with no one there: closed on nothing at the hover; then the 2x4 brick
    beside the floor board, the base moved across to it, found by colour, lined up as a pick
    does, and closed on across its short side and (a quarter turn) its long side, `repeats`
    times each. The base stays where the line-up left it: each base move, as the Pi reported it
    ((dx, dy, dtheta_deg)), goes into `moves` for the caller to undo."""
    arm = Arm(robot)
    sk = Skills(robot, arm, Geometry.load(), Log(out), main=ColourBoxes())
    log_ev = sk.ev

    def ev(name: str, **kw) -> None:
        if moves is not None and name == "move" and kw.get("part") == "base" and kw.get("moved"):
            moves.append(tuple(kw["moved"]))
        log_ev(name, **kw)

    sk.ev = ev
    points: list[dict] = []
    wide = arm.calib.limits("arm_gripper")[1]
    try:
        _ok(actions.hover(sk), "hover")
        for _ in range(repeats):
            robot.gripper(wide)
            points.append({"mm": 0.0, "what": "nothing", **close_measured(robot)})
        print(f"    nothing: closed at {[p['deg'] for p in points]} deg")
        robot.gripper(wide)
        _ok(sk.move("base", delta=(0.0, brick_side_y(board_pose))), "across to the brick")
        if sk.find(BRICK, "wrist") is None:
            raise RuntimeError("no brick in the wrist camera's view right of the floor board")
        _ok(actions.line_up(sk, BRICK), "line up")
        for side in ("short", "long"):
            for _ in range(repeats):
                points.append(_grasp_once(sk, robot, wide, side))
    finally:
        robot.gripper(wide)
        arm.reset()
    return points


def merged(points: list[dict], old: list[dict]) -> list[dict]:
    """These points, and the old ones of widths these don't measure (the 2x6 brick's 47.8 mm,
    which the gripper step doesn't grasp: it's at the jaws' widest)."""
    widths = {p["mm"] for p in points}
    return points + [p for p in old if p["mm"] not in widths]


def write_curve(points: list[dict], method: str) -> list[list[float]]:
    """The curve from the points into calib.json "gripper_opening"; refused (nothing written)
    when the angle doesn't grow with the width."""
    c = curve(points)
    if any(b[0] <= a[0] for a, b in zip(c, c[1:], strict=False)):
        raise RuntimeError(f"the angle doesn't grow with the width ({c}): a slipped object? nothing written")
    calib = json.loads(CALIB.read_text())
    calib["gripper_opening"] = {
        "value": {"curve": c},
        "meaning": "jaw opening (mm, at the tips) against the gripper angle (deg open from its recorded closed end): "
                   "straight between these [deg, mm] points; beyond the last, its last segment's slope",
        "method": method,
        "date": datetime.date.today().isoformat(),
        "points": points,
    }
    CALIB.write_text(json.dumps(calib, indent=2) + "\n")
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--objects", nargs="+", choices=list(OBJECTS), default=list(OBJECTS))
    ap.add_argument("--add", action="store_true", help="keep the points already in calib.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    old = json.loads(CALIB.read_text()).get("gripper_opening", {}).get("points", []) if args.add else []
    points = []
    with Robot() as robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, Geometry.load())
        try:
            r = actions.hover(sk)
            if not r.ok:
                raise SystemExit(f"hover failed: {r.reason}")
            wide = arm.calib.limits("arm_gripper")[1]
            for k in range(args.repeats):
                for what, mm in (OBJECTS[o] for o in args.objects):
                    robot.gripper(wide)
                    input(f"\n[{k + 1}/{args.repeats}] Place {what} ({mm:.1f} mm) at the jaw tips, then press ENTER ")
                    p = {"mm": mm, "what": what, **close_measured(robot)}
                    points.append(p)
                    print(f"    closed at {p['deg']:.2f} deg (stalled: {p['stalled']})")
            robot.gripper(wide)
            input("\nTake the brick out, then press ENTER to fold the arm ")
        finally:
            arm.reset()

    points = old + points
    c = curve(points)
    print("\npoints:", [(p["mm"], p["deg"]) for p in points])
    spread = {w: round(max(p["deg"] for p in points if p["mm"] == w) - min(p["deg"] for p in points if p["mm"] == w), 2)
              for _, w in c}
    print("curve (deg -> mm):", c, "| repeat spread (deg):", spread)
    for (d0, w0), (d1, w1) in zip(c, c[1:], strict=False):
        print(f"  {w0:4.1f} -> {w1:4.1f} mm over {d1 - d0:5.2f} deg: {(w1 - w0) / (d1 - d0):.2f} mm/deg")
    if args.write:
        try:
            write_curve(points, "calibrate_gripper.py: closed at the pick's speed until the stall on nothing and LEGO bricks of known width")
        except RuntimeError as e:
            raise SystemExit(str(e)) from None
        print(f"wrote {CALIB}")
    elif any(b[0] <= a[0] for a, b in zip(c, c[1:], strict=False)):
        raise SystemExit("the angle doesn't grow with the width: a slipped or misplaced object?")


if __name__ == "__main__":
    main()
