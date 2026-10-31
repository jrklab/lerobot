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

# ruff: noqa: E402, N802, N803, N806
# (N8: transforms and camera matrices keep their textbook capitals: T, R, K, the hand-eye G, C, X)

"""Solve the cameras from the ChArUco captures (calibration/capture_cameras.py) into calib.json
"cameras". No robot motion.

    .venv/bin/python calibration/solve_cameras.py --scene runs/<t>-scene [runs/<t2>-scene ...] \\
        [--closeup runs/<t>-closeup ...] [--lens-wrist runs/<t>-lens-wrist] [--lens-front ...] [--write]

Several scenes solve together, each with its own board poses: runs in both pose orders average
out the arm's approach-dependent play (Oct 29: ~1.7 mm the encoders don't see).

The hand-held lens captures are optional (a fallback): the scene's own views are always used,
the arm's for the wrist lens and the base sweep's for the front one (Oct 29: focal lengths +/-2
px, the front centre within 8 px between lens models, 48 px apart before the sweep).

1. Each lens (focal length, optical centre, radial distortion k1 k2: LENS_FLAGS): OpenCV's
   calibrateCamera over every view of a board.
2. The wrist camera's pose on the gripper and the boards' poses in the arm's frame: a linear
   hand-eye solution (Park and Martin, AX = XB) to start from, then a least-squares fit of every
   detected corner's reprojection. With --fit-offsets, also the zero offsets of the three joints
   that tilt the gripper (shoulder lift, elbow, wrist flex), reported with their spread when
   each arm shape (anchor) is left out in turn. Off by default: on Oct 29 they came out under
   1 deg but moved up to 2.6 deg with the shapes, the rigid arm model missing 1-3 mm of
   shape-dependent bending or play that the offsets can't remove, and fitting them lowered the
   error only from 3.8 to 3.1 px. The shoulder pan's and wrist roll's offsets can't be told
   apart from where the boards lie and how the camera is mounted at all.
3. The front camera's pose in the arm's frame: from its views of the floor board (FRONT_VIA),
   whose pose step 2 found, taken with the arm folded before the base moved.

Frames: the arm base frame of so101_kinematics, metres; joint angles in degrees from the homing
tick (joint_angles.py). A joint offset is what is added to the measured angle to get the true
one. No SciPy: the fit is a small Levenberg-Marquardt in NumPy (~25 unknowns).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from common.arm import IK_ARM_JOINTS, kin

from calibration import boards

CALIB = ROOT / "data" / "calib.json"
IMAGE_SIZE = (640, 480)
# The arm's frame (so101_kinematics: the URDF's base_link) has its origin on the arm's mounting
# surface, 3.9 cm behind the pan axis; positions read more easily from the axis.
PAN_AXIS_BASE = np.array([*kin._ORIGIN_TRANSFORMS[0][:2, 3], 0.0])
# The lens model: radial distortion k1, k2 only. The full model's tangential terms and k3 lowered
# the error by 0.01-0.02 px but let the optical centre wander: across four Oct 29 runs the front
# camera's centre spread 15 px (x) and 8 px (y) with them, 1.5 and 4.8 px without; the wrist's
# 2.1 / 0.9 px with, 1.5 / 0.6 px without.
LENS_FLAGS = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST  # both cameras' frames from the Pi (/auto/frame)
# The board that places the front camera: the floor board, close to the wrist camera and seen
# from every arm pose. The upright boards stand ~65 cm out, located only from the forward poses:
# through them the front camera came out up to 20 mm and 3 deg apart (Oct 29 scene).
FRONT_VIA = "wrist"
# A board view counts for the turning centre from this many corners: a partial view of an
# upright board 60 cm away (fewer) moved the estimate by a few mm (convention).
MIN_TURN_CORNERS = 12
# The joints whose zero offsets the scene can tell: they tilt the gripper.
OFFSET_JOINTS = ["arm_shoulder_lift", "arm_elbow_flex", "arm_wrist_flex"]


# --- poses ---


def to_T(rvec, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def from_T(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def fk(joints_deg: dict[str, float], offsets_deg: dict[str, float] | None = None) -> np.ndarray:
    """The TCP's pose in the arm's frame for joint angles in degrees (plus zero offsets)."""
    off = offsets_deg or {}
    return kin.forward_kinematics(np.radians([joints_deg[j] + off.get(j, 0.0) for j in IK_ARM_JOINTS]))


# --- 1. lenses ---


def lens(views: list[dict], image_size: tuple[int, int]) -> dict:
    """Intrinsics from views [{board: {"ids", "px"}}]: each board in each view is one planar
    view for calibrateCamera. Returns {"K", "dist", "rms_px", "views"}."""
    obj, img = [], []
    for v in views:
        for name, d in v.items():
            o = boards.object_points(name, d["ids"]).astype(np.float32)
            # A view of one row of corners isn't a plane: calibrateCamera can't start from it
            # (Oct 30: two 7-corner wrist views, OpenCV's initIntrinsicParams2D assertion).
            if np.linalg.matrix_rank(o[:, :2] - o[:, :2].mean(0), tol=1e-6) < 2:
                continue
            obj.append(o)
            img.append(np.asarray(d["px"], np.float32).reshape(-1, 1, 2))
    if len(obj) < 3:  # convention: calibrateCamera needs a few views to separate focal length from pose
        raise ValueError(f"only {len(obj)} board views: capture more")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None, None, flags=LENS_FLAGS)
    return {"K": K, "dist": dist.ravel(), "rms_px": float(rms), "views": len(obj)}


def board_in_camera(K, dist, name: str, d: dict) -> np.ndarray | None:
    """The board's pose in the camera (T_cam_board) from one detection, or None."""
    ok, rvec, tvec = cv2.solvePnP(boards.object_points(name, d["ids"]), np.asarray(d["px"], float), K, dist)
    return to_T(rvec, tvec) if ok else None


# --- 2. the wrist camera, the boards and the joint offsets ---


def _log_rot(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R)[0].ravel()


def hand_eye(G: list[np.ndarray], C: list[np.ndarray]) -> np.ndarray:
    """X = the camera's pose on the gripper (T_gripper_cam) from gripper poses G (T_base_gripper)
    and one fixed target's poses C (T_cam_target), Park and Martin: G_i X C_i is the same for
    all i, so A X = X B with A = G_j^-1 G_i and B = C_j C_i^-1 over pairs."""
    A, B = [], []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            A.append(inv(G[j]) @ G[i])
            B.append(C[j] @ inv(C[i]))
    M = sum(np.outer(_log_rot(b[:3, :3]), _log_rot(a[:3, :3])) for a, b in zip(A, B, strict=True))
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    rhs = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B, strict=True)])
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return X


def _residuals(p, obs, K, dist, names, offset_joints):
    X = to_T(p[:3], p[3:6])
    B = {n: to_T(p[6 + 6 * k : 9 + 6 * k], p[9 + 6 * k : 12 + 6 * k]) for k, n in enumerate(names)}
    off = dict(zip(offset_joints, np.degrees(p[6 + 6 * len(names) :]), strict=True))
    out = []
    for o in obs:
        T_cam_board = inv(X) @ inv(fk(o["joints"], off)) @ B[o["board"]]
        proj, _ = cv2.projectPoints(
            o["obj"], cv2.Rodrigues(T_cam_board[:3, :3])[0], T_cam_board[:3, 3], K, dist
        )
        out.append((proj.reshape(-1, 2) - o["px"]).ravel())
    return np.concatenate(out)


def _lm(f, p, iters: int = 60):  # convention: Levenberg-Marquardt, enough for ~25 unknowns
    lam = 1e-3
    r = f(p)
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            dp = np.zeros_like(p)
            dp[k] = 1e-6
            J[:, k] = (f(p + dp) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            r_new = f(p + step)
            if r_new @ r_new < r @ r:
                p, r, lam = p + step, r_new, lam / 10
                break
            lam *= 10
            if lam > 1e12:
                return p, r
        if np.linalg.norm(step) < 1e-10:
            break
    return p, r


def solve_wrist(records: list[dict], K, dist, offset_joints=OFFSET_JOINTS) -> dict:
    """records: [{"joints": {joint: deg}, "boards": {name: {"ids", "px"}}}] from the scene's arm
    poses. Returns the camera's pose on the gripper, the boards' poses in the arm's frame, the
    joint offsets (deg) and the reprojection error."""
    obs, per_board = [], {}
    for rec in records:
        for name, d in rec["boards"].items():
            C = board_in_camera(K, dist, name, d)
            if C is None:
                continue
            per_board.setdefault(name, []).append((fk(rec["joints"]), C))
            obs.append(
                {
                    "joints": rec["joints"],
                    "board": name,
                    "obj": boards.object_points(name, d["ids"]),
                    "px": np.asarray(d["px"], float),
                }
            )
    # The hand-eye start from the board seen in most poses (at least 3: two motions).
    best = max(per_board, key=lambda n: len(per_board[n]))
    if len(per_board[best]) < 3:  # convention: AX = XB needs two independent motions
        raise ValueError(f"the most-seen board ({best}) is in only {len(per_board[best])} poses")
    X = hand_eye([g for g, _ in per_board[best]], [c for _, c in per_board[best]])
    names = sorted(per_board)
    B0 = {}
    for n in names:  # each board's pose: the mean over its views (rotation by its first view)
        Ts = [g @ X @ c for g, c in per_board[n]]
        T = Ts[0].copy()
        T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
        B0[n] = T
    p0 = np.concatenate([from_T(X)] + [from_T(B0[n]) for n in names] + [np.zeros(len(offset_joints))])
    p, r = _lm(lambda q: _residuals(q, obs, K, dist, names, offset_joints), p0)
    rms = float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2))
    k = 6 + 6 * len(names)
    return {
        "T_tcp_cam": to_T(p[:3], p[3:6]),
        "boards": {n: to_T(p[6 + 6 * i : 9 + 6 * i], p[9 + 6 * i : 12 + 6 * i]) for i, n in enumerate(names)},
        "joint_offsets_deg": dict(zip(offset_joints, np.degrees(p[k:]).tolist(), strict=True)),
        "rms_px": rms,
        "poses": len(records),
        "corners": int(sum(len(o["px"]) for o in obs)),
    }


def offset_spread(records: list[dict], K, dist) -> dict | None:
    """The joint offsets solved again with each anchor's poses left out in turn, and their spread
    per joint: how much the offsets depend on which arm shapes were seen. The error the model
    lacks (backlash, sag, link lengths) is pose-dependent and the offsets soak some of it up, so
    this spread, not the reprojection noise, says how far to trust them. None without anchors."""
    anchors = sorted({r["anchor"] for r in records if "anchor" in r})
    if len(anchors) < 3:
        return None
    without = {
        a: solve_wrist([r for r in records if r.get("anchor") != a], K, dist)["joint_offsets_deg"]
        for a in anchors
    }
    return {
        "without": without,
        "spread_deg": {
            j: max(o[j] for o in without.values()) - min(o[j] for o in without.values())
            for j in OFFSET_JOINTS
        },
    }


def offsets_to_apply(offsets_deg: dict[str, float], spread_deg: dict[str, float]) -> dict[str, float]:
    """The joint zero offsets worth writing into the robot: those larger than their spread over
    the arm shapes left out in turn, i.e. larger than this method can resolve on this arm (Oct 29,
    zeros set by eye: offsets <= 1.1 deg, spreads 1.1-2.7 deg, none applied; a zero set 5-10 deg
    off is found within 0.01 deg and applied)."""
    return {j: v for j, v in offsets_deg.items() if abs(v) > spread_deg[j]}


# --- 3. the front camera ---


def mean_pose(Ts: list[np.ndarray]) -> np.ndarray:
    """The mean of poses: translations averaged, rotations by the nearest rotation to their sum."""
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def solve_front(views: list[dict], K, dist, boards_in_base: dict[str, np.ndarray]) -> dict:
    """The front camera's pose in the arm's frame (T_base_cam) from its views of boards whose
    pose in the arm's frame is known: one estimate per board view, averaged."""
    est = []
    for v in views:
        for name, d in v.items():
            if name in boards_in_base:
                C = board_in_camera(K, dist, name, d)
                if C is not None:
                    est.append(boards_in_base[name] @ inv(C))
    if not est:
        raise ValueError("the front camera saw none of the boards the wrist camera located")
    T = mean_pose(est)
    spread = float(np.max([np.linalg.norm(e[:3, 3] - T[:3, 3]) for e in est]))
    return {"T_base_cam": T, "views": len(est), "spread_m": spread}


def base_offset(T_cam_board_ref, T_cam_board_now, T_base_cam) -> tuple[float, float, float]:
    """Where the base is now against where it was in the reference frame, in the reference
    base frame: (x m, y m, heading deg). The board is fixed and the camera rides the base."""
    move = T_base_cam @ T_cam_board_ref @ inv(T_cam_board_now) @ inv(T_base_cam)
    return float(move[0, 3]), float(move[1, 3]), float(np.degrees(np.arctan2(move[1, 0], move[0, 0])))


def turn_centre(scenes: list[dict], K, dist, T_base_cam) -> tuple[np.ndarray, float]:
    """The point the base turns about, (x, y) in the arm's frame, from the base sweeps' turns in
    place (0 cm back): each turn moves the front camera along an arc about it, measured by the
    boards against the scene's first frame; least squares over all. Returns it and the median
    residual (m). Oct 29: 4.6 cm behind the arm frame's origin, centred, 0.2 cm residual over 20
    turns; it puts the front camera 12.8 cm ahead of the turn, the parallax that made a fitted
    image-widths-per-degree 23% larger than the lens alone gives."""
    A, b = [], []
    for sc in scenes:
        if not sc["front"]:
            continue
        ref = {n: board_in_camera(K, dist, n, d) for n, d in sc["front"][0]["boards"].items()}
        for fr in sc.get("sweep", []):
            if fr["base"]["back_m"] != 0.0:
                continue
            est = [
                base_offset(ref[n], board_in_camera(K, dist, n, d), T_base_cam)
                for n, d in fr["boards"].items()
                if n in ref and len(d["ids"]) >= MIN_TURN_CORNERS
            ]
            if not est:
                continue
            x, y, th = np.median(np.asarray(est), axis=0)
            a = np.radians(th)
            A.append(np.eye(2) - np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]))
            b.append([x, y])
    if len(A) < 2:  # convention: two turns pin a point in the plane
        raise ValueError("no base sweep turns to find the turning centre from")
    c, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    resid = np.linalg.norm((np.vstack(A) @ c - np.concatenate(b)).reshape(-1, 2), axis=1)
    return c, float(np.median(resid))


# --- the captures ---


def _load(folder: Path, key: str) -> dict:
    return json.loads((folder / f"{key}.json").read_text())


def solve(
    scene_dirs: list[Path],
    closeup_dirs: list[Path] = (),
    lens_wrist: Path | None = None,
    lens_front: Path | None = None,
    fit_offsets: bool = False,
    floor_board_thickness: float = 0.0,
    write: bool = False,
    front_extra: list[Path] = (),
) -> dict:
    """The solve, printed as it goes; returns the zero offsets (and which to apply), the floor
    board's top and pose in the arm's frame, and the fit errors (calibration/calibrate_all.py).
    front_extra: scenes whose front frames (start, base sweep, close-up) only add to the front
    lens and the turning centre: the front camera's views don't depend on the arm's zeros, so a
    sweep taken before a zero correction still counts (calibrate_all.py sweeps once a run)."""

    scenes = [_load(d, "scene") for d in scene_dirs]
    extra = [_load(d, "scene") for d in front_extra]
    held = {
        cam: _load(d, "views")["views"] if d else []
        for cam, d in (("wrist", lens_wrist), ("front", lens_front))
    }
    # Each scene's boards are its own (placed, or the robot returned, a little differently):
    # their names carry the scene's index, "2/wrist".
    records = [
        {**r, "boards": {f"{i}/{n}": d for n, d in r["boards"].items()}}
        for i, sc in enumerate(scenes)
        for r in sc["wrist"]
    ]
    closeups = [f["boards"] for d in closeup_dirs for f in _load(d, "closeup")["closeup"]]
    wrist_views = [v["boards"] for v in held["wrist"]] + [r["boards"] for r in records]
    front_views = (
        [v["boards"] for v in held["front"]]
        + [
            f["boards"]
            for sc in scenes + extra
            for f in sc["front"] + sc.get("sweep", []) + sc.get("closeup", [])
        ]
        + closeups
    )
    wl = lens(wrist_views, IMAGE_SIZE)
    fl = lens(front_views, IMAGE_SIZE)
    print(
        f"wrist lens: rms {wl['rms_px']:.2f} px over {wl['views']} views; front lens: rms {fl['rms_px']:.2f} px over {fl['views']}"
    )
    offset_joints = OFFSET_JOINTS if fit_offsets else []
    w = solve_wrist(records, wl["K"], wl["dist"], offset_joints=offset_joints)
    print(
        f"wrist camera on the gripper: {np.round(w['T_tcp_cam'][:3, 3] * 1000, 1).tolist()} mm; "
        f"reprojection rms {w['rms_px']:.2f} px over {w['poses']} poses in {len(scenes)} scenes, {w['corners']} corners"
    )
    spread = None
    if fit_offsets:
        print("joint zero offsets (deg):", {j: round(v, 2) for j, v in w["joint_offsets_deg"].items()})
        spread = offset_spread(records, wl["K"], wl["dist"])
        if spread:
            print("  each anchor left out in turn:")
            for anchor, off in spread["without"].items():
                print(f"    without {anchor:12s}", {j: round(v, 2) for j, v in off.items()})
            print("  spread (max - min, deg):", {j: round(v, 2) for j, v in spread["spread_deg"].items()})
            apply = offsets_to_apply(w["joint_offsets_deg"], spread["spread_deg"])
            print(
                "  to apply (larger than their spread):",
                {j: round(v, 2) for j, v in apply.items()}
                or "none: the arm's zeros are as good as this can tell",
            )
    # The front camera: one estimate per scene, through that scene's floor board, then their mean.
    # A scene whose first front frames missed the floor board (Oct 30: it lay at the view's edge
    # and the robot came back 1.6 deg turned) places nothing: the others do.
    fronts = []
    for i, sc in enumerate(scenes):
        try:
            fronts.append(
                solve_front(
                    [fr["boards"] for fr in sc["front"]],
                    fl["K"],
                    fl["dist"],
                    {FRONT_VIA: w["boards"][f"{i}/{FRONT_VIA}"]},
                )["T_base_cam"]
            )
        except ValueError as e:
            print(f"scene {scene_dirs[i]}: {e}: not used for the front camera")
    if not fronts:
        raise ValueError(
            f"no scene's front frames saw the {FRONT_VIA} board: the front camera can't be placed"
        )
    f = {"T_base_cam": mean_pose(fronts)}
    tilt = float(np.degrees(np.arcsin(-f["T_base_cam"][2, 2])))
    across = max(np.linalg.norm(T[:3, 3] - f["T_base_cam"][:3, 3]) for T in fronts)
    print(
        f"front camera in the arm's frame: {np.round(f['T_base_cam'][:3, 3] * 1000, 1).tolist()} mm, "
        f"{tilt:.1f} deg down (through the {FRONT_VIA} board; the scenes' estimates within {1000 * across:.1f} mm "
        f"of their mean, tilts {min(np.degrees(np.arcsin(-T[2, 2])) for T in fronts):.1f}-"
        f"{max(np.degrees(np.arcsin(-T[2, 2])) for T in fronts):.1f} deg)"
    )
    centre, centre_rms = turn_centre(scenes + extra, fl["K"], fl["dist"], f["T_base_cam"])
    board_top = float(np.mean([w["boards"][f"{i}/{FRONT_VIA}"][2, 3] for i in range(len(scenes))]))
    floor_seen = board_top - floor_board_thickness
    print(
        f"the floor board's top lies at z {1000 * board_top:.1f} mm in the arm's frame; the floor under it "
        f"{1000 * floor_seen:.1f} mm (the board {1000 * floor_board_thickness:.2f} mm thick)"
    )
    print(
        f"the base turns about ({100 * centre[0]:.1f}, {100 * centre[1]:.1f}) cm in the arm's frame (residual {100 * centre_rms:.2f} cm)"
    )
    ahead, left, below = f["T_base_cam"][:3, 3] - PAN_AXIS_BASE
    print(
        f"  i.e. {100 * ahead:.1f} cm ahead of the arm's pan axis, {100 * left:.1f} cm to its left, "
        f"{-100 * f['T_base_cam'][2, 3]:.1f} cm below the arm's mounting surface"
    )
    if write:
        calib = json.loads(CALIB.read_text())
        calib["cameras"] = {
            "value": {
                "wrist": {
                    "K": wl["K"].tolist(),
                    "dist": wl["dist"].tolist(),
                    "T_tcp_cam": w["T_tcp_cam"].tolist(),
                },
                "front": {
                    "K": fl["K"].tolist(),
                    "dist": fl["dist"].tolist(),
                    "T_base_cam": f["T_base_cam"].tolist(),
                },
                "joint_offsets_deg": w["joint_offsets_deg"],
                "base_turn_centre": centre.tolist(),
                "floor_z": floor_seen,
                "floor_board_thickness_m": floor_board_thickness,
            },
            "meaning": "camera intrinsics (K, dist: OpenCV), the wrist camera's pose in the TCP frame and the front "
            "camera's pose in the arm base frame (metres); base_turn_centre: (x, y) the base turns about (m); "
            "floor_z: the floor under the floor board the cameras were placed against (its top less "
            "floor_board_thickness_m, a ruler's) (m); "
            "joint_offsets_deg: the zero offsets the solve assumed "
            "(deg, added to the measured angle; empty = none, the arm's own zeros)",
            "method": "solve_cameras.py on capture_cameras.py scenes: calibrateCamera, radial k1 k2 only (the front "
            "lens with the base sweep and the side board close-up); Park-Martin hand-eye refined with every "
            "corner's reprojection over all scenes, each with its own board poses; the front camera through each "
            f"scene's {FRONT_VIA} board, averaged",
            "date": datetime.date.today().isoformat(),
            "log": [
                str(p.resolve().relative_to(ROOT))
                for p in (lens_wrist, lens_front, *scene_dirs, *closeup_dirs)
                if p
            ],
            "rms_px": {"wrist_lens": wl["rms_px"], "front_lens": fl["rms_px"], "wrist_scene": w["rms_px"]},
            "offset_spread_deg": spread["spread_deg"] if spread else None,
            "offsets_to_apply_deg": offsets_to_apply(w["joint_offsets_deg"], spread["spread_deg"])
            if spread
            else {},
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")
    return {
        "joint_offsets_deg": w["joint_offsets_deg"],
        "spread_deg": spread["spread_deg"] if spread else None,
        "to_apply_deg": offsets_to_apply(w["joint_offsets_deg"], spread["spread_deg"]) if spread else {},
        "board_top_z": board_top,
        "floor_z": floor_seen,
        "floor_board_pose": [w["boards"][f"{i}/{FRONT_VIA}"].tolist() for i in range(len(scenes))],
        "rms_px": {"wrist_lens": wl["rms_px"], "front_lens": fl["rms_px"], "wrist_scene": w["rms_px"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-wrist", type=Path, help="a hand-held lens capture (optional)")
    ap.add_argument("--lens-front", type=Path, help="a hand-held lens capture (optional)")
    ap.add_argument("--scene", type=Path, nargs="+", required=True, help="one or more scene captures")
    ap.add_argument("--closeup", type=Path, nargs="*", default=[], help="closeup captures (front lens views)")
    ap.add_argument(
        "--fit-offsets", action="store_true", help="fit the joint zero offsets too (reported, not trusted)"
    )
    ap.add_argument(
        "--floor-board-thickness", type=float, default=0.0, help="the floor board's thickness (m, a ruler's)"
    )
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    solve(
        args.scene,
        closeup_dirs=args.closeup,
        lens_wrist=args.lens_wrist,
        lens_front=args.lens_front,
        fit_offsets=args.fit_offsets,
        floor_board_thickness=args.floor_board_thickness,
        write=args.write,
    )


if __name__ == "__main__":
    main()
