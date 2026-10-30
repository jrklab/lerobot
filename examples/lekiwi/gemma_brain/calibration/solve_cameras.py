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

    .venv/bin/python calibration/solve_cameras.py --lens-wrist runs/<t>-lens-wrist \\
        --lens-front runs/<t>-lens-front --scene runs/<t>-scene [--write]

1. Each lens (focal length, optical centre, distortion): OpenCV's calibrateCamera over every
   view of a board, the hand-held ones and the scene's.
2. The wrist camera's pose on the gripper, the boards' poses in the arm's frame, and the zero
   offsets of the three joints that tilt the gripper (shoulder lift, elbow, wrist flex): a
   linear hand-eye solution (Park and Martin, AX = XB) to start from, then a least-squares fit
   of every detected corner's reprojection. The shoulder pan's and wrist roll's offsets can't
   be told apart from where the boards lie and how the camera is mounted, so they stay 0.
3. The front camera's pose in the arm's frame: from its view of the upright front board, whose
   pose in the arm's frame step 2 found (the arm folded out of the view).

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
            obj.append(boards.object_points(name, d["ids"]).astype(np.float32))
            img.append(np.asarray(d["px"], np.float32).reshape(-1, 1, 2))
    if len(obj) < 3:  # convention: calibrateCamera needs a few views to separate focal length from pose
        raise ValueError(f"only {len(obj)} board views: capture more")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None, None)
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


# --- 3. the front camera ---


def solve_front(views: list[dict], K, dist, boards_in_base: dict[str, np.ndarray]) -> dict:
    """The front camera's pose in the arm's frame (T_base_cam) from its views of boards whose
    pose in the arm's frame is known: one estimate per board view, averaged (rotation by the
    first, which is enough for views from one place)."""
    est = []
    for v in views:
        for name, d in v.items():
            if name in boards_in_base:
                C = board_in_camera(K, dist, name, d)
                if C is not None:
                    est.append(boards_in_base[name] @ inv(C))
    if not est:
        raise ValueError("the front camera saw none of the boards the wrist camera located")
    T = est[0].copy()
    T[:3, 3] = np.mean([e[:3, 3] for e in est], axis=0)
    spread = float(np.max([np.linalg.norm(e[:3, 3] - T[:3, 3]) for e in est]))
    return {"T_base_cam": T, "views": len(est), "spread_m": spread}


# --- the captures ---


def _load(folder: Path, key: str) -> dict:
    return json.loads((folder / f"{key}.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-wrist", type=Path, required=True)
    ap.add_argument("--lens-front", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    lw, lf, sc = _load(args.lens_wrist, "views"), _load(args.lens_front, "views"), _load(args.scene, "scene")
    wrist_views = [v["boards"] for v in lw["views"]] + [r["boards"] for r in sc["wrist"]]
    front_views = [v["boards"] for v in lf["views"]] + [f["boards"] for f in sc["front"]]
    wl = lens(wrist_views, lw["image_size"])
    fl = lens(front_views, lf["image_size"])
    print(
        f"wrist lens: rms {wl['rms_px']:.2f} px over {wl['views']} views; front lens: rms {fl['rms_px']:.2f} px over {fl['views']}"
    )
    w = solve_wrist(sc["wrist"], wl["K"], wl["dist"])
    print(
        f"wrist camera on the gripper: {np.round(w['T_tcp_cam'][:3, 3] * 1000, 1).tolist()} mm; "
        f"reprojection rms {w['rms_px']:.2f} px over {w['poses']} poses, {w['corners']} corners"
    )
    print("joint zero offsets (deg):", {j: round(v, 2) for j, v in w["joint_offsets_deg"].items()})
    f = solve_front([fr["boards"] for fr in sc["front"]], fl["K"], fl["dist"], w["boards"])
    print(
        f"front camera in the arm's frame: {np.round(f['T_base_cam'][:3, 3] * 1000, 1).tolist()} mm "
        f"({f['views']} views, spread {1000 * f['spread_m']:.1f} mm)"
    )
    if args.write:
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
                "boards_in_base": {n: T.tolist() for n, T in w["boards"].items()},
                "joint_offsets_deg": w["joint_offsets_deg"],
            },
            "meaning": "camera intrinsics (K, dist: OpenCV), the wrist camera's pose in the TCP frame, the front camera's "
            "pose in the arm base frame, and the zero offsets of the gripper-tilting joints (deg, added to the "
            "measured angle)",
            "method": "solve_cameras.py: calibrateCamera; Park-Martin hand-eye refined with every corner's reprojection",
            "date": datetime.date.today().isoformat(),
            "log": [str(p.relative_to(ROOT)) for p in (args.lens_wrist, args.lens_front, args.scene)],
            "rms_px": {"wrist_lens": wl["rms_px"], "front_lens": fl["rms_px"], "wrist_scene": w["rms_px"]},
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")


if __name__ == "__main__":
    main()
