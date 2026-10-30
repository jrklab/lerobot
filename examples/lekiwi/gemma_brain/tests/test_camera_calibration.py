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

# ruff: noqa: N802, N803, N806
# (N8: transforms and camera matrices keep their textbook capitals: T, R, K, the hand-eye G, C, X)

"""Offline checks for the camera calibration (calibration/boards.py, solve_cameras.py): the
boards detected together, and the solve recovering known cameras and joint offsets from
synthetic views."""

import json

import cv2
import numpy as np
from calibration import boards, capture_cameras, solve_cameras as sc
from common.arm import IK_ARM_JOINTS, Calibration, kin
from helpers import _real_cal


def _scene_with_all_boards():
    """A 1280x720 grey image with the three boards, each warped in perspective."""
    img = np.full((720, 1280), 200, np.uint8)
    spots = {
        "wrist": [(40, 60), (560, 90), (540, 430), (60, 400)],
        "front": [(640, 80), (900, 60), (920, 330), (650, 300)],
        "side": [(950, 380), (1230, 420), (1200, 690), (970, 650)],
    }
    for name, quad in spots.items():
        b = boards.BOARDS[name]
        sx, sy = b.getChessboardSize()
        flat = b.generateImage((sx * 60, sy * 60), marginSize=20, borderBits=1)
        h, w = flat.shape
        H = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), np.float32(quad))
        warped = cv2.warpPerspective(flat, H, (1280, 720), borderValue=0)
        mask = cv2.warpPerspective(np.full_like(flat, 255), H, (1280, 720)) > 0
        img[mask] = warped[mask]
    return img


def test_the_three_boards_are_found_together_each_with_its_own_corners():
    found = boards.detect(_scene_with_all_boards())
    assert set(found) == {"wrist", "front", "side"}, found.keys()
    for name, n in (("wrist", 54), ("front", 24), ("side", 24)):
        assert len(found[name]["ids"]) >= 0.8 * n, (name, len(found[name]["ids"]))
    # object points: metres in the board's frame, the square size apart
    pts = boards.object_points("front", [0, 1])
    assert abs(np.linalg.norm(pts[1] - pts[0]) - 0.035) < 1e-9


# --- the solve, on a synthetic robot with known cameras, boards and joint zero errors ---


FLOOR_Z = -0.112  # the jaw tips' floor in the arm's frame (Oct 29)
TRUE_OFFSETS = {"arm_shoulder_lift": 1.5, "arm_elbow_flex": -1.2, "arm_wrist_flex": 0.8}
K_WRIST = np.array([[480.0, 0, 322], [0, 482.0, 236], [0, 0, 1]])
DIST_WRIST = np.array([-0.08, 0.03, 0.001, -0.001, 0.0])
K_FRONT = np.array([[500.0, 0, 318], [0, 498.0, 230], [0, 0, 1]])
DIST_FRONT = np.array([-0.2, 0.06, 0.0, 0.0, 0.0])


def _rot(axis, a):
    return cv2.Rodrigues(np.asarray(axis, float) * a)[0]


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# The wrist camera 6 cm back from the TCP along the approach axis, 3 cm off it, looking along it,
# tilted 8 deg; the floor board flat, printed side up, around the reach point; the front board
# upright 42 cm out, facing the robot; the front camera 4 cm above the floor, 6.7 deg down.
X_TRUE = _T(_rot([1, 0, 0], np.radians(8)) @ _rot([0, 0, 1], np.radians(3)), [0.03, 0.002, -0.06])
BOARDS_TRUE = {
    "wrist": _T(np.eye(3), [0.14, -0.075, FLOOR_Z]),
    "front": _T(np.column_stack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]), [0.42, -0.12, FLOOR_Z + 0.19]),
}
_FWD = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # camera x right, y down, z forward
FRONT_TRUE = _T(_FWD @ _rot([1, 0, 0], np.radians(-6.7)), [0.08, 0.0, FLOOR_Z + 0.04])


def _project(T_cam_board, name, K, dist, rng, size=(640, 480)):
    obj = np.asarray(boards.BOARDS[name].getChessboardCorners(), float)
    cam = (T_cam_board[:3, :3] @ obj.T).T + T_cam_board[:3, 3]
    px, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_cam_board[:3, :3])[0], T_cam_board[:3, 3], K, dist)
    px = px.reshape(-1, 2) + rng.normal(0, 0.3, (len(obj), 2))
    keep = (
        (cam[:, 2] > 0.02)
        & (px[:, 0] > 5)
        & (px[:, 0] < size[0] - 5)
        & (px[:, 1] > 5)
        & (px[:, 1] < size[1] - 5)
    )
    ids = np.flatnonzero(keep)
    return {"ids": ids.tolist(), "px": px[keep].tolist()} if len(ids) >= boards.MIN_CORNERS else None


class _Oct29(Calibration):
    """The Oct 28 fixture's mapping with the Oct 29 recalibration's wider wrist and elbow limits."""

    def limits(self, joint):
        return {"arm_wrist_flex": (-95.0, 95.0), "arm_elbow_flex": (-96.8, 95.1)}.get(
            joint
        ) or super().limits(joint)


def _scene(rng):
    """The capture's own plan on a true arm: the planned joints are where the arm truly is, the
    measured angles are off by the joint zero errors."""
    poses = capture_cameras.plan(_Oct29(*_real_cal()), floor_z=FLOOR_Z)
    assert all("joints" in p for p in poses), [p for p in poses if "refused" in p]
    records = []
    for p in poses:
        G = kin.forward_kinematics(np.radians([p["joints"][j] for j in IK_ARM_JOINTS]))
        seen = {}
        for name, B in BOARDS_TRUE.items():
            d = _project(sc.inv(X_TRUE) @ sc.inv(G) @ B, name, K_WRIST, DIST_WRIST, rng)
            if d:
                seen[name] = d
        if seen:
            records.append(
                {
                    "joints": {j: v - TRUE_OFFSETS.get(j, 0.0) for j, v in p["joints"].items()},
                    "boards": seen,
                    "anchor": p["anchor"],
                }
            )
    return records


def test_the_solve_recovers_the_cameras_and_the_joint_zero_errors():
    rng = np.random.default_rng(0)
    records = _scene(rng)
    assert len(records) >= 12 and any("front" in r["boards"] for r in records)
    # the wrist lens from the scene's own views plus hand-held ones
    held = [
        {"wrist": d}
        for d in (
            _project(
                _T(_rot([1, 0.3, 0], a), [-0.1 + 0.02 * k, -0.08, 0.3 + 0.03 * k]),
                "wrist",
                K_WRIST,
                DIST_WRIST,
                rng,
            )
            for k, a in enumerate(np.linspace(-0.5, 0.5, 9))
        )
        if d
    ]
    wl = sc.lens([r["boards"] for r in records] + held, (640, 480))
    assert wl["rms_px"] < 0.6 and abs(wl["K"][0, 0] - 480) < 3, wl
    w = sc.solve_wrist(records, wl["K"], wl["dist"])
    assert w["rms_px"] < 0.6, w["rms_px"]
    for j, v in TRUE_OFFSETS.items():
        assert abs(w["joint_offsets_deg"][j] - v) < 0.2, (j, w["joint_offsets_deg"])
    assert np.linalg.norm(w["T_tcp_cam"][:3, 3] - X_TRUE[:3, 3]) < 0.003
    dR = w["T_tcp_cam"][:3, :3].T @ X_TRUE[:3, :3]
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0])) < 0.5
    # the front camera, through the upright board the wrist camera located
    fviews = [
        {"front": _project(sc.inv(FRONT_TRUE) @ BOARDS_TRUE["front"], "front", K_FRONT, DIST_FRONT, rng)}
        for _ in range(3)
    ]
    assert all(v["front"] for v in fviews), "the front board must be in the front camera's view"
    f = sc.solve_front(fviews, K_FRONT, DIST_FRONT, w["boards"])
    assert np.linalg.norm(f["T_base_cam"][:3, 3] - FRONT_TRUE[:3, 3]) < 0.005, f["T_base_cam"][:3, 3]
    dR = f["T_base_cam"][:3, :3].T @ FRONT_TRUE[:3, :3]
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0])) < 0.5


def test_hand_eye_is_exact_without_noise():
    G, C = [], []
    for k in range(5):
        axis = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)][k]
        Gk = _T(_rot(axis, 0.3 + 0.1 * k), [0.2 + 0.02 * k, 0.01 * k, 0.1 - 0.01 * k])
        G.append(Gk)
        C.append(sc.inv(X_TRUE) @ sc.inv(Gk) @ BOARDS_TRUE["wrist"])
    X = sc.hand_eye(G, C)
    assert np.allclose(X, X_TRUE, atol=1e-6)


def test_the_lens_capture_keeps_a_still_view_once(tmp_path, monkeypatch):
    """Two identical frames make a still view, kept once; the same view again isn't new."""
    jpeg = cv2.imencode(".jpg", _scene_with_all_boards())[1].tobytes()

    class Camera:
        frames = 0

        def frame(self, cam, after=0.0):
            self.frames += 1
            if self.frames > 6:
                raise KeyboardInterrupt  # the operator's Ctrl-C
            return jpeg, after

    monkeypatch.setattr(capture_cameras, "ROOT", tmp_path)
    out = capture_cameras.capture_lens(Camera(), "wrist", n_views=5)
    views = json.loads((out / "views.json").read_text())
    assert views["image_size"] == [1280, 720] and len(views["views"]) == 1
    assert set(views["views"][0]["boards"]) == {"wrist", "front", "side"}
    assert (out / views["views"][0]["frame"]).exists()


def test_the_offsets_hold_with_any_anchor_left_out():
    """On a model without pose-dependent error the leave-one-anchor-out spread is near zero; on
    the robot it measures how far the offsets can be trusted."""
    records = _scene(np.random.default_rng(1))
    assert len({r["anchor"] for r in records}) == 6
    spread = sc.offset_spread(records, K_WRIST, DIST_WRIST)
    assert all(v < 0.3 for v in spread["spread_deg"].values()), spread["spread_deg"]


def test_the_base_sweep_never_drives_forward_and_frames_every_stop(tmp_path, monkeypatch):
    jpeg = cv2.imencode(".jpg", _scene_with_all_boards())[1].tobytes()

    class Base:
        moves = []

        def base(self, dx=0.0, dy=0.0, dtheta_deg=0.0, speed=None, theta_speed=None):
            self.moves.append((dx, dtheta_deg))

        def frame(self, cam, after=0.0):
            assert cam == "front"
            return jpeg, after

    monkeypatch.setattr(capture_cameras, "SETTLE_S", 0.0)
    robot = Base()
    frames = capture_cameras.capture_sweep(robot, tmp_path, capture_cameras.sweep_plan())
    assert all(dx <= 0 for dx, _ in robot.moves)
    assert abs(sum(dx for dx, _ in robot.moves) + 0.20) < 1e-9 and abs(sum(t for _, t in robot.moves)) < 1e-9
    assert all(abs(t) <= 20 for _, t in robot.moves)  # no turn larger than the sweep's widest
    assert len(frames) == 15 and len({(f["base"]["back_m"], f["base"]["heading_deg"]) for f in frames}) == 15


def _base_move(x, y, th_deg):
    return _T(_rot([0, 0, 1], np.radians(th_deg)), [x, y, 0.0])


def test_the_base_offset_from_the_front_camera_is_the_move_it_made():
    ref = sc.inv(FRONT_TRUE) @ BOARDS_TRUE["wrist"]
    for x, y, th in ((-0.2, 0.0, 0.0), (-0.1, 0.03, 12.0), (0.01, -0.02, -20.0)):
        now = sc.inv(FRONT_TRUE) @ sc.inv(_base_move(x, y, th)) @ BOARDS_TRUE["wrist"]
        got = capture_cameras.base_offset(ref, now, FRONT_TRUE)
        assert np.allclose(got, (x, y, th), atol=1e-9), got


def test_the_return_reaches_the_start_though_the_base_moves_8_percent_short():
    class Base:
        pose = _base_move(-0.20, 0.015, 4.0)  # after the sweep, odometry drift included
        forward = []

        def base(self, dx=0.0, dy=0.0, dtheta_deg=0.0, speed=None, theta_speed=None):
            self.forward.append(dx)
            self.pose = self.pose @ _base_move(0.92 * dx, 0.92 * dy, 0.92 * dtheta_deg)

    robot = Base()

    def measure():
        p = robot.pose
        return p[0, 3], p[1, 3], float(np.degrees(np.arctan2(p[1, 0], p[0, 0])))

    x, y, th = capture_cameras.return_to_start(robot, measure)
    assert np.hypot(x, y) < capture_cameras.RETURN_TOL_M and abs(th) < capture_cameras.RETURN_TOL_DEG
    assert max(robot.forward) <= capture_cameras.MAX_RETURN_FORWARD_M


def test_the_reversed_plan_visits_the_same_poses_backwards():
    cal = _Oct29(*_real_cal())
    ahead, back = (
        capture_cameras.plan(cal, floor_z=FLOOR_Z),
        capture_cameras.plan(cal, floor_z=FLOOR_Z, reverse=True),
    )
    assert all("joints" in p for p in back), [p for p in back if "refused" in p]
    assert [p["name"] for p in back] == [p["name"] for p in reversed(ahead)]


class _SideWorld:
    """A base that moves 8% short, beside a board 53 cm off to one side (y), seen when within
    35 deg of straight ahead; board_ahead() measures it from the base's simulated pose."""

    def __init__(self, side_y):
        self.pose, self.board, self.closest = _base_move(0, 0, 0), np.array([0.0, side_y, 0.0]), np.inf
        self.jpeg = cv2.imencode(".jpg", _scene_with_all_boards())[1].tobytes()

    def base(self, dx=0.0, dy=0.0, dtheta_deg=0.0, speed=None, theta_speed=None):
        self.pose = self.pose @ _base_move(0.92 * dx, 0.92 * dy, 0.92 * dtheta_deg)

    def frame(self, cam, after=0.0):
        return self.jpeg, after

    def ahead(self, robot, cams, name="side"):
        if self.board is None:
            return None
        v = sc.inv(self.pose) @ np.append(self.board, 1.0)
        if abs(np.degrees(np.arctan2(v[1], v[0]))) > 35:
            return None
        self.closest = min(self.closest, v[0])
        return float(v[0])


def test_the_closeup_finds_the_side_board_on_either_side_and_comes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_cameras, "SETTLE_S", 0.0)
    for side_y, turn in ((-0.53, -90.0), (0.53, 90.0)):
        world = _SideWorld(side_y)
        monkeypatch.setattr(capture_cameras, "board_ahead", world.ahead)
        frames, found = capture_cameras.capture_closeup(world, tmp_path, cams={})
        assert found == turn and len(frames) == 3
        assert world.closest > capture_cameras.CLOSEUP_STANDOFF_M - 0.02  # never nearer than the stand-off
        x, y = world.pose[0, 3], world.pose[1, 3]
        heading = np.degrees(np.arctan2(world.pose[1, 0], world.pose[0, 0]))
        # back near the start: the closed-loop return_to_start() takes the rest
        assert np.hypot(x, y) < 0.03 and abs(heading) < 20, (x, y, heading)


def test_the_closeup_is_skipped_without_a_side_board(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_cameras, "SETTLE_S", 0.0)
    world = _SideWorld(0.53)
    world.board = None
    monkeypatch.setattr(capture_cameras, "board_ahead", world.ahead)
    assert capture_cameras.capture_closeup(world, tmp_path, cams={}) == ([], None)


def test_only_offsets_larger_than_their_spread_are_applied():
    spread = {"arm_shoulder_lift": 1.1, "arm_elbow_flex": 2.7, "arm_wrist_flex": 2.5}  # Oct 29
    by_eye = {"arm_shoulder_lift": 0.9, "arm_elbow_flex": -1.0, "arm_wrist_flex": -1.1}
    assert sc.offsets_to_apply(by_eye, spread) == {}
    sloppy = {"arm_shoulder_lift": 0.9, "arm_elbow_flex": -9.0, "arm_wrist_flex": 6.0}
    assert sc.offsets_to_apply(sloppy, spread) == {"arm_elbow_flex": -9.0, "arm_wrist_flex": 6.0}
