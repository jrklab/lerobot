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

"""calibration/calibrate_all.py's decisions, the robot and the solve stood in for: when to
correct, when the verdict is in (two scenes under one calibration), where the board is touched."""

import json
from pathlib import Path

import numpy as np
import pytest
from calibration import calibrate_all as ca, capture_cameras


class Fake:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)  # to_apply per solve with fit_offsets
        self.captures, self.solves, self.corrections, self.written = [], [], [], None
        self.swept, self.n_poses, self.extras = [], [], []

    def run_scene(self, robot, poses, moves, closeup=True):
        p = Path(f"scene{len(self.captures)}")
        self.captures.append((p, poses[0]["reverse"]))
        self.swept.append(bool(moves) and closeup)
        self.n_poses.append(len(poses))
        return p

    def solve(self, scenes, fit_offsets=False, floor_board_thickness=0.0, write=False, front_extra=()):
        self.solves.append(([s.name for s in scenes], fit_offsets, write))
        self.extras.append([s.name for s in front_extra])
        apply = self.verdicts.pop(0) if fit_offsets else {}
        return {
            "joint_offsets_deg": {},
            "spread_deg": {},
            "to_apply_deg": apply,
            "rms_px": {},
            "board_top_z": -0.104,
            "floor_z": -0.105,
            "floor_board_pose": [np.eye(4).tolist()],
        }


@pytest.fixture
def fake(monkeypatch):
    def make(verdicts):
        f = Fake(verdicts)
        monkeypatch.setattr(
            ca.capture_cameras, "plan", lambda calib, x, reverse: [{"joints": {}, "reverse": reverse}] * 38
        )
        monkeypatch.setattr(ca.capture_cameras, "run_scene", f.run_scene)
        monkeypatch.setattr(ca.capture_cameras, "sweep_plan", lambda: [{"turn_deg": 10}])
        monkeypatch.setattr(ca.capture_cameras, "position_for_floor_board", lambda robot: {"ok": True})
        monkeypatch.setattr(ca.solve_cameras, "solve", f.solve)
        monkeypatch.setattr(ca.sc, "correct", lambda robot, offsets: f.corrections.append(offsets))
        monkeypatch.setattr(ca.Calibration, "from_robot", classmethod(lambda cls, robot: None))
        return f

    return make


def _log():
    return {"solves": [], "corrections": []}


def test_a_correction_restarts_the_round_and_the_verdict_takes_two_scenes(fake):
    f = fake([{"arm_elbow_flex": -4.1}, {}, {}])
    out = ca.fine(None, 0.00125, _log())
    assert f.corrections == [{"arm_elbow_flex": -4.1}]
    # scene0 (forward) -> correct; scene1 (reverse) alone; scene1 + scene2 (forward) together
    assert [s for s, fit, _ in f.solves if fit] == [["scene0"], ["scene1"], ["scene1", "scene2"]]
    assert [r for _, r in f.captures] == [False, True, False]
    assert f.solves[-1] == (["scene1", "scene2"], False, True)  # the cameras written from both
    assert [s.name for s in out["scenes"]] == ["scene1", "scene2"]
    # Only the first scene sweeps the base, with every other pose; the later ones reuse its sweep.
    assert f.swept == [True, False, False] and f.n_poses == [19, 38, 38]
    assert f.extras[-1] == ["scene0"]


def test_no_correction_needed_still_takes_both_orders(fake):
    f = fake([{}, {}])
    ca.fine(None, 0.0, _log())
    assert f.corrections == [] and [r for _, r in f.captures] == [False, True]


def test_corrections_stop_at_the_limit(fake):
    f = fake([{"arm_wrist_flex": 2.0}] * (ca.MAX_CORRECTIONS + 2))
    log = _log()
    ca.fine(None, 0.0, log)
    assert len(f.corrections) == ca.MAX_CORRECTIONS == len(log["corrections"])


def _board_at(x_corner, y_corner=0.127):
    # As the cameras placed it Oct 30: the board's x axis along the arm's -y, its y along -x; its
    # inner corners from 0.025 to 0.15 m along its y, so the printed area starts at x_corner - 0.15.
    return np.array(
        [[0.0, -1.0, 0.0, x_corner], [-1.0, 0.0, 0.0, y_corner], [0.0, 0.0, -1.0, -0.104], [0, 0, 0, 1]]
    )


def test_touches_spread_over_the_reach_off_the_boards_edge():
    # The first placement: printed area from 0.292, edge 0.267: two on the floor, one on the board.
    assert ca.touch_points(_board_at(0.442)) == [(0.18, "floor"), (0.237, "floor"), (0.3, "board")]
    # The second, nearer: printed area from 0.255, edge 0.23: one on the floor, two on the board.
    assert ca.touch_points(_board_at(0.405)) == [(0.18, "floor"), (0.255, "board"), (0.3, "board")]
    # Nearer still (Oct 30: printed from 0.22, the edge 0.195): no bare floor in reach, all on the board.
    assert ca.touch_points(_board_at(0.37)) == [(0.22, "board"), (0.24, "board"), (0.3, "board")]
    # Out of reach altogether: all on the floor.
    assert [on for _, on in ca.touch_points(_board_at(0.60))] == ["floor"] * 3
    with pytest.raises(RuntimeError, match="centre line"):
        ca.touch_points(_board_at(0.442, y_corner=-0.15))


def test_the_floor_stage_writes_the_plane_and_the_check(tmp_path, monkeypatch):
    """Oct 30: a name clash left the touches unwritten after all three had run."""
    (tmp_path / "calib.json").write_text("{}")
    monkeypatch.setattr(ca, "CALIB", tmp_path / "calib.json")
    monkeypatch.setattr(ca, "TOUCHES", tmp_path / "floor_touches.json")

    class Arm:
        def __init__(self, robot):
            pass

        def pose(self, name, speed=None):
            pass

        def reset(self):
            pass

    monkeypatch.setattr(ca, "Arm", Arm)
    monkeypatch.setattr(ca, "lay_down", lambda robot: None)
    tcp = {0.18: -0.1036, 0.237: -0.1034, 0.3: -0.1039}
    monkeypatch.setattr(ca.calibrate_floor, "probe", lambda arm, x: {"x": x, "floor_tcp_z": tcp[round(x, 3)]})
    cams = {"floor_board_pose": [_board_at(0.442).tolist()], "board_top_z": -0.1077}
    log = {}
    out = ca.floor(None, cams, 0.00125, log)
    touches = json.loads((tmp_path / "floor_touches.json").read_text())["points"]
    assert [p["how"] for p in touches] == ["probe on the floor", "probe on the floor", "probe on the board"]
    assert touches[-1]["z"] == pytest.approx(
        -0.1039 - 0.00125, abs=1e-4
    )  # the board's thickness off (0.1 mm rounding)
    ck = out["check"]
    want = 1000 * (-0.1039 - ca.JAW_TIP_OFFSET_M) - 1000 * ca.board_top_at(cams["floor_board_pose"], 0.3)
    assert ck["made"] and [q["x"] for q in ck["points"]] == [0.3]
    assert ck["points"][0]["diff_mm"] == pytest.approx(want, abs=0.1)
    assert ck["pass"] == (abs(want) <= ca.CHECK_BAND_MM)
    assert json.loads((tmp_path / "calib.json").read_text())["floor_tcp_z_m"]["value"] == out["floor_tcp_z"]


def test_the_board_top_is_read_where_it_was_touched():
    pose = _board_at(0.442)
    assert ca.board_top_at([pose], 0.30) == pytest.approx(-0.104)  # level: the same everywhere
    tilted = pose.copy()
    a = np.radians(1.0)  # a board tilted 1 deg about the arm's y axis
    tilted[:3, :3] = (
        np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ pose[:3, :3]
    )
    near, far = ca.board_top_at([tilted], 0.25), ca.board_top_at([tilted], 0.35)
    assert abs(abs(far - near) - 0.10 * np.tan(a)) < 1e-4  # 1.7 mm over 10 cm


class Base:
    """The base's position, and the front camera's view of the floor board from it."""

    def __init__(self, view):
        self.fwd = self.left = 0.0
        self.view = view
        self.moves = []

    def base(self, dx=0.0, dy=0.0):
        self.fwd += dx
        self.left += dy
        self.moves.append((round(dx, 3), round(dy, 3)))

    def look(self):
        return self.view(self.fwd, self.left)


def _far_board(fwd, left, need=0.06, u0=0.5):
    """A board read well from `need` forward on; nearer, its lowest corners drop down the image."""
    v = 0.55 + 2.0 * fwd
    if fwd < need - 0.035:
        return 0, None, None
    return (18 if fwd >= need - 1e-9 else 9), u0 + 2.5 * left, v  # moving right shifts it left in the image


def test_a_far_board_is_driven_up_to():
    b = Base(_far_board)
    out = capture_cameras.position_for_floor_board(b, look=b.look)
    assert out["ok"] and out["forward_m"] == pytest.approx(0.06) and b.moves == [(0.03, 0.0), (0.03, 0.0)]


def test_a_well_read_board_moves_nothing():
    b = Base(lambda f, left: (20, 0.5, 0.7))
    assert capture_cameras.position_for_floor_board(b, look=b.look)["ok"] and b.moves == []


def test_a_board_too_near_is_backed_away_from():
    # Read poorly with its near edge low in the view: back up until it reads well.
    b = Base(lambda f, left: (18, 0.5, 0.8) if f <= -0.06 + 1e-9 else (8, 0.5, 0.92))
    out = capture_cameras.position_for_floor_board(b, look=b.look)
    assert out["ok"] and out["forward_m"] == pytest.approx(-0.06)


def test_never_nearer_than_the_near_row():
    # Read better the nearer it is, but well only past the near row: stop there, nothing better.
    b = Base(lambda f, left: (int(6 + 60 * f), 0.5, 0.80 + f))
    out = capture_cameras.position_for_floor_board(b, look=b.look)
    assert not out["ok"]
    stop = max(f for f in (0.0, 0.03, 0.06, 0.09) if 0.80 + f - 0.03 <= capture_cameras.NEAR_ROW)
    assert out["forward_m"] == pytest.approx(stop)  # one step past the row at most, then no further
    assert all(abs(dx) <= capture_cameras.MAX_REPOSITION_M for dx, _ in b.moves)


def test_a_board_off_to_one_side_is_centred():
    b = Base(lambda f, left: _far_board(f, left, need=0.0, u0=0.75))  # read well, right of centre
    out = capture_cameras.position_for_floor_board(b, look=b.look)
    assert out["left_m"] < 0  # moved right
    assert abs(0.75 + 2.5 * out["left_m"] - 0.5) <= capture_cameras.CENTRED_U


def _status(loads, speeds=None, kind=None, state="done"):
    j = list(ca.sc.IK_ARM_JOINTS)
    return {"loads": dict(zip(j, loads, strict=True)), "speeds": dict(zip(j, speeds or [0] * 5, strict=True)),
            "command": {"kind": kind, "state": state} if kind else None}


def test_the_load_log_keeps_still_joints_outside_probes_and_touches():
    log = ca.LoadLog()
    log(_status([100, 200, 150, 80, 50]))  # holding: kept
    log(_status([900] * 5, kind="probe", state="running"))  # a probe presses on purpose
    log(_status([700] * 5, speeds=[500] * 5))  # moving fast: the stall rule wouldn't judge it
    log.paused = True
    log(_status([800] * 5))  # the floor touches
    assert [log.samples[j] for j in ca.sc.IK_ARM_JOINTS] == [[100.0], [200.0], [150.0], [80.0], [50.0]]


def test_the_proposal_is_the_still_loads_percentile_with_the_margin(tmp_path, monkeypatch):
    log = ca.LoadLog()
    for k in range(ca.MIN_LOAD_SAMPLES):
        log(_status([100 + k % 5, 300, 200, 120, 60]))
    p = log.propose()
    assert p["arm_shoulder_lift"]["proposed"] == pytest.approx(ca.STALL_MARGIN * 0.3)
    assert p["arm_shoulder_pan"]["load_max"] == 104
    assert set(p) == set(ca.sc.IK_ARM_JOINTS)  # not the gripper
    (tmp_path / "calib.json").write_text("{}")
    monkeypatch.setattr(ca, "CALIB", tmp_path / "calib.json")
    ca.save_load_proposal(p)
    saved = json.loads((tmp_path / "calib.json").read_text())["joint_stall_thresholds"]
    assert saved["enabled"] is False and saved["value"]["arm_shoulder_lift"] == pytest.approx(0.45)


def test_too_few_samples_propose_nothing():
    log = ca.LoadLog()
    log(_status([100] * 5))
    assert log.propose() == {}
