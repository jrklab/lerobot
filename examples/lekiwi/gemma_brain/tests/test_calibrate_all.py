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

from pathlib import Path

import numpy as np
import pytest
from calibration import calibrate_all as ca


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
    # Out of reach altogether: all on the floor.
    assert [on for _, on in ca.touch_points(_board_at(0.60))] == ["floor"] * 3
    with pytest.raises(RuntimeError, match="centre line"):
        ca.touch_points(_board_at(0.442, y_corner=-0.15))
