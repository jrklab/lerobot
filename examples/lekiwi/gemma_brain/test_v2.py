
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

"""Offline checks for the v2 skills (v2/): geometry, and move/land/grip run by the Pi's
AutoController in the web_server simulation (as test_arm.py)."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web_server"))

from arm import Arm, Calibration  # noqa: E402
from test_arm import SimRobot as _SimRobot  # noqa: E402
from test_auto_control import Sim  # noqa: E402
from v2 import actions, params as prm  # noqa: E402
from v2.geometry import Geometry, fit_floor  # noqa: E402
from v2.skills import Skills2  # noqa: E402

CAL = Calibration()


class SimRobot(_SimRobot):
    def trajectory(self, waypoints, durations, z_min=None, contact_stop=False):
        cmd = {"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min}
        if contact_stop:
            cmd["contact_stop"] = True
        return self.run(cmd)

    def gripper(self, pos, speed=None):
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed})


def geom(floor_z):
    return Geometry(floor_coef=np.array([floor_z, 0.0, 0.0]), floor_points=3, jac=np.array([[4.2, 0.5], [0.7, -5.3]]),
                    jac_height=0.094, cam_above_tips=0.1075, descent_p0=np.array([0.5, 0.15]),
                    fixed_jaw_inner_u=0.564, fixed_jaw_tip_v=0.754, sources={})


def skills_over(real_floor_tcp_z, predicted):
    # The sim's floor stops the TCP; ours is TCP z at contact, so the same number.
    sim = Sim(floor_z=real_floor_tcp_z)
    sk = Skills2(SimRobot(sim), Arm(SimRobot(sim), CAL, body_check=False), geom(predicted))
    assert sk.move(xy=(0.22, 0.0), height=0.05).ok
    return sim, sk


def test_floor_fit_is_a_plane_and_falls_back_flat():
    pts = [{"x": x, "y": y, "z": -0.11 + 0.01 * x - 0.02 * y} for x, y in ((0.18, 0), (0.3, 0), (0.24, 0.05), (0.2, -0.04))]
    assert np.allclose(fit_floor(pts, -0.2), [-0.11, 0.01, -0.02])
    assert np.allclose(fit_floor(pts[:2], -0.2), [np.mean([p["z"] for p in pts[:2]]), 0, 0])
    assert np.allclose(fit_floor([], -0.2), [-0.2, 0, 0])


def test_jacobian_grows_closer_to_the_floor_and_turns_with_roll():
    g = geom(-0.11)
    assert np.linalg.norm(g.jacobian(0.03)) > np.linalg.norm(g.jacobian(0.094))
    assert np.allclose(g.jacobian(0.094), g.jac)
    j90 = g.jacobian(0.094, 90.0) * [[640], [480]]
    assert np.allclose(np.linalg.svd(j90, compute_uv=False), np.linalg.svd(g.jac * [[640], [480]], compute_uv=False))


def test_every_param_names_its_source():
    assert prm.ALL and all(p.source in ("CALIB", "DEMO", "SPEC", "RULE") and p.why for p in prm.ALL.values())


def test_land_on_the_predicted_floor_does_not_press():
    sim, sk = skills_over(real_floor_tcp_z=-0.12, predicted=-0.12)
    r = actions.land(sk)
    assert r.ok and not r.data["contact"]
    assert abs(sk.height() - prm.GRASP_LIFT_M.value) < 0.002


def test_land_on_a_slightly_higher_floor_stops_on_it():
    # 5 mm higher than predicted: the goal is ~1 mm below the real floor, under the Pi's
    # contact threshold, so the arm just stops on it (pressing at most that 1 mm).
    sim, sk = skills_over(real_floor_tcp_z=-0.115, predicted=-0.12)
    assert actions.land(sk).ok
    assert sk.tcp()[2] >= -0.115 - 0.0005


def test_land_treats_a_contact_over_half_a_brick_up_as_not_the_floor():
    # 7.5 mm above the predicted floor: more than half the brick's height, so it might be the
    # brick's edge; land refuses rather than close there (the fit is good to a few mm).
    sim, sk = skills_over(real_floor_tcp_z=-0.1125, predicted=-0.12)
    r = actions.land(sk)
    assert not r.ok and "above the floor" in r.reason, r


def test_land_refuses_to_close_on_something_tall():
    sim, sk = skills_over(real_floor_tcp_z=-0.12 + 0.03, predicted=-0.12)  # 3 cm up: not the floor
    r = actions.land(sk)
    assert not r.ok and "above the floor" in r.reason, r


def test_move_out_of_vertical_reach_is_refused_without_moving():
    sim, sk = skills_over(-0.12, -0.12)
    before = dict(sim.pos)
    r = sk.move(xy=(0.6, 0.0), height=0.10)
    assert not r.ok and "can't move" in r.reason
    assert sim.pos == before


def test_base_moves_are_one_kind_at_a_time():
    sim, sk = skills_over(-0.12, -0.12)
    assert not sk.move("base", delta=(0.05, 0.0), roll=10.0).ok


def test_actions_and_chains_hold_no_numbers():
    # Layer 3 composes primitives; every value must come from params.py or geometry.py.
    import ast

    for name in ("actions.py", "chains.py"):
        path = HERE / "v2" / name
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                assert node.value in (0, 1, 2, 1000), f"{name}:{node.lineno}: number {node.value} (put it in params.py with its source)"
