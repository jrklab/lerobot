
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

"""Geometry (brain/geometry.py) and the parameter rules: every number in params.py names its
source, and the layers above the primitives hold none."""

from pathlib import Path

import numpy as np
from brain import params as prm
from brain.geometry import fit_floor
from helpers import (
    geom,
    jaw_geom,
)

HERE = Path(__file__).resolve().parent


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

def _stray_numbers(path):
    """Numeric literals in a v2 module that aren't plain 0/1/2, an index, a rounding digit
    count, or on a line marked "# convention: <why>" (pitch 90 = straight down, the +/-100
    joint-unit ends, the brick's 180 deg symmetry, the image size, ...)."""
    import ast

    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):  # indices and slices
            allowed.update(id(n) for n in ast.walk(node.slice))
        if isinstance(node, ast.Call):  # round(x, 4), arr.round(4)
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn == "round":
                allowed.update(id(a) for a in node.args[1 if isinstance(node.func, ast.Name) else 0:])
    stray = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if node.value in (0, 1, 2) or id(node) in allowed or "# convention" in lines[node.lineno - 1]:
                continue
            stray.append(f"{path.name}:{node.lineno}: {node.value!r}")
    return stray

def test_brain_code_holds_no_stray_numbers():
    # Layers 1-3: every value comes from params.py (with its source) or geometry.py (CALIB);
    # actions and chains may not even mark conventions.
    for name in ("primitives.py", "goals.py", "geometry.py", "actions.py", "chains.py"):
        path = HERE.parent / "brain" / name
        stray = _stray_numbers(path)
        assert not stray, f"numbers outside params.py: {stray} (move them there with their source, or mark a convention)"
    for name in ("actions.py", "chains.py"):
        assert "# convention" not in (HERE.parent / "brain" / name).read_text()

def test_the_stray_number_check_catches_one(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("A = 0.002\nB = round(0.5, 4)\nC = s[9]\nD = 90.0  # convention: straight down\n")
    assert _stray_numbers(p) == ["x.py:1: 0.002", "x.py:2: 0.5"]

def test_locate_from_one_view_recovers_a_cup_placed_in_a_simulated_view():
    # A camera model consistent with geometry.py: a point at (dx, dy) from the TCP's vertical
    # and height r is seen at under_tcp(z) - J(h) (d + h) / z @ (dx, dy), z = d + h - r.
    from brain.geometry import locate_from_one_view

    g = jaw_geom()
    g.grasp_box, g.grasp_target_height = [0.411, 0.613, 0.545, 0.927], 0.0015
    d, rim, true, h = g.cam_above_tips, 0.044, np.array([0.012, -0.020]), 0.106
    taught = np.array([(0.411 + 0.545) / 2, (0.613 + 0.927) / 2])
    z_t, z = d + 0.0015 - prm.BRICK_HEIGHT_M.value, d + h - rim
    c = g.descent_p0 + (taught - g.descent_p0) * z_t / z - (g.jacobian(h) * (d + h) / z) @ true
    assert np.allclose(locate_from_one_view(g, c, h, rim), true, atol=1e-9)

def test_front_object_height_of_the_cup_in_two_logged_views():
    from brain.geometry import front_object_height

    g = geom(-0.11)
    g.front = {"u_per_deg": 0.0184, "range_k": 0.0583, "range_horizon_v": 0.335}
    near, far = front_object_height(g, [0.25, 0.321, 0.7, 0.803]), front_object_height(g, [0.5, 0.333, 0.88, 0.647])
    assert abs(near - far) < 0.003 and 0.04 < near < 0.05  # Oct 28: 4.3 and 4.2 cm
