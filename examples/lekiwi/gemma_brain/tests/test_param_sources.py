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

"""brain/params.py values read from data/: where each one comes from when its source is there,
and the fallback when it isn't; and this robot's values, read from its config (configs/)."""

import json
import tomllib

import pytest
from brain import params


def _calib(tmp_path, monkeypatch, check):
    f = tmp_path / "calib.json"
    f.write_text(json.dumps({"floor_tcp_z_m": {"value": -0.104, **({"check": check} if check else {})}}))
    monkeypatch.setattr(params, "CALIB_FILE", f)


def test_the_landing_search_is_the_verified_floor_band(tmp_path, monkeypatch):
    _calib(tmp_path, monkeypatch, {"made": True, "pass": True, "band_mm": 3.0, "worst_mm": 0.9})
    assert params._floor_accuracy(0.008) == 0.003


def test_without_a_passed_check_the_old_value_stands(tmp_path, monkeypatch):
    for check in (None, {"made": True, "pass": False, "band_mm": 3.0}, {"made": False, "why": "no touch on the board"}):
        _calib(tmp_path, monkeypatch, check)
        assert params._floor_accuracy(0.008) == 0.008
    monkeypatch.setattr(params, "CALIB_FILE", tmp_path / "missing.json")
    assert params._floor_accuracy(0.008) == 0.008


def test_every_robot_value_is_its_configs_and_the_config_has_no_unused_one():
    config = tomllib.loads(params.ROBOT_FILE.read_text())
    keys = {f"{section}.{name}" for section, values in config.items() for name in values}
    robot = {n: p for n, p in params.ALL.items() if p.source == "ROBOT"}
    assert robot and len(keys) == len(robot)
    used = {p.value for p in robot.values()}
    assert {float(config[k.split(".")[0]][k.split(".")[1]]) for k in keys} == used


def test_a_value_missing_from_the_config_stops_the_brain(monkeypatch):
    monkeypatch.setattr(params, "_ROBOT_CONFIG", {"arm": {}})
    with pytest.raises(KeyError, match="arm.reach_point_x_m"):
        params._robot("arm.reach_point_x_m")
