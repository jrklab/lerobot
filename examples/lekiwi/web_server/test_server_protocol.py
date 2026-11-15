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

"""The web server speaks the executor protocol (gemma_bot.protocol): every /v1 path is served, the
paths before /v1 run the same handlers, and /v1/describe is a Description the brain can read."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

import server  # noqa: E402
from auto_control import IK_ARM_JOINTS  # noqa: E402
from gemma_bot import protocol  # noqa: E402
from robot_bridge import MockLeKiwi, RobotBridge  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    bridge = RobotBridge(MockLeKiwi(), allow_uncalibrated=True)
    bridge.start()
    monkeypatch.setattr(server, "bridge", bridge)
    yield TestClient(server.app)
    bridge.stop()


def _routes():
    return {(m, r.path): r.endpoint for r in server.app.routes for m in getattr(r, "methods", ())}


def test_every_protocol_path_is_served():
    served = {path for _, path in _routes()}
    paths = [protocol.DESCRIBE, protocol.ENABLE, protocol.DISABLE, protocol.HEARTBEAT, protocol.CMD]
    paths += [protocol.STOP, protocol.STATUS, protocol.CALIBRATION, protocol.frame("{cam_name}")]
    paths += [protocol.camera_controls("{cam_name}"), protocol.camera_controls("{cam_name}") + "/reset"]
    assert set(paths) <= served


def test_the_paths_before_v1_run_the_same_handlers():
    routes = _routes()
    for v1, legacy in protocol.LEGACY.items():
        methods = {m for m, p in routes if p == v1}
        assert methods, v1
        for m in methods:
            assert routes[(m, legacy)] is routes[(m, v1)], (m, legacy)


def test_describe_is_a_description(client):
    d = protocol.Description.from_dict(client.get(protocol.DESCRIBE).json())
    assert d.protocol == protocol.VERSION and d.joint_units == "deg"
    assert [j.name for j in d.joints] == IK_ARM_JOINTS
    assert all(j.min_deg < j.max_deg for j in d.joints)
    assert d.gripper.name == "arm_gripper" and d.gripper.min_deg == 0.0 < d.gripper.max_deg
    assert d.base.kind == "omni" and d.base.max_move_m == 1.0 and d.base.max_turn_deg == 180.0
    assert d.camera("front").mount == "body" and d.camera("wrist").mount == "arm"
    assert d.calibration is None  # the mock has no motor bus


def test_a_session_over_v1(client):
    assert client.post(protocol.ENABLE).json()["mode"] == "auto"
    assert client.post(protocol.HEARTBEAT).json()["ok"]
    assert client.get(protocol.STATUS).json()["mode"] == "auto"
    assert client.post(protocol.STOP).json()["ok"]
    assert client.post(protocol.DISABLE).json()["mode"] == "gamepad"
    # a command outside auto mode is refused, with its reason
    r = client.post(protocol.CMD, json={"cmd": "gripper", "pos": 10.0})
    assert r.status_code == 409 and "not in auto mode" in r.json()["reason"]
