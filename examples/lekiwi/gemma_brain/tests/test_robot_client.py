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

"""common/robot.py's enable(): no task on a provisional calibration, the calibration itself excepted;
none on a robot that doesn't serve the protocol's /v1."""

import pytest
from common.robot import Robot, RobotError
from gemma_bot import protocol


def _description(state):
    return protocol.Description(
        protocol=1, embodiment="test", joint_units="deg", arm_model="so101_new_calib",
        joints=[protocol.Joint("arm_shoulder_pan", -110.0, 110.0)], max_joint_speed_dps=60.0,
        heartbeat_timeout_s=1.5, calibration=state,
    ).to_dict()


def _robot(state, describe=True, **kw):
    r = Robot(url="http://robot.invalid", **kw)
    calls = []

    def fake_json(method, path, body=None, timeout=5.0):
        calls.append((method, path))
        if path == protocol.DESCRIBE:
            if not describe:  # a web server from before /v1
                raise RobotError(f"GET {path}: HTTP 404: Not Found")
            return _description(state)
        if path == protocol.STATUS:
            return {"joint_units": "deg", "arm_calibration": state, "targets": {"arm_gripper": 0.0}}
        return {"ok": True}

    r._json = fake_json
    r._heartbeat_loop = lambda: None
    return r, calls


def test_a_provisional_arm_is_refused_before_anything_changes():
    r, calls = _robot("provisional")
    with pytest.raises(RobotError, match="calibrate_all"):
        r.enable()
    assert ("POST", protocol.ENABLE) not in calls


@pytest.mark.parametrize("state", ["calibrated", None])  # None: a mock robot, no motor bus
def test_a_calibrated_or_mock_arm_enables(state):
    r, calls = _robot(state)
    r.enable()
    assert ("POST", protocol.ENABLE) in calls


def test_the_calibration_may_enable_a_provisional_arm():
    r, calls = _robot("provisional", require_calibrated=False)
    r.enable()
    assert ("POST", protocol.ENABLE) in calls


def test_a_robot_without_v1_is_refused_before_anything_changes():
    r, calls = _robot("calibrated", describe=False)
    with pytest.raises(RobotError, match="deploy web_server"):
        r.enable()
    assert ("POST", protocol.ENABLE) not in calls


def test_enable_keeps_the_description():
    r, _ = _robot("calibrated")
    r.enable()
    assert r.description.joints[0].name == "arm_shoulder_pan" and r.description.base is None
