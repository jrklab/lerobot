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

"""common/robot.py's enable(): no task on a provisional calibration, the calibration itself excepted."""

import pytest
from common.robot import Robot, RobotError


def _robot(state, **kw):
    r = Robot(url="http://robot.invalid", **kw)
    calls = []

    def fake_json(method, path, body=None, timeout=5.0):
        calls.append((method, path))
        if path == "/auto/status":
            return {"joint_units": "deg", "arm_calibration": state, "targets": {"arm_gripper": 0.0}}
        return {"ok": True}

    r._json = fake_json
    r._heartbeat_loop = lambda: None
    return r, calls


def test_a_provisional_arm_is_refused_before_anything_changes():
    r, calls = _robot("provisional")
    with pytest.raises(RobotError, match="calibrate_all"):
        r.enable()
    assert ("POST", "/auto/enable") not in calls


@pytest.mark.parametrize("state", ["calibrated", None])  # None: a mock robot, no motor bus
def test_a_calibrated_or_mock_arm_enables(state):
    r, calls = _robot(state)
    r.enable()
    assert ("POST", "/auto/enable") in calls


def test_the_calibration_may_enable_a_provisional_arm():
    r, calls = _robot("provisional", require_calibrated=False)
    r.enable()
    assert ("POST", "/auto/enable") in calls
