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

"""What the skills do on a robot without a base or a camera, as its description says (brain/body.py):
no base command is sent, a look is the one view there is, a missing camera's frame is None."""

from brain import actions, body
from brain.goals import Detection
from brain.primitives import Skills
from gemma_bot import protocol

LEKIWI_BASE = protocol.Base(kind="omni", max_speed_mps=0.15, max_turn_dps=45.0, max_move_m=1.0,
                            max_turn_deg=180.0, pos_tol_m=0.01, turn_tol_deg=2.0)


def described(base=LEKIWI_BASE, cameras=("front", "wrist")) -> protocol.Description:
    return protocol.Description(
        protocol=1, embodiment="test", joint_units="deg", arm_model="so101_new_calib",
        joints=[protocol.Joint("arm_shoulder_pan", -110.0, 110.0)], max_joint_speed_dps=60.0,
        heartbeat_timeout_s=1.5, base=base,
        cameras=[protocol.Camera(c, "arm" if c == "wrist" else "body") for c in cameras],
    )


class Robot:
    """Records what reaches the robot."""

    def __init__(self, description=None):
        self.description, self.sent = description, []

    def base(self, **kw):
        self.sent.append(("base", kw))
        return {"result": {"moved": [kw.get("dx", 0.0), kw.get("dy", 0.0), kw.get("dtheta_deg", 0.0)]}}

    def frame(self, cam, after=0.0):
        self.sent.append(("frame", cam))
        return b"\xff\xd8jpeg", after


def _skills(robot):
    sk = Skills.__new__(Skills)
    sk.robot, sk.log = robot, None
    return sk


def test_without_a_description_the_robot_is_lekiwi():
    assert body.has_base(None) and body.has_camera(None, "front") and body.has_camera(None, "wrist")
    assert not body.has_camera(None, "scene")


def test_a_description_says_what_the_robot_has():
    desk = Robot(described(base=None, cameras=("scene", "wrist")))
    assert not body.has_base(desk) and body.has_camera(desk, "scene") and not body.has_camera(desk, "front")
    assert body.has_base(Robot(described())) and body.base_limits(Robot(described())) == LEKIWI_BASE


def test_no_base_command_reaches_a_robot_without_a_base():
    robot = Robot(described(base=None))
    sk = _skills(robot)
    r = sk.move("base", delta=(0.1, 0.0))
    assert not r.ok and "no base to drive" in r.reason
    r = sk.move("base", roll=30.0)
    assert not r.ok and "no base to turn" in r.reason
    assert robot.sent == []


def test_a_missing_cameras_frame_is_none_and_never_asked_for():
    robot = Robot(described(cameras=("wrist",)))
    sk = _skills(robot)
    sk._settle = lambda: None
    assert sk.frame("front") is None
    assert robot.sent == []


class LookSkills:
    """What look_around reads: the robot, a move, a look; no geometry (nothing turns)."""

    def __init__(self, robot, seen):
        self.robot, self.seen_now, self.moves = robot, seen, []

    def move(self, part, **kw):
        self.moves.append((part, kw))
        return actions.Result(True)

    def find(self, obj, cam, fast_ok=True):
        return Detection(box=[0.4, 0.4, 0.6, 0.6]) if self.seen_now else None


def test_without_a_base_a_look_around_is_the_one_view():
    sk = LookSkills(Robot(described(base=None)), seen=True)
    r = actions.look_around(sk, "tennis ball")
    assert r.ok and r.data["turned"] == 0.0 and sk.moves == []
    sk = LookSkills(Robot(described(base=None)), seen=False)
    r = actions.look_around(sk, "tennis ball")
    assert not r.ok and "no base to turn" in r.reason and sk.moves == []
