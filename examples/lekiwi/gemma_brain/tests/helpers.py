
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

"""Shared by the brain tests: the web_server arm simulation's robot (SimRobot), measured-like
geometry, a fake box model, synthetic frames, a servo world with a chosen true Jacobian, and
the robot's own calibration (fixtures/arm_calibration.json)."""

import json
from pathlib import Path

import cv2
import numpy as np
from brain import goals
from brain.cameras import Cameras
from brain.geometry import Geometry, near_distance
from brain.goals import Detection
from brain.primitives import Skills
from common.arm import Arm, Calibration
from test_arm import SimRobot as _SimRobot
from test_auto_control import Sim

CAL = Calibration()


class SimRobot(_SimRobot):
    def trajectory(self, waypoints, durations, z_min=None, contact_stop=False):
        cmd = {"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min}
        if contact_stop:
            cmd["contact_stop"] = True
        return self.run(cmd)

    def gripper(self, pos, speed=None):
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed})

    def frame(self, cam, after=0.0):
        return b"", after

# The measured cameras (calib.json "cameras", Oct 30), and how far below the TCP-at-contact floor
# the floor they see lies (-0.0966 vs -0.1071: the floor board's top less its thickness), and the front camera's box
# bottom of a brick at the reach point (calibrate_front.py: u_target, near_bottom).
CAMS = Cameras.from_calib(json.loads((Path(__file__).parents[1] / "data" / "calib.json").read_text())["cameras"]["value"])
SEEN_BELOW_CONTACT_M = 0.0105
NEAR_BOTTOM_UV = (0.451, 0.699)

def geom(floor_z):
    g = Geometry(floor_coef=np.array([floor_z, 0.0, 0.0]), floor_points=3, cams=CAMS,
                 floor_seen_z=floor_z - SEEN_BELOW_CONTACT_M, sources={}, gripper_curve=GRIPPER_CURVE)
    g.front_near_x = near_distance(CAMS, g.front_point(NEAR_BOTTOM_UV))
    return g


# calibrate_gripper.py on this robot, Oct 29: [deg open, mm] (nothing, 2x4 both ways, 2x6 long).
GRIPPER_CURVE = [[1.41, 0.0], [9.93, 15.8], [22.32, 31.8], [34.67, 47.8]]

def skills_over(real_floor_tcp_z, predicted):
    # The sim's floor stops the TCP; ours is TCP z at contact, so the same number.
    sim = Sim(floor_z=real_floor_tcp_z)
    sk = Skills(SimRobot(sim), Arm(SimRobot(sim), CAL, body_check=False), geom(predicted))
    assert sk.move(xy=(0.22, 0.0), height=0.05).ok
    return sim, sk

class FakeModel:
    def __init__(self, boxes):
        self.boxes, self.calls = list(boxes), 0

    def box(self, jpeg, what):
        from common.gemma import Answer

        self.calls += 1
        self.asked = what
        return Answer(text="", value=self.boxes.pop(0) if self.boxes else None)

def red_frame(box):
    img = np.full((480, 640, 3), 170, np.uint8)
    x0, y0, x1, y1 = (int(v * s) for v, s in zip(box, (640, 480, 640, 480), strict=True))
    cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 220), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()

class FrameRobot:
    def __init__(self, jpeg):
        self.jpeg = jpeg

    def frame(self, cam, after=0.0):
        return self.jpeg, after

def finder(jpeg, main, fast):
    sk = Skills.__new__(Skills)
    sk.robot, sk.log, sk.main, sk.fast, sk.seen, sk.looks = FrameRobot(jpeg), None, main, fast, {}, {}
    sk.heights, sk._height_fallbacks = {}, set()
    return sk

class SimServo:
    """A world for servo(): the object's image point responds to arm moves through a TRUE
    Jacobian, which may differ from the calibrated one the goal uses."""

    def __init__(self, geom, start_err, true_scale=1.0, height=0.03):
        self.geom, self.roll, self.h, self.true_scale = geom, 0.0, height, true_scale
        self.goal = goals.JawGoal()
        self.point = self.goal.target(self, height) - np.asarray(start_err, float)
        self.moves = 0

    def height(self):
        return self.h

    def tcp(self):  # servo() logs the TCP at each look
        return np.zeros(3)

    cmd_tcp = tcp

    def move_measured(self, delta):
        return self.move("arm", delta=delta)

    def find(self, obj, cam, angle=False):
        u, v = self.point
        return Detection(box=[u - 0.08, v - 0.1, u, v + 0.1])

    def move(self, part, delta=None, roll=None, **kw):
        j = self.goal._jac(self, self.h) * self.true_scale
        self.point = self.point + j @ np.asarray(delta[:2], float)  # the object, seen from the moving camera
        self.moves += 1
        from brain.primitives import Result

        return Result(True)

    def ev(self, *a, **k):
        pass

    servo = Skills.servo

def jaw_geom():
    g = geom(-0.11)
    g.grasp_target = np.array([0.55, 0.80])
    return g

def _chain_sk(calls, held=True):
    from brain.primitives import Result

    sk = Skills.__new__(Skills)
    sk.servo = lambda obj, goal, part=None: (calls.append(f"{type(goal).__name__}:{part or goal.part}"), Result(True))[1]
    sk.move = lambda part, **k: (calls.append("move"), Result(True))[1]
    sk.grip = lambda how, **k: (calls.append(how), Result(True))[1]
    sk.held = lambda: held
    sk.log, sk.find, sk.seen = None, lambda *a, **k: None, {}  # the place check sees nothing: passes, unchecked
    sk.heights, sk._height_fallbacks = {}, set()
    return sk

def _real_cal():
    c = json.loads((Path(__file__).parent / "fixtures" / "arm_calibration.json").read_text())  # robot.calibration(), Oct 28
    return c["joints"], c["motor_resolution"]
