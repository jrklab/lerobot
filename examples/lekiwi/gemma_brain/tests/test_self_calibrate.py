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

"""calibration/self_calibrate.py end to end on a simulated brand-new arm: the motors' factory
offsets unknown, the joints' ends past the URDF's limits and off-centre (as found on this arm,
Oct 29), the rest pose placed a few degrees off by eye. The Pi side is web_server's auto-mode
simulation (test_auto_control.Sim), its units following whatever calibration was last written."""

import pytest
from calibration import self_calibrate as sc
from common.arm import IK_ARM_JOINTS, NAMED_POSES
from common.robot import RobotError
from joint_angles import GRIPPER, URDF_LIMITS_DEG, JointAngles
from test_auto_control import Sim

RES = 4096
TPD = RES / 360.0
# True angle (the URDF's) of each joint's two ends; the gripper's in degrees open.
ENDS = {
    "arm_shoulder_pan": (-113.0, 115.6),
    "arm_shoulder_lift": (-106.0, 102.2),
    "arm_elbow_flex": (-101.0, 96.4),
    "arm_wrist_flex": (-100.0, 94.4),
    "arm_wrist_roll": (-150.0, 172.6),
    GRIPPER: (0.0, 100.0),
}
# Where the motors' ticks sit at each joint's true zero (a new arm: anywhere on the turn).
ZERO_TICK = {
    "arm_shoulder_pan": 2900,
    "arm_shoulder_lift": 1500,
    "arm_elbow_flex": 2200,
    "arm_wrist_flex": 1800,
    "arm_wrist_roll": 2100,
    GRIPPER: 1300,
}
REST_ERROR = {
    "arm_shoulder_pan": -3.0,
    "arm_shoulder_lift": 2.0,
    "arm_elbow_flex": -4.0,
    "arm_wrist_flex": 2.5,
    "arm_wrist_roll": 4.0,
    GRIPPER: 0.0,
}


class NewArm:
    """robot.Robot's calls used by self_calibrate, on a Sim whose units follow the motors'
    calibration: Feetech's Present = (Actual - Homing_Offset) mod 4096."""

    def __init__(self):
        rest = {**NAMED_POSES["rest"], GRIPPER: 0.0}
        self.true = {j: rest[j] + REST_ERROR[j] for j in rest}  # where the arm is placed
        self.calib = {
            j: {"homing_offset": 0, "range_min": 0, "range_max": RES - 1, "drive_mode": 0} for j in self.true
        }
        self.state = "provisional"  # the server's uncalibrated start
        self.saved = None
        self.sim = None
        self._resync()

    def actual(self, joint, true_deg):
        return ZERO_TICK[joint] + true_deg * TPD

    def _units(self, joint, actual):
        c = self.calib[joint]
        present = (actual - c["homing_offset"]) % RES
        span = c["range_max"] - c["range_min"]
        if joint == GRIPPER:
            return (present - c["range_min"]) / span * 100.0
        return (present - c["range_min"]) / span * 200.0 - 100.0

    def _true(self, joint, units):
        """Inverse of _units: the true angle of a reading (the arm's travel never wraps here)."""
        c = self.calib[joint]
        span = c["range_max"] - c["range_min"]
        frac = units / 100.0 if joint == GRIPPER else (units + 100.0) / 200.0
        present = c["range_min"] + frac * span
        actual = (present + c["homing_offset"]) % RES
        return ((actual - ZERO_TICK[joint]) / TPD + 180.0) % 360.0 - 180.0

    def _resync(self):
        if self.sim is not None:
            self.true = {j: self._true(j, u) for j, u in self.sim.pos.items()}
        units = {j: self._units(j, self.actual(j, t)) for j, t in self.true.items()}
        angles = JointAngles(self.calib, RES)
        self.sim = Sim(start=units, angles=angles)
        for j, (lo, hi) in ENDS.items():
            a, b = self._units(j, self.actual(j, lo)), self._units(j, self.actual(j, hi))
            self.sim.walls[j] = (min(a, b), max(a, b))

    def status(self):
        return self.sim.auto.status(dict(self.sim.pos))

    def calibration(self):
        return {
            "motor_resolution": RES,
            "joints": {j: dict(c) for j, c in self.calib.items()},
            "state": self.state,
        }

    def write_calibration(self, joints, save):
        self.true = {j: self._true(j, u) for j, u in self.sim.pos.items()}
        for j, c in joints.items():
            assert abs(c["homing_offset"]) <= 2047 and 0 <= c["range_min"] < c["range_max"] <= RES - 1
            self.calib[j].update(c)
        self.sim = None
        self._resync()
        self.state = "calibrated" if save else "provisional"
        self.saved = self.calibration()["joints"] if save else self.saved
        return {"ok": True}

    def run(self, cmd, timeout=60.0):
        self.sim.submit(cmd)
        st = self.sim.run_until_done()
        if st["state"] != "done":
            raise RobotError(f"{cmd['cmd']} {st['state']}: {st['reason']}", st)
        return st

    def gripper(self, pos, speed=None):
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed})

    def joints(self, targets, speed=None, z_min=None):
        return self.run({"cmd": "joints", "targets": targets, "speed": speed, "z_min": z_min})


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "STATE", tmp_path / "self_calibration.json")


def test_every_end_has_a_clear_pose_to_probe_from():
    for j in sc.ORDER:
        for d in (-1, 1):
            pose = sc.probe_pose(j, d)
            assert all(URDF_LIMITS_DEG[k][0] <= pose[k] <= URDF_LIMITS_DEG[k][1] for k in IK_ARM_JOINTS)


def test_the_provisional_offsets_read_rest_and_wrap_into_the_register():
    calib = {"arm_elbow_flex": {"homing_offset": 2000, "range_min": 0, "range_max": RES - 1, "drive_mode": 0}}
    # Reads 150 deg: actual = 2047 + 1707 + 2000 = 5754, i.e. 1658 on the turn.
    new = sc.provisional_calibration(calib, {"arm_elbow_flex": 150.0}, {"arm_elbow_flex": 81.29})
    off = new["arm_elbow_flex"]["homing_offset"]
    assert abs(off) <= 2047
    assert (5754 - off) % RES == round(2047 + 81.29 * TPD)


def test_a_new_arm_calibrates_itself(state):
    robot = NewArm()
    sc.bootstrap(robot)
    rest_read = robot.status()["joints"]
    assert all(abs(rest_read[j] - sc.REST[j]) < 0.2 for j in IK_ARM_JOINTS)
    for j in sc.ORDER:
        ends = sc.probe_joint(robot, j)
        assert ends["-"] < ends["+"]
    sc.write(robot, dry_run=False)
    assert robot.state == "calibrated" and robot.saved
    read = robot.status()["joints"]
    angles = JointAngles(robot.calib, RES)
    for j in IK_ARM_JOINTS:
        lo, hi = ENDS[j]
        # The zero from the ends' midpoint is off by how far the ends aren't centred on the
        # URDF's; pan and roll keep the rest placement's error.
        expect = -(lo + hi) / 2 if j in sc.ZERO_FROM_ENDS else -REST_ERROR[j]
        err = read[j] - robot.true[j]
        assert abs(err - expect) < 0.5, (j, err, expect)
        # The range: the ends (as now read) less the back-off, clipped to the URDF's.
        ulo, uhi = URDF_LIMITS_DEG[j]
        want = (max(ulo, lo + err + sc.PROBE_BACKOFF_DEG), min(uhi, hi + err - sc.PROBE_BACKOFF_DEG))
        assert angles.limits(j) == pytest.approx(want, abs=1.0), j
    assert abs(read[GRIPPER] - robot.true[GRIPPER]) < 0.5  # closed reads 0
    assert angles.limits(GRIPPER)[1] == pytest.approx(100.0 - sc.PROBE_BACKOFF_DEG, abs=0.5)


def test_a_joint_short_of_the_urdf_is_probed_again_then_kept(state, monkeypatch, capsys):
    monkeypatch.setitem(ENDS, "arm_wrist_roll", (-150.0, 120.0))  # a cable stops it early
    robot = NewArm()
    sc.bootstrap(robot)
    sc.probe_joint(robot, GRIPPER)
    capsys.readouterr()
    ends = sc.probe_joint(robot, "arm_wrist_roll")
    assert "again at torque 0.30" in capsys.readouterr().out
    assert abs(ends["+"] - (120.0 - REST_ERROR["arm_wrist_roll"])) < 1.0  # read from the rest placement
