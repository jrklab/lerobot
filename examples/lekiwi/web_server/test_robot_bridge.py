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

"""RobotBridge's motor-side pieces for the arm calibrating itself, on a fake Feetech bus: the
uncalibrated start, the calibration write between two control ticks, the probe's register
writes. Run: uv run pytest -q web_server/test_robot_bridge.py (from examples/lekiwi)."""

import time

import pytest
from robot_bridge import ARM_JOINTS, MockLeKiwi, RobotBridge

from lerobot.motors import MotorCalibration

RES = 4096
EEPROM = {"Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"}


class FakeBus:
    """Feetech's arithmetic: Present_Position = (Actual - Homing_Offset) mod 4096, goals in the
    same reported ticks, EEPROM registers writable with the torque off only."""

    def __init__(self, actual: dict[str, int], offsets: dict[str, int], limits=(0, RES - 1)):
        self.actual = dict(actual)
        self.regs = {
            m: {
                "Homing_Offset": offsets[m],
                "Min_Position_Limit": limits[0],
                "Max_Position_Limit": limits[1],
                "Max_Torque_Limit": 1000,
                "Torque_Limit": 1000,
                "Goal_Position": (actual[m] - offsets[m]) % RES,
            }
            for m in actual
        }
        self.torque = dict.fromkeys(actual, True)
        self.calibration: dict[str, MotorCalibration] = {}
        self.fail_writes = False

    def read(self, reg, motor, normalize=True):
        return self.regs[motor][reg]

    def write(self, reg, motor, value, normalize=True):
        if self.fail_writes:
            raise ConnectionError("no status packet")
        if reg in EEPROM and self.torque[motor]:
            raise RuntimeError(f"{motor}: {reg} written with the torque on")
        self.regs[motor][reg] = int(value)

    def disable_torque(self, motors):
        for m in motors:
            self.torque[m] = False

    def enable_torque(self, motors):
        for m in motors:
            self.torque[m] = True

    def read_calibration(self):
        return {
            m: MotorCalibration(
                id=i,
                drive_mode=0,
                homing_offset=r["Homing_Offset"],
                range_min=r["Min_Position_Limit"],
                range_max=r["Max_Position_Limit"],
            )
            for i, (m, r) in enumerate(self.regs.items())
        }

    def write_calibration(self, calibration, cache=True):
        for m, c in calibration.items():
            self.write("Homing_Offset", m, c.homing_offset)
            self.write("Min_Position_Limit", m, c.range_min)
            self.write("Max_Position_Limit", m, c.range_max)
        if cache:
            self.calibration = calibration

    @property
    def is_calibrated(self):
        return self.read_calibration() == {
            m: MotorCalibration(**{**c.__dict__, "drive_mode": 0}) for m, c in self.calibration.items()
        } and set(self.calibration) == set(self.regs)

    def present(self, m) -> int:
        return (self.actual[m] - self.regs[m]["Homing_Offset"]) % RES

    def _units(self, m, ticks):
        c = self.calibration[m]
        span = c.range_max - c.range_min
        if m == "arm_gripper":
            return (ticks - c.range_min) / span * 100.0
        return (ticks - c.range_min) / span * 200.0 - 100.0

    def _ticks(self, m, units):
        c = self.calibration[m]
        span = c.range_max - c.range_min
        if m == "arm_gripper":
            return round(c.range_min + units / 100.0 * span)
        return round(c.range_min + (units + 100.0) / 200.0 * span)

    def sync_read(self, reg, motors):
        if reg == "Present_Load":
            return dict.fromkeys(motors, 100.0)
        assert reg == "Present_Position"
        return {m: self._units(m, self.present(m)) for m in motors}

    def sync_write(self, reg, values):
        assert reg == "Goal_Position"
        for m, v in values.items():
            self.regs[m]["Goal_Position"] = self._ticks(m, v)

    def goal_actual(self, m) -> int:
        """The physical position a motor's goal register asks for."""
        return (self.regs[m]["Goal_Position"] + self.regs[m]["Homing_Offset"]) % RES


class FakeLeKiwi:
    def __init__(self, bus: FakeBus, calibration: dict[str, MotorCalibration] | None = None):
        self.bus = bus
        self.calibration = calibration or {}
        self.bus.calibration = self.calibration
        self.saved: dict | None = None
        self.action_features, self.observation_features = {}, {}

    def connect(self, calibrate=True):
        if calibrate and not self.is_calibrated:
            raise RuntimeError("LeKiwi would ask at the keyboard")

    @property
    def is_calibrated(self):
        return self.bus.is_calibrated

    def _save_calibration(self):
        self.saved = dict(self.calibration)

    def get_observation(self):
        return {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position", ARM_JOINTS).items()}

    def send_action(self, action):
        self.bus.sync_write(
            "Goal_Position", {m: action[f"{m}.pos"] for m in ARM_JOINTS if f"{m}.pos" in action}
        )
        return action

    def stop_base(self):
        pass

    def disconnect(self):
        pass


# A brand-new arm: offsets from the factory (0), the joints anywhere on the turn.
ACTUAL = {
    "arm_shoulder_pan": 1980,
    "arm_shoulder_lift": 900,
    "arm_elbow_flex": 3100,
    "arm_wrist_flex": 2600,
    "arm_wrist_roll": 40,
    "arm_gripper": 2010,
}


def _new_arm(**kw):
    return FakeLeKiwi(FakeBus(ACTUAL, dict.fromkeys(ACTUAL, 0), **kw))


@pytest.fixture
def started():
    bridges = []

    def start(robot, **kw):
        bridge = RobotBridge(robot, **kw)
        bridge.start()
        bridges.append(bridge)
        return bridge

    yield start
    for b in bridges:
        b.stop()


def test_an_uncalibrated_arm_starts_on_a_provisional_calibration_only_when_allowed(started):
    with pytest.raises(RuntimeError, match="keyboard"):
        RobotBridge(_new_arm()).start()
    robot = _new_arm(limits=(500, 3500))
    robot.bus.regs["arm_elbow_flex"]["Torque_Limit"] = 300  # left lowered by a probe cut short
    bridge = started(robot, allow_uncalibrated=True)
    assert bridge.arm_calibration_state() == "provisional"
    cal = bridge.get_arm_calibration()
    assert all(cal[j]["range_min"] == 0 and cal[j]["range_max"] == RES - 1 for j in ARM_JOINTS)
    assert robot.bus.regs["arm_elbow_flex"]["Min_Position_Limit"] == 500  # the motors aren't touched
    assert robot.bus.regs["arm_elbow_flex"]["Torque_Limit"] == 1000  # but a probe's limit is put back


def test_a_calibrated_arm_starts_as_before(started):
    bus = FakeBus(ACTUAL, dict.fromkeys(ACTUAL, 0))
    calibration = bus.read_calibration()
    bridge = started(FakeLeKiwi(bus, calibration), allow_uncalibrated=True)
    assert bridge.arm_calibration_state() == "calibrated"
    assert started(MockLeKiwi()).arm_calibration_state() is None


def test_a_calibration_write_moves_nothing_and_saves_only_when_asked(started):
    robot = _new_arm()
    bridge = started(robot, allow_uncalibrated=True)
    time.sleep(0.1)  # a few control ticks
    # Each joint's actual position to read the homing tick (2047), the range the whole turn.
    new = {j: {"homing_offset": ACTUAL[j] - 2047, "range_min": 0, "range_max": RES - 1} for j in ARM_JOINTS}
    result = bridge.write_arm_calibration(new, save=False)
    assert result["motors_match"] and not result["saved"] and result["state"] == "provisional"
    assert robot.saved is None
    assert all(robot.bus.present(j) == 2047 for j in ARM_JOINTS)
    assert all(robot.bus.torque.values())
    time.sleep(0.1)  # the loop's goals since the write, in the new calibration
    assert all(abs(robot.bus.goal_actual(j) - ACTUAL[j]) <= 1 for j in ARM_JOINTS)

    final = {
        "arm_elbow_flex": {
            "homing_offset": ACTUAL["arm_elbow_flex"] - 2047,
            "range_min": 1100,
            "range_max": 3000,
        }
    }
    result = bridge.write_arm_calibration(final, save=True)
    assert result["motors_match"] and result["state"] == "calibrated"
    assert robot.saved["arm_elbow_flex"].range_min == 1100
    time.sleep(0.1)
    assert abs(robot.bus.goal_actual("arm_elbow_flex") - ACTUAL["arm_elbow_flex"]) <= 1


def test_bad_calibration_writes_are_refused(started):
    bridge = started(_new_arm(), allow_uncalibrated=True)
    ok = {"homing_offset": 0, "range_min": 0, "range_max": RES - 1}
    for bad in (
        {"arm_nose": ok},
        {"arm_elbow_flex": {**ok, "homing_offset": 2100}},
        {"arm_elbow_flex": {**ok, "range_min": 3000, "range_max": 2000}},
        {"arm_elbow_flex": {**ok, "range_max": RES}},
        {"arm_elbow_flex": {"homing_offset": 0}},
    ):
        with pytest.raises(ValueError):
            bridge.write_arm_calibration(bad, save=False)
    with pytest.raises(ValueError, match="mock"):
        started(MockLeKiwi()).write_arm_calibration({"arm_elbow_flex": ok}, save=False)


def test_a_probes_torque_limit_goes_in_and_back_and_a_failed_write_ends_it():
    robot = _new_arm()
    bridge = RobotBridge(robot, allow_uncalibrated=True)
    bridge._apply_bus_writes([("Torque_Limit", "arm_elbow_flex", 300)])
    assert robot.bus.regs["arm_elbow_flex"]["Torque_Limit"] == 300
    bridge._apply_bus_writes([("Torque_Limit", "arm_elbow_flex", None)])
    assert robot.bus.regs["arm_elbow_flex"]["Torque_Limit"] == 1000

    bridge.auto.heartbeat()
    units = dict.fromkeys(ARM_JOINTS, 0.0)
    bridge.auto.step(time.monotonic(), units, {j: {"pos": 0.0} for j in ARM_JOINTS}, {})  # a first tick
    bridge.auto.submit(
        {"cmd": "probe", "joint": "arm_elbow_flex", "direction": 1, "torque": 0.3, "units": "deg"}
    )
    robot.bus.fail_writes = True
    bridge._apply_bus_writes([("Torque_Limit", "arm_elbow_flex", 300)])
    st = bridge.auto.status({})["command"]
    assert st["state"] == "aborted" and "Torque_Limit" in st["reason"]


def test_a_camera_stall_does_not_freeze_the_joint_readings():
    robot = _new_arm()
    bridge = RobotBridge(robot, allow_uncalibrated=True)
    bridge.start()
    try:
        robot.get_observation = lambda: (_ for _ in ()).throw(TimeoutError("latest frame is too old"))
        robot.bus.actual["arm_elbow_flex"] += 100  # the arm moves while the camera is stalled
        time.sleep(0.2)
        want = robot.bus.sync_read("Present_Position", ["arm_elbow_flex"])["arm_elbow_flex"]
        assert bridge.get_joint_state()["arm_elbow_flex"]["pos"] == pytest.approx(want)
    finally:
        bridge.stop()


def test_a_relax_lets_the_arm_settle_before_the_write(started):
    robot = _new_arm()
    bridge = started(robot, allow_uncalibrated=True)
    time.sleep(0.1)
    real_sleep = time.sleep

    def settle(s):  # gravity lays the elbow 6 deg further down while its torque is off
        assert not robot.bus.torque["arm_elbow_flex"]
        robot.bus.actual["arm_elbow_flex"] += round(6 * RES / 360)
        real_sleep(s)

    import robot_bridge

    robot_bridge.time.sleep, orig = settle, robot_bridge.time.sleep
    try:
        bridge.write_arm_calibration({}, save=False, relax_s=0.2)
    finally:
        robot_bridge.time.sleep = orig
    time.sleep(0.1)
    assert all(robot.bus.torque.values())
    assert (
        abs(robot.bus.goal_actual("arm_elbow_flex") - robot.bus.actual["arm_elbow_flex"]) <= 1
    )  # held where it lies
    with pytest.raises(ValueError, match="relax_s"):
        bridge.write_arm_calibration({}, save=False, relax_s=10.0)
