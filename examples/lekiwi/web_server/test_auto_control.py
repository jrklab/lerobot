#!/usr/bin/env python

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

"""Offline checks for auto_control.AutoController against a simulated arm and base.

    python examples/lekiwi/web_server/test_auto_control.py     (or: pytest <this file>)

The simulation (Sim, also used by gemma_brain/tests/test_arm.py): each joint moves toward its
target at a capped speed, or is blocked at a "wall" (which then reports a stall); the base's
measured velocity is its command. It stands in for RobotBridge, so it works in normalized
units, under SIM_CALIBRATION: one unit is exactly one degree, so its positions and the
controller's degrees read the same here, while every value still goes through joint_angles.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import so101_kinematics as kin  # noqa: E402
from auto_control import (  # noqa: E402
    GRIPPER,
    HEARTBEAT_TIMEOUT_S,
    IK_ARM_JOINTS,
    AutoController,
    CommandRefused,
)
from joint_angles import JointAngles, shifted_calibration  # noqa: E402

PERIOD = 1 / 30
READY = dict.fromkeys(IK_ARM_JOINTS, 0.0)

# Arm joints: -100..100 units = -100..100 deg about the homing tick; gripper: 0..100 units =
# 0..100 deg open. Limits: that range clipped to the URDF's (the elbow to +/-96.8, the wrist
# flex to +/-95).
TICKS_PER_DEG = 4096 / 360
SIM_CALIBRATION = {
    j: {"range_min": 2047 - 100 * TICKS_PER_DEG, "range_max": 2047 + 100 * TICKS_PER_DEG, "drive_mode": 0}
    for j in IK_ARM_JOINTS
}
SIM_CALIBRATION[GRIPPER] = {"range_min": 0.0, "range_max": 100 * TICKS_PER_DEG, "drive_mode": 0}
SIM_ANGLES = JointAngles(SIM_CALIBRATION)


def norm_to_rad(joint, norm):
    return math.radians(SIM_ANGLES.to_deg(joint, norm))


class Sim:
    def __init__(self, start=None, floor_z=None, angles=None):
        """angles: the calibration the simulated bridge's units follow (SIM_ANGLES: 1 unit = 1 deg)."""
        self.floor_z = floor_z  # if set, the TCP can't go below it (the arm stops, like on a floor)
        self.pos = dict(start or {**READY, GRIPPER: 50.0})
        self.targets = dict(self.pos)
        self.walls = {}  # joint -> (lo, hi): the joint can't move outside this (contact)
        self.extra = {}  # joint -> extra joint_state readings, e.g. the gripper's temperature
        self.base_cmd = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self.t = 0.0
        self.angles = angles or SIM_ANGLES
        self.auto = AutoController(lambda: self.angles, clock=lambda: self.t)
        self.run(0.1)  # a few ticks so the controller has seen the arm's targets

    def joint_state(self):
        state = {}
        for j, p in self.pos.items():
            lo, hi = self.walls.get(j, (-1e9, 1e9))
            blocked = (self.targets[j] < lo - 1 and p <= lo + 0.5) or (
                self.targets[j] > hi + 1 and p >= hi - 0.5
            )
            state[j] = {
                "pos": p,
                "load": 900.0 if blocked else 50.0,
                "speed": 0.0,
                "severity": 0.8 if blocked else 0.0,
            }
            state[j].update(self.extra.get(j, {}))
        return state

    def run(self, seconds, heartbeat=True):
        for _ in range(max(1, round(seconds / PERIOD))):
            self.t += PERIOD
            if heartbeat:
                self.auto.heartbeat()
            self.base_cmd = self.auto.step(self.t, self.targets, self.joint_state(), self.base_cmd)
            new = dict(self.pos)
            for j, target in self.targets.items():
                step = max(-4.0, min(4.0, target - self.pos[j]))  # a 120 units/s servo
                lo, hi = self.walls.get(j, (-1e9, 1e9))
                new[j] = max(lo, min(hi, self.pos[j] + step))
            if self.floor_z is not None and self._tcp_z(new) < self.floor_z:
                new = {j: (new[j] if j == GRIPPER else self.pos[j]) for j in new}  # blocked by the floor
            self.pos = new

    def run_until_done(self, limit_s=60.0):
        for _ in range(round(limit_s / 0.1)):
            self.run(0.1)
            st = self.command()
            if st and st["state"] != "running":
                return st
        raise AssertionError("command never finished")

    def submit(self, cmd):
        """A planner on this version: joint positions in degrees."""
        self.auto.heartbeat()
        return self.auto.submit({"units": "deg", **cmd})

    def command(self):
        return self.auto.status(dict(self.pos))["command"]

    def _tcp_z(self, pos):
        return float(kin.forward_kinematics(np.array([norm_to_rad(j, pos[j]) for j in IK_ARM_JOINTS]))[2, 3])

    def tcp(self):
        return kin.forward_kinematics(np.array([norm_to_rad(j, self.pos[j]) for j in IK_ARM_JOINTS]))


def refused(sim, cmd, *words):
    try:
        sim.submit(cmd)
    except CommandRefused as e:
        assert all(w in str(e) for w in words), str(e)
        return str(e)
    raise AssertionError(f"expected a refusal for {cmd}")


def test_trajectory_plays_back():
    sim = Sim()
    wps = [{"arm_shoulder_pan": 10.0}, {"arm_shoulder_pan": 20.0, "arm_elbow_flex": 10.0}]
    sim.submit({"cmd": "trajectory", "waypoints": wps, "durations": [0.5, 0.5]})
    st = sim.run_until_done()
    assert st["state"] == "done", st
    assert abs(sim.pos["arm_shoulder_pan"] - 20.0) < 0.5 and abs(sim.pos["arm_elbow_flex"] - 10.0) < 0.5


def test_trajectory_too_fast_refused():
    sim = Sim()
    refused(
        sim, {"cmd": "trajectory", "waypoints": [{"arm_shoulder_pan": 50.0}], "durations": [0.2]}, "deg/s"
    )


def test_trajectory_out_of_range_refused():
    sim = Sim()
    refused(
        sim, {"cmd": "trajectory", "waypoints": [{"arm_elbow_flex": 140.0}], "durations": [5.0]}, "outside"
    )


def test_trajectory_below_floor_refused():
    sim = Sim()
    # Wrist flexed 90 deg points the TCP down from the ready pose, to z ~0.075; with z_min
    # above that the Pi's FK check must refuse it.
    cmd = {"cmd": "trajectory", "waypoints": [{"arm_wrist_flex": 90.0}], "durations": [2.0], "z_min": 0.1}
    refused(sim, cmd, "below z_min")


def test_floor_checked_between_waypoints():
    sim = Sim()
    # A floor that only the middle of the move would touch: both ends are clear.
    sim.auto._lowest_z = lambda j: -1.0 if 20.0 < j["arm_shoulder_pan"] < 30.0 else 1.0
    cmd = {"cmd": "joints", "targets": {"arm_shoulder_pan": 50.0}, "speed": 30, "z_min": 0.0}
    refused(sim, cmd, "% of the way")


def test_done_reports_settle_error():
    sim = Sim()
    sim.walls["arm_shoulder_pan"] = (-1e9, 6.0)  # sags 4 deg short of 10, no stall reported
    orig = sim.joint_state

    def no_stall():
        s = orig()
        for v in s.values():
            v["severity"] = 0.0
        return s

    sim.joint_state = no_stall
    sim.submit({"cmd": "joints", "targets": {"arm_shoulder_pan": 10.0}, "speed": 10})
    st = sim.run_until_done()
    assert st["state"] == "done" and not st["result"]["settled"] and st["result"]["max_error"] >= 3.9, st
    assert "settle timeout" in st["reason"], st


def test_joints_move_is_smooth_and_capped():
    sim = Sim()
    sim.submit({"cmd": "joints", "targets": {"arm_shoulder_pan": 40.0}, "speed": 200})  # capped
    prev, peak = sim.targets["arm_shoulder_pan"], 0.0
    while sim.command()["state"] == "running":
        sim.run(PERIOD)
        peak = max(peak, abs(sim.targets["arm_shoulder_pan"] - prev) / PERIOD)
        prev = sim.targets["arm_shoulder_pan"]
    assert sim.command()["state"] == "done"
    assert peak <= 60.5, peak


def test_contact_stops_and_holds_measured_pose():
    sim = Sim()
    lift = "arm_shoulder_lift"
    sim.walls[lift] = (-1e9, sim.pos[lift] + 5.0)  # the arm can only move 5 deg further
    sim.submit({"cmd": "joints", "targets": {lift: sim.pos[lift] + 40.0}, "speed": 30, "z_min": -1.0})
    st = sim.run_until_done()
    assert st["state"] == "failed" and ("stall" in st["reason"] or "tracking" in st["reason"]), st
    assert abs(sim.targets[lift] - sim.pos[lift]) < 1e-6  # holding where it stopped, not pushing


def _watch(cmd_z, meas_z):
    """Run AutoController._watch_contact over two height tracks (one per tick)."""
    auto = AutoController(lambda: SIM_ANGLES, clock=lambda: 0.0)
    cmd = {"watch": [], "contact_gap_m": 0.0015}
    for i, (zc, zm) in enumerate(zip(cmd_z, meas_z, strict=True)):
        # the targets carry a marker; the measured pose is rebuilt from joint_state without it
        auto._tcp_z = lambda js, zc=zc, zm=zm: zc if "commanded" in js else zm
        how = auto._watch_contact(
            cmd, i * PERIOD, {**READY, "commanded": 0.0}, {j: {"pos": 0.0} for j in IK_ARM_JOINTS}
        )
        if how:
            return how
    return ""


def test_contact_follow_check_ignores_an_arm_hanging_below_its_command():
    # Oct 27: the vertical gripper hung 4 mm below its command; the command descends 11 mm/s,
    # and the arm only follows once the command passes it.
    n = 40
    cmd = [-0.0812 - 0.011 * i * PERIOD for i in range(n)]
    meas = [min(-0.0853, c - 0.0041 + 0.004) for c in cmd]
    assert _watch(cmd, meas) == ""
    # Stopped by the floor: the command goes on, the arm doesn't.
    cmd2 = cmd + [cmd[-1] - 0.011 * i * PERIOD for i in range(1, 20)]
    meas2 = meas + [meas[-1]] * 19
    assert _watch(cmd2, meas2)


def test_contact_held_up_check_waits_until_the_arm_follows():
    # Oct 27: hanging 3.4 mm below the command, the arm stays put for the first ~0.4 s; the
    # free gap must not be learnt then (it was, and "held up" fired at 0.47 s, 24 mm up).
    n = 60
    cmd = [-0.0812 - 0.011 * i * PERIOD for i in range(n)]
    meas = [-0.0846 if i < 12 else cmd[i] - 0.0034 for i in range(n)]
    assert _watch(cmd, meas) == ""


def test_contact_detected_when_starting_on_the_brick():
    cmd = [-0.0812 - 0.011 * i * PERIOD for i in range(40)]
    meas = [-0.0815] * 40  # resting on something: never moves
    assert "from the start" in _watch(cmd, meas)


def test_loaded_but_tracking_joint_is_not_a_stall():
    sim = Sim()
    # Holding the arm out against gravity: loaded and stationary, but on target.
    orig = sim.joint_state

    def loaded():
        s = orig()
        s["arm_shoulder_lift"].update(load=900.0, severity=0.8)
        return s

    sim.joint_state = loaded
    sim.submit({"cmd": "joints", "targets": {"arm_shoulder_pan": 20.0}, "speed": 30})
    assert sim.run_until_done()["state"] == "done"


def test_heartbeat_loss_aborts_and_holds():
    sim = Sim()
    sim.submit({"cmd": "joints", "targets": {"arm_shoulder_pan": 60.0}, "speed": 10})
    sim.run(0.5)
    sim.run(HEARTBEAT_TIMEOUT_S + 0.5, heartbeat=False)
    st = sim.command()
    assert st["state"] == "aborted" and "heartbeat" in st["reason"], st
    pan = sim.pos["arm_shoulder_pan"]
    sim.run(1.0, heartbeat=False)
    assert abs(sim.pos["arm_shoulder_pan"] - pan) < 1e-6


def test_heartbeat_required_to_start():
    sim = Sim()
    sim.t += 5.0  # no heartbeat for 5 s
    try:
        sim.auto.submit({"cmd": "gripper", "pos": 10, "units": "deg"})
    except CommandRefused as e:
        assert "heartbeat" in str(e)
    else:
        raise AssertionError("expected a refusal")


def test_busy_refused():
    sim = Sim()
    sim.submit({"cmd": "joints", "targets": {"arm_shoulder_pan": 30.0}, "speed": 10})
    sim.run(0.2)
    refused(sim, {"cmd": "gripper", "pos": 0}, "busy")


def test_gripper_stalls_on_object():
    sim = Sim()
    sim.walls[GRIPPER] = (22.0, 1e9)  # a block the jaws meet 22 deg open
    sim.submit({"cmd": "gripper", "pos": 0, "speed": 50})
    st = sim.run_until_done()
    assert st["state"] == "done" and st["result"]["stalled"], st
    assert 15.0 <= sim.targets[GRIPPER] < 22.0  # squeezing a little past contact, not to 0


def test_gripper_closes_empty():
    sim = Sim()
    sim.submit({"cmd": "gripper", "pos": 0, "speed": 50})
    st = sim.run_until_done()
    assert st["state"] == "done" and not st["result"]["stalled"] and st["result"]["pos"] < 2.0, st


def holding_sim(temp=40.0):
    """A sim whose gripper has just closed and stalled on a block, 22 deg open."""
    sim = Sim()
    sim.walls[GRIPPER] = (22.0, 1e9)
    sim.extra[GRIPPER] = {"temp": temp}
    sim.submit({"cmd": "gripper", "pos": 0, "speed": 50})
    st = sim.run_until_done()
    assert st["result"]["stalled"] and st["result"]["temp"] == temp, st
    return sim


def hold(sim):
    return sim.auto.status(dict(sim.pos))["gripper_hold"]


def test_hold_stays_holding_while_the_block_is_there():
    sim = holding_sim()
    sim.run(3.0)
    h = hold(sim)
    assert h["state"] == "holding" and h["squeeze"] == 5.0 and 21.0 <= h["pos"] <= 23.0, h
    assert h["events"] == [], h


def test_hold_spots_a_drop():
    sim = holding_sim()
    sim.walls.pop(GRIPPER)  # the block slips out: the jaws travel on toward the squeeze target
    sim.run(0.5)
    h = hold(sim)
    assert h["state"] == "dropped" and "dropped" in h["events"][-1]["what"], h
    assert sim.targets[GRIPPER] >= sim.pos[GRIPPER] - 0.5  # stopped pushing on nothing


def test_temperature_eases_then_stops_the_squeeze():
    sim = holding_sim(temp=56.0)
    sim.run(0.2)
    assert hold(sim)["squeeze"] == 2.6, hold(sim)
    sim.extra[GRIPPER]["temp"] = 63.0
    sim.run(0.2)
    h = hold(sim)
    assert h["squeeze"] == 0.0 and "temperature" in h["events"][-1]["what"] and h["state"] == "holding", h
    assert sim.targets[GRIPPER] == h["pos"]  # holds where it stalled; never opens


def test_a_new_gripper_command_ends_the_hold():
    sim = holding_sim()
    sim.submit({"cmd": "gripper", "pos": 40, "speed": 50})
    assert hold(sim) is None
    sim.walls.pop(GRIPPER)
    st = sim.run_until_done()
    assert st["state"] == "done" and hold(sim) is None, (st, hold(sim))


def test_joint_commands_must_be_in_degrees():
    # Oct 29: the interface moved from normalized units to degrees; a planner still sending
    # units would have its numbers read as degrees, so it's refused instead.
    sim = Sim()
    sim.auto.heartbeat()
    for cmd in (
        {"cmd": "trajectory", "waypoints": [{"arm_shoulder_pan": 5.0}], "durations": [1.0]},
        {"cmd": "joints", "targets": {"arm_shoulder_pan": 5.0}},
        {"cmd": "gripper", "pos": 10},
    ):
        try:
            sim.auto.submit(cmd)
        except CommandRefused as e:
            assert "degrees" in str(e), e
        else:
            raise AssertionError(f"accepted {cmd} without units")
    st = sim.auto.status(dict(sim.pos))
    assert st["joint_units"] == "deg"


def test_status_and_targets_are_degrees_the_bridge_keeps_units():
    # A real calibration (this arm's, Oct 28): 1 unit is 0.88-1.52 deg depending on the joint.
    import json

    cal = json.loads(
        (
            Path(__file__).parents[1] / "gemma_brain" / "tests" / "fixtures" / "arm_calibration.json"
        ).read_text()
    )
    angles = JointAngles(cal["joints"], cal["motor_resolution"])
    auto = AutoController(lambda: angles, clock=lambda: 0.0)
    units = dict.fromkeys(IK_ARM_JOINTS, 10.0) | {GRIPPER: 20.0}
    targets = dict(units)
    auto.step(0.0, targets, {j: {"pos": v} for j, v in units.items()}, {})
    assert all(abs(targets[j] - units[j]) < 1e-9 for j in units)  # no command: the bridge's units untouched
    st = auto.status(units)
    for j in units:
        assert abs(st["joints"][j] - angles.to_deg(j, units[j])) < 1e-9
    assert abs(st["joints"]["arm_elbow_flex"] - 10.0 * 0.980) > 5  # degrees, not the units


def test_base_move_relative():
    sim = Sim()
    sim.submit({"cmd": "base", "dx": 0.30, "dy": -0.10, "dtheta_deg": 30})
    st = sim.run_until_done()
    assert st["state"] == "done", st
    odom = sim.auto.status({})["odom"]
    assert abs(odom["x"] - 0.30) < 0.015 and abs(odom["y"] + 0.10) < 0.015, odom
    assert abs(odom["theta_deg"] - 30) < 2.5, odom


def test_base_too_far_refused():
    sim = Sim()
    refused(sim, {"cmd": "base", "dx": 2.0}, "limited")


def test_base_heartbeat_loss_stops_wheels():
    sim = Sim()
    sim.submit({"cmd": "base", "dx": 0.5})
    sim.run(1.0)
    assert sim.base_cmd["x.vel"] > 0
    sim.run(HEARTBEAT_TIMEOUT_S + 0.5, heartbeat=False)
    st = sim.command()
    assert st["state"] == "aborted" and sim.base_cmd["x.vel"] == 0.0
    assert 0.0 < st["result"]["moved"][0] < 0.5, st  # how far it got, so the rest can be finished


def run_tests(namespace: dict) -> int:
    """Runs every test_* function in namespace (a module's globals()); returns an exit code."""
    tests = [(n, f) for n, f in sorted(namespace.items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_tests(globals()))


def _deg_at(joints: dict, joint: str, actual_ticks: int) -> float:
    """The degrees the planner reads for a raw encoder position: Present_Position = actual -
    Homing_Offset, normalized over the recorded range as LeRobot's bus does, then to degrees."""
    c = joints[joint]
    present = (actual_ticks - c["homing_offset"]) % 4096  # the encoder wraps
    units = (present - c["range_min"]) / (c["range_max"] - c["range_min"]) * 200 - 100
    return JointAngles(joints).to_deg(joint, units)


def test_a_shifted_zero_reads_the_true_angle():
    joints = {  # as the robot's calibration file has them (drive_mode 1: a reversed motor)
        "arm_elbow_flex": {"drive_mode": 0, "homing_offset": 2014, "range_min": 914, "range_max": 3129},
        "arm_wrist_flex": {"drive_mode": 1, "homing_offset": -1993, "range_min": 859, "range_max": 3171},
    }
    offsets = {"arm_elbow_flex": -8.0, "arm_wrist_flex": 5.0}
    fixed = shifted_calibration(joints, offsets)
    for joint, d in offsets.items():
        for actual in (1200, 1700, 2600):  # physical positions, raw encoder ticks
            assert abs(_deg_at(fixed, joint, actual) - _deg_at(joints, joint, actual) - d) < 0.1, joint
        assert (
            fixed[joint]["range_max"] - fixed[joint]["range_min"]
            == joints[joint]["range_max"] - joints[joint]["range_min"]
        )
    # This arm's elbow offset (2014) is near the register's +/-2047: -9 deg needs it wrapped.
    elbow = shifted_calibration(joints, {"arm_elbow_flex": -9.0})["arm_elbow_flex"]
    assert -2047 <= elbow["homing_offset"] <= 2047 and elbow["homing_offset"] == 2014 + 102 - 4096
    for actual in (1200, 1700, 2600):
        moved = _deg_at({"arm_elbow_flex": elbow}, "arm_elbow_flex", actual) - _deg_at(
            joints, "arm_elbow_flex", actual
        )
        assert abs(moved + 9.0) < 0.1
    with pytest.raises(ValueError):
        shifted_calibration(joints, {"arm_gripper": 1.0})
    with pytest.raises(ValueError):
        shifted_calibration(joints, {"arm_elbow_flex": 90.0})  # range_max 3129 + 1024 > 4095


# --- the stop probe (arm self-calibration) ---


def _probe(sim, joint, direction, **kw):
    return sim.submit(
        {"cmd": "probe", "joint": joint, "direction": direction, "torque": 0.3, "speed": 10.0, **kw}
    )


def test_a_probe_finds_the_joints_end_backs_off_and_puts_its_torque_back():
    sim = Sim()
    sim.walls["arm_elbow_flex"] = (-90.0, 41.0)  # the end, past where the joint starts (0)
    _probe(sim, "arm_elbow_flex", +1)
    st = sim.run_until_done()
    assert st["state"] == "done", st
    assert abs(st["result"]["stop_deg"] - 41.0) < 1.0, st["result"]
    assert abs(sim.pos["arm_elbow_flex"] - (41.0 - 3.0)) < 1.0  # backed off the end
    writes = sim.auto.take_bus_writes()
    assert writes == [("Torque_Limit", "arm_elbow_flex", 300), ("Torque_Limit", "arm_elbow_flex", None)]
    assert all(abs(sim.pos[j]) < 1e-6 for j in IK_ARM_JOINTS if j != "arm_elbow_flex")  # one joint only


def test_a_joint_lagging_under_its_weight_but_moving_is_not_an_end():
    sim = Sim()
    sim.walls["arm_shoulder_lift"] = (-100.0, 60.0)
    lag = 6.0  # deg behind its target the whole way, as a loaded joint at a low torque limit
    orig = sim.run

    def lagging(seconds, heartbeat=True):
        orig(seconds, heartbeat)
        t = sim.targets["arm_shoulder_lift"]
        sim.pos["arm_shoulder_lift"] = max(
            -100.0,
            min(60.0, t - lag if t - lag > sim.pos["arm_shoulder_lift"] else sim.pos["arm_shoulder_lift"]),
        )

    sim.run = lagging
    _probe(sim, "arm_shoulder_lift", +1)
    st = sim.run_until_done()
    assert st["state"] == "done" and abs(st["result"]["stop_deg"] - 60.0) < 1.5, st["result"]
    assert st["result"]["travelled_deg"] > 50  # it went the whole way, not stopping at the lag


def test_a_probe_with_no_end_in_reach_fails_and_still_restores_the_torque():
    sim = Sim()
    _probe(sim, "arm_wrist_roll", -1, max_travel=20.0)
    st = sim.run_until_done()
    assert st["state"] == "failed" and "no end within 20" in st["reason"]
    assert sim.auto.take_bus_writes()[-1] == ("Torque_Limit", "arm_wrist_roll", None)


def test_a_probe_stops_on_an_encoder_wrap():
    sim = Sim()
    _probe(sim, "arm_shoulder_pan", +1)
    sim.run(1.0)
    sim.pos["arm_shoulder_pan"] -= 359.0  # the reading jumps a turn: the range crossed 4095 -> 0
    st = sim.run_until_done()
    assert st["state"] == "failed" and "wrapped" in st["reason"]


def test_a_probe_losing_the_heartbeat_aborts_and_restores_the_torque():
    sim = Sim()
    _probe(sim, "arm_wrist_flex", +1)
    sim.run(0.5)
    sim.run(HEARTBEAT_TIMEOUT_S + 0.5, heartbeat=False)
    assert sim.command()["state"] == "aborted"
    assert sim.auto.take_bus_writes()[-1] == ("Torque_Limit", "arm_wrist_flex", None)


def test_bad_probes_are_refused():
    sim = Sim()
    for bad in ({"joint": "arm_nose"}, {"direction": 0}, {"torque": 0.9}, {"torque": 0.0}, {"speed": 50.0}):
        with pytest.raises(CommandRefused):
            _probe(sim, **{"joint": "arm_elbow_flex", "direction": 1, **bad})
    with pytest.raises(CommandRefused):  # degrees only, like every joint command
        sim.auto.submit({"cmd": "probe", "joint": "arm_elbow_flex", "direction": 1, "torque": 0.3})


def test_a_joint_slow_to_start_under_its_weight_is_not_an_end():
    """Stuck for a second, the motor below its torque limit (load 100 of 300), then moving."""
    sim = Sim()
    sim.walls["arm_elbow_flex"] = (-60.0, 100.0)
    orig_run, orig_state = sim.run, sim.joint_state
    held = {"until": 1.0}

    def state():
        st = orig_state()
        if sim.t < held["until"]:
            st["arm_elbow_flex"]["load"] = 100.0
        return st

    def run(seconds, heartbeat=True):
        for _ in range(max(1, round(seconds / PERIOD))):
            pos = sim.pos["arm_elbow_flex"]
            orig_run(PERIOD, heartbeat)
            if sim.t < held["until"]:
                sim.pos["arm_elbow_flex"] = pos  # gravity holds it

    sim.joint_state, sim.run = state, run
    _probe(sim, "arm_elbow_flex", -1)
    st = sim.run_until_done()
    assert st["state"] == "done" and abs(st["result"]["stop_deg"] + 60.0) < 1.0, st["result"]
