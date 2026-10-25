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

The simulation (Sim, also used by gemma_brain/test_arm.py): each joint moves toward its
target at a capped speed, or is blocked at a "wall" (which then reports a stall); the base's
measured velocity is its command. Joint units map to radians as +/-100 units = +/-180 deg,
RobotBridge's uncalibrated fallback.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import so101_kinematics as kin  # noqa: E402
from auto_control import GRIPPER, IK_ARM_JOINTS, AutoController, CommandRefused  # noqa: E402

PERIOD = 1 / 30
READY = dict.fromkeys(IK_ARM_JOINTS, 0.0)


def norm_to_rad(joint, norm):
    return math.radians(norm / 100.0 * 180.0)


class Sim:
    def __init__(self, start=None, floor_z=None):
        self.floor_z = floor_z  # if set, the TCP can't go below it (the arm stops, like on a floor)
        self.pos = dict(start or {**READY, GRIPPER: 50.0})
        self.targets = dict(self.pos)
        self.walls = {}  # joint -> (lo, hi): the joint can't move outside this (contact)
        self.base_cmd = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self.t = 0.0
        self.auto = AutoController(norm_to_rad, clock=lambda: self.t)
        self.run(0.1)  # a few ticks so the controller has seen the arm's targets

    def joint_state(self):
        state = {}
        for j, p in self.pos.items():
            lo, hi = self.walls.get(j, (-1e9, 1e9))
            blocked = (self.targets[j] < lo - 1 and p <= lo + 0.5) or (self.targets[j] > hi + 1 and p >= hi - 0.5)
            state[j] = {"pos": p, "load": 900.0 if blocked else 50.0, "speed": 0.0, "severity": 0.8 if blocked else 0.0}
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
        self.auto.heartbeat()
        return self.auto.submit(cmd)

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
    refused(sim, {"cmd": "trajectory", "waypoints": [{"arm_shoulder_pan": 50.0}], "durations": [0.2]}, "units/s")


def test_trajectory_out_of_range_refused():
    sim = Sim()
    refused(sim, {"cmd": "trajectory", "waypoints": [{"arm_elbow_flex": 140.0}], "durations": [5.0]}, "outside")


def test_trajectory_below_floor_refused():
    sim = Sim()
    # Wrist flexed 50 units (90 deg) points the TCP down from the ready pose, to z ~0.075;
    # with z_min above that the Pi's FK check must refuse it.
    cmd = {"cmd": "trajectory", "waypoints": [{"arm_wrist_flex": 50.0}], "durations": [2.0], "z_min": 0.1}
    refused(sim, cmd, "below z_min")


def test_floor_checked_between_waypoints():
    sim = Sim()
    # A floor that only the middle of the move would touch: both ends are clear.
    sim.auto._lowest_z = lambda j: -1.0 if 20.0 < j["arm_shoulder_pan"] < 30.0 else 1.0
    cmd = {"cmd": "joints", "targets": {"arm_shoulder_pan": 50.0}, "speed": 30, "z_min": 0.0}
    refused(sim, cmd, "% of the way")


def test_done_reports_settle_error():
    sim = Sim()
    sim.walls["arm_shoulder_pan"] = (-1e9, 6.0)  # sags 4 units short of 10, no stall reported
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
    sim.walls[lift] = (-1e9, sim.pos[lift] + 5.0)  # the arm can only move 5 units further
    sim.submit({"cmd": "joints", "targets": {lift: sim.pos[lift] + 40.0}, "speed": 30, "z_min": -1.0})
    st = sim.run_until_done()
    assert st["state"] == "failed" and ("stall" in st["reason"] or "tracking" in st["reason"]), st
    assert abs(sim.targets[lift] - sim.pos[lift]) < 1e-6  # holding where it stopped, not pushing


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
    sim.run(1.0, heartbeat=False)
    st = sim.command()
    assert st["state"] == "aborted" and "heartbeat" in st["reason"], st
    pan = sim.pos["arm_shoulder_pan"]
    sim.run(1.0, heartbeat=False)
    assert abs(sim.pos["arm_shoulder_pan"] - pan) < 1e-6


def test_heartbeat_required_to_start():
    sim = Sim()
    sim.t += 5.0  # no heartbeat for 5 s
    try:
        sim.auto.submit({"cmd": "gripper", "pos": 10})
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
    sim.walls[GRIPPER] = (22.0, 1e9)  # a block 22 units wide
    sim.submit({"cmd": "gripper", "pos": 0, "speed": 50})
    st = sim.run_until_done()
    assert st["state"] == "done" and st["result"]["stalled"], st
    assert 15.0 <= sim.targets[GRIPPER] < 22.0  # squeezing a little past contact, not to 0


def test_gripper_closes_empty():
    sim = Sim()
    sim.submit({"cmd": "gripper", "pos": 0, "speed": 50})
    st = sim.run_until_done()
    assert st["state"] == "done" and not st["result"]["stalled"] and st["result"]["pos"] < 2.0, st


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
    sim.run(1.0, heartbeat=False)
    assert sim.command()["state"] == "aborted" and sim.base_cmd["x.vel"] == 0.0


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
