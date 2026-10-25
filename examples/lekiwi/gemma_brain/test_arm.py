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

"""Offline checks for the Mac-side arm planner: its plans, executed by the Pi's
AutoController in the web_server test simulation.

    .venv/bin/python test_arm.py
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web_server"))

from arm import FLOOR_CLEARANCE_M, IK_ARM_JOINTS, NAMED_POSES, Arm, Calibration, PlanError, arm_points, plan_ee_line, tool_orientation, tool_pitch_roll  # noqa: E402
from test_auto_control import Sim, run_tests  # noqa: E402

CAL = Calibration()  # uncalibrated: the same +/-100 = +/-180 deg mapping as the Sim


class SimRobot:
    """robot.Robot's primitives, run synchronously on a Sim."""

    def __init__(self, sim):
        self.sim = sim

    def status(self):
        return self.sim.auto.status(dict(self.sim.pos))

    def run(self, cmd):
        self.sim.submit(cmd)
        st = self.sim.run_until_done()
        if st["state"] != "done":
            raise RuntimeError(f"{cmd['cmd']} {st['state']}: {st['reason']}")
        return st

    def trajectory(self, waypoints, durations, z_min=None):
        return self.run({"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min})

    def joints(self, targets, speed=None, z_min=None):
        return self.run({"cmd": "joints", "targets": targets, "speed": speed, "z_min": z_min})


def test_tool_orientation_round_trip():
    assert np.allclose(tool_orientation(0.25, 0.0, 90.0, 0.0)[:, 2], [0, 0, -1], atol=1e-9)
    assert np.allclose(tool_orientation(0.25, 0.0, 0.0, 0.0)[:, 2], [1, 0, 0], atol=1e-9)
    for pitch, roll in ((90.0, 0.0), (60.0, 35.0), (10.0, -120.0)):
        got = tool_pitch_roll(tool_orientation(0.2, 0.1, pitch, roll), 0.2, 0.1)
        assert np.allclose(got, (pitch, roll), atol=1e-6), got


def test_top_down_move_reaches_goal():
    sim = Sim()
    arm = Arm(SimRobot(sim), CAL, floor_z=-0.25, body_check=False)
    goal = [0.24, 0.05, 0.02]
    arm.move_to(goal, pitch_deg=90, roll_deg=0, speed=0.06)
    pose = sim.tcp()
    assert np.linalg.norm(pose[:3, 3] - goal) < 0.006, pose[:3, 3]
    assert pose[2, 2] < -0.99, pose[:3, 2]  # approach axis pointing down


def test_plan_below_z_min_refused():
    try:
        plan_ee_line(CAL, {**NAMED_POSES["ready"]}, [0.25, 0.0, -0.10], z_min=-0.05)
    except PlanError as e:
        assert "z_min" in str(e)
    else:
        raise AssertionError("expected a refusal")


def test_unreachable_refused():
    try:
        plan_ee_line(CAL, {**NAMED_POSES["ready"]}, [0.8, 0.0, 0.1])
    except PlanError as e:
        assert "IK" in str(e) or "limit" in str(e)
    else:
        raise AssertionError("expected a refusal")


def test_planned_speeds_pass_the_pi_check():
    # Planned paths must never trip the Pi's speed cap, even for a large reorientation.
    wps, durs = plan_ee_line(CAL, {**NAMED_POSES["ready"]}, [0.2, -0.1, 0.0], pitch_deg=90, roll_deg=80, speed=0.08)
    sim = Sim()
    sim.submit({"cmd": "trajectory", "waypoints": wps, "durations": durs, "z_min": -0.2})


def test_reset_goes_home_clear_of_the_floor():
    """From a low pose, reset ends at rest, every arm point clear of the floor on the way
    (a direct joint move when that's clear, else lift + "ready" + rest)."""
    sim = Sim()
    arm = Arm(SimRobot(sim), CAL, floor_z=-0.25, body_check=False)
    arm.move_to([0.24, 0.0, 0.0], pitch_deg=90)
    lows = []
    orig_run = sim.run

    def tracking_run(seconds, heartbeat=True):
        orig_run(seconds, heartbeat)
        lows.append(arm_points(np.array([math.radians(sim.pos[j] / 100 * 180) for j in IK_ARM_JOINTS]))[:, 2].min())

    sim.run = tracking_run
    arm.reset()
    for j, v in NAMED_POSES["rest"].items():
        assert abs(sim.pos[j] - v) < 1.0, (j, sim.pos[j])
    assert min(lows) >= arm.floor_z + FLOOR_CLEARANCE_M - 0.002, min(lows)


def test_trapezoid_timing():
    # Short moves aren't padded: a 2 cm move at 3 cm/s, accel 0.15 m/s^2 takes ~0.73 s
    # (triangular profile), not 3x its constant-speed time.
    from arm import _trapezoid_time
    total = _trapezoid_time(0.02, 0.02, 0.03, 0.15)
    assert 0.6 < total < 0.9, total
    # A long move cruises: 30 cm at 3 cm/s ~ 10.2 s.
    assert 10.0 < _trapezoid_time(0.3, 0.3, 0.03, 0.15) < 10.5


def _descent(sim, depth, speed=0.01):
    arm = Arm(SimRobot(sim), CAL, floor_z=-0.5, body_check=False)
    arm.move_to([0.24, 0.0, 0.0], pitch_deg=90)
    p = sim.tcp()[:3, 3]
    wps, durs = plan_ee_line(CAL, dict(sim.targets), [p[0], p[1], p[2] - depth], pitch_deg=90, speed=speed, z_min=-0.5)
    sim.submit({"cmd": "trajectory", "waypoints": wps, "durations": durs, "z_min": -0.5, "contact_stop": True})
    return p, sim.run_until_done()


def test_contact_stop_finds_the_floor():
    sim = Sim(floor_z=-0.03)
    p0, st = _descent(Sim.__new__(Sim) if False else sim, 0.06)
    r = st["result"]
    assert st["state"] == "done" and r["contact"], st
    assert abs(r["z_meas"] - (-0.03)) < 0.004, r  # stopped at the floor
    # Holding the measured pose: not pushing on.
    assert abs(sim.tcp()[2, 3] - (-0.03)) < 0.004


def test_contact_stop_without_floor_reports_no_contact():
    sim = Sim()
    _, st = _descent(sim, 0.04)
    assert st["state"] == "done" and st["result"]["contact"] is False, st


def test_contact_stop_when_starting_on_the_floor():
    sim = Sim(floor_z=-0.0005)  # the move to the start pose ends right at the floor
    _, st = _descent(sim, 0.04)
    assert st["state"] == "done" and st["result"]["contact"], st


if __name__ == "__main__":
    sys.exit(run_tests(globals()))
