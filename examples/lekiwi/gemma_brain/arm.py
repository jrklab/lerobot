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

"""Arm planning on the Mac: IK, straight-line end-effector paths, named poses, arm reset.

Plans become joint-space trajectories (normalized joint units, the same calibration
conversion the Pi uses, fetched from /auto/calibration) that the Pi only validates and
plays back (web_server/auto_control.py).

Frames: the arm base frame of so101_kinematics, metres: +X forward, +Y left, +Z up. The
TCP's local +Z is the gripper's approach direction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web_server"))

import so101_kinematics as kin  # noqa: E402
from auto_control import JOINT_RANGE, MAX_JOINT_SPEED  # noqa: E402

IK_ARM_JOINTS = [f"arm_{name}" for name in kin.JOINT_NAMES]
GRIPPER = "arm_gripper"

# Named poses (normalized units). "rest" is the folded stow pose (RobotBridge's
# ARM_NEUTRAL_POS, what the gamepad's reset button uses); "ready" is all zeros: reaching
# forward and slightly up, clear of every joint limit.
NAMED_POSES: dict[str, dict[str, float]] = {
    "rest": {
        "arm_shoulder_pan": 0.0,
        "arm_shoulder_lift": -98.0,
        "arm_elbow_flex": 99.0,
        "arm_wrist_flex": 75.0,
        "arm_wrist_roll": 52.0,
    },
    "ready": dict.fromkeys(IK_ARM_JOINTS, 0.0),
}

# Waypoint spacing along a straight-line move. Joint-space interpolation between waypoints
# 1 cm apart stays within ~2 mm of the line. With an orientation in the target, a 5-joint
# arm never meets the IK tolerance exactly, so each solve runs its whole iteration budget:
# IK_ITERS warm-started from the previous waypoint, IK_RETRY_ITERS only when that wasn't
# enough. ~55 ms for a 27-waypoint path on an M1 Pro.
EE_STEP_M = 0.01
IK_ITERS = 10
IK_RETRY_ITERS = 60
IK_POS_TOL_M = 0.005
IK_ROT_TOL_DEG = 6.0
DEFAULT_EE_SPEED = 0.04  # m/s
MAX_EE_SPEED = 0.08
# Joint speed budget for planned paths, below the Pi's hard cap so a plan is never refused
# for a rounding-level overshoot.
PLAN_JOINT_SPEED = 0.9 * MAX_JOINT_SPEED
# Within this (normalized units, every joint) of a named pose counts as being at it.
AT_POSE_UNITS = 3.0
# Height the TCP is lifted to before reset() swings the arm home.
RESET_SAFE_Z_M = 0.10
# Floor height in the arm base frame. The owner measured the base top 100 mm above the floor
# (2026-10-23), and the first floor probe's contact point agrees (calib.json, when written,
# overrides this).
DEFAULT_FLOOR_Z_M = -0.100
# Every point of the arm (joint frames, TCP, jaw tips) stays this far above the floor on a
# free move; only grasp descents (which pass their own z_min) go closer.
FLOOR_CLEARANCE_M = 0.02
# The jaw tips extend this far past the TCP along its approach axis (estimated from the floor
# probe: contact at TCP z -0.113 against a floor at -0.100).
JAW_TIP_OFFSET_M = 0.015
CLEARANCE_SAMPLES = 40


class PlanError(ValueError):
    """No safe plan; nothing was sent to the robot."""


class Calibration:
    """Normalized joint units <-> radians, the same conversion as RobotBridge's
    _joint_norm_to_rad(): zero is the calibrated range's midpoint tick."""

    def __init__(self, joints: dict[str, dict] | None = None, motor_resolution: int = 4096):
        self.joints = joints or {}
        self.resolution = motor_resolution

    @classmethod
    def from_robot(cls, robot) -> Calibration:
        c = robot.calibration()
        return cls(c.get("joints"), c.get("motor_resolution", 4096))

    def to_rad(self, joint: str, norm: float) -> float:
        c = self.joints.get(joint)
        if not c:
            return math.radians(norm / 100.0 * 180.0)
        mid = (c["range_min"] + c["range_max"]) / 2.0
        ticks = c["range_min"] + (norm + 100.0) / 200.0 * (c["range_max"] - c["range_min"])
        rad = (ticks - mid) * (2.0 * math.pi / self.resolution)
        return -rad if c["drive_mode"] else rad

    def to_norm(self, joint: str, rad: float) -> float:
        c = self.joints.get(joint)
        if not c:
            return math.degrees(rad) / 180.0 * 100.0
        rad = -rad if c["drive_mode"] else rad
        mid = (c["range_min"] + c["range_max"]) / 2.0
        ticks = rad / (2.0 * math.pi / self.resolution) + mid
        return (ticks - c["range_min"]) / (c["range_max"] - c["range_min"]) * 200.0 - 100.0

    def q(self, joints: dict[str, float]) -> np.ndarray:
        return np.array([self.to_rad(j, joints[j]) for j in IK_ARM_JOINTS])

    def fk(self, joints: dict[str, float]) -> np.ndarray:
        return kin.forward_kinematics(self.q(joints))


def arm_points(q: np.ndarray) -> np.ndarray:
    """(N, 3) points along the arm for clearance checks: each joint frame's origin, the TCP,
    and the jaw tips, in the arm base frame."""
    t = np.eye(4)
    pts = []
    for origin, qi in zip(kin._ORIGIN_TRANSFORMS, q, strict=True):
        rz = np.eye(4)
        rz[:3, :3] = kin._rotz(qi)
        t = t @ origin @ rz
        pts.append(t[:3, 3].copy())
    tcp = t @ kin._TCP_TRANSFORM
    pts += [tcp[:3, 3], tcp[:3, 3] + JAW_TIP_OFFSET_M * tcp[:3, 2]]
    return np.array(pts)


def lowest_on_segment(calib: Calibration, a: dict[str, float], b: dict[str, float]) -> float:
    """Lowest z of any arm point along the joint-space line from a to b (the geometric path of
    both a "joints" move and a trajectory segment)."""
    lo = math.inf
    for i in range(CLEARANCE_SAMPLES + 1):
        s = i / CLEARANCE_SAMPLES
        q = calib.q({j: a[j] + (b[j] - a[j]) * s for j in IK_ARM_JOINTS})
        lo = min(lo, float(arm_points(q)[:, 2].min()))
    return lo


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tool_orientation(x: float, y: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """TCP orientation for a target at (x, y): the tool faces the target's azimuth (the only
    yaw a 5-joint arm can reach for a given (x, y)), tilted pitch_deg down from horizontal
    (90 = pointing straight down), spun roll_deg about its approach axis."""
    # At pitch 0 the approach (local Z) points along +X, local Y along +Y, local X along -Z:
    # the all-zeros pose's orientation, up to the URDF's 0.05 rad wrist offset.
    horizontal = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    yaw = math.atan2(y, x)
    return _rot_z(yaw) @ _rot_y(math.radians(pitch_deg)) @ horizontal @ _rot_z(math.radians(roll_deg))


def tool_pitch_roll(r: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Inverse of tool_orientation() for a TCP rotation r at (x, y)."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 2]))))
    rel = tool_orientation(x, y, pitch, 0.0).T @ r
    return pitch, math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def plan_ee_line(
    calib: Calibration,
    start: dict[str, float],
    goal_xyz,
    pitch_deg: float | None = None,
    roll_deg: float | None = None,
    speed: float = DEFAULT_EE_SPEED,
    z_min: float = -0.05,
) -> tuple[list[dict[str, float]], list[float]]:
    """A straight TCP line from the pose of `start` (normalized joints) to goal_xyz, as a
    trajectory: (waypoints, durations). Every waypoint is solved by IK and checked (position
    error, joint limits, z_min). With pitch_deg, the orientation is interpolated in pitch and
    roll toward it (each waypoint facing its own azimuth), never by a free rotation blend,
    which a 5-joint arm can't follow; without it the start orientation is held."""
    goal = np.asarray(goal_xyz, dtype=float)
    speed = max(0.005, min(MAX_EE_SPEED, speed))
    if goal[2] < z_min:
        raise PlanError(f"goal z {goal[2]:.3f} m is below z_min {z_min:.3f} m")
    q = calib.q(start)
    pose0 = kin.forward_kinematics(q)
    p0 = pose0[:3, 3]
    if pitch_deg is not None:
        pr0 = np.array(tool_pitch_roll(pose0[:3, :3], p0[0], p0[1]))
        pr1 = np.array([pitch_deg, roll_deg or 0.0])
        pr1[1] = pr0[1] + (pr1[1] - pr0[1] + 180.0) % 360.0 - 180.0  # shortest roll turn
        weight = 0.5
    else:
        weight = 0.05
    n = max(1, math.ceil(float(np.linalg.norm(goal - p0)) / EE_STEP_M))
    prev = {j: start[j] for j in IK_ARM_JOINTS}
    waypoints, durations = [], []
    for i in range(1, n + 1):
        a = i / n
        target = np.eye(4)
        target[:3, 3] = p0 + (goal - p0) * a
        if pitch_deg is not None:
            target[:3, :3] = tool_orientation(target[0, 3], target[1, 3], *(pr0 + (pr1 - pr0) * a))
        else:
            target[:3, :3] = pose0[:3, :3]
        q_prev = q
        q = kin.inverse_kinematics(q_prev, target, orientation_weight=weight, max_iters=IK_ITERS)
        pose = kin.forward_kinematics(q)
        err = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
        if err > IK_POS_TOL_M:
            # Along the way the orientation is a soft goal (only the final one is checked):
            # give position priority before giving up.
            q = kin.inverse_kinematics(q_prev, target, orientation_weight=weight / 3, max_iters=IK_RETRY_ITERS)
            pose = kin.forward_kinematics(q)
            err = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
        where = f"{100 * a:.0f}% of the way, at {np.round(target[:3, 3], 3).tolist()}"
        if err > IK_POS_TOL_M:
            raise PlanError(f"no IK solution {where} (position error {1000 * err:.1f} mm)")
        if pose[2, 3] < z_min - 0.002:
            raise PlanError(f"path dips to z {pose[2, 3]:.3f} m below z_min {z_min:.3f} m {where}")
        if pitch_deg is not None and i == n:
            rerr = math.degrees(float(np.linalg.norm(kin.matrix_to_rotvec(pose[:3, :3] @ target[:3, :3].T))))
            if rerr > IK_ROT_TOL_DEG:
                raise PlanError(f"goal orientation unreachable (off by {rerr:.1f} deg)")
        wp = {}
        for j, rad in zip(IK_ARM_JOINTS, q, strict=True):
            norm = calib.to_norm(j, float(rad))
            lo, hi = JOINT_RANGE[j]
            if not lo + 1.0 <= norm <= hi - 1.0:
                raise PlanError(f"{j} would hit its limit ({norm:.1f}) {where}")
            wp[j] = norm
        seg = float(np.linalg.norm(goal - p0)) / n
        joint_span = max(abs(wp[j] - prev[j]) for j in IK_ARM_JOINTS)
        durations.append(max(seg / speed, joint_span / PLAN_JOINT_SPEED, 0.01))
        waypoints.append(wp)
        prev = wp
    # Ease in and out: stretch the first and last two segments.
    for k, f in ((0, 3.0), (1, 2.0)):
        if k < len(durations):
            durations[k] *= f
            if len(durations) - 1 - k != k:
                durations[-1 - k] *= f
    return waypoints, durations


class Arm:
    """The arm skills the brain uses, on top of robot.Robot's primitives."""

    def __init__(self, robot, calib: Calibration | None = None, floor_z: float = DEFAULT_FLOOR_Z_M):
        self.robot = robot
        self.calib = calib or Calibration.from_robot(robot)
        self.floor_z = floor_z
        # The Pi's TCP floor guard for free moves (it checks the TCP only; check_clear()
        # here checks the whole arm, the jaw tips included).
        self.z_min = floor_z + FLOOR_CLEARANCE_M + JAW_TIP_OFFSET_M

    def check_clear(self, a: dict[str, float], b: dict[str, float], what: str) -> None:
        lo = lowest_on_segment(self.calib, a, b)
        if lo < self.floor_z + FLOOR_CLEARANCE_M:
            raise PlanError(
                f"{what}: the arm would pass {1000 * (lo - self.floor_z):.0f} mm above the floor "
                f"(needs {1000 * FLOOR_CLEARANCE_M:.0f} mm)"
            )

    def targets(self) -> dict[str, float]:
        """Where the arm is commanded to be (the start of the next plan)."""
        st = self.robot.status()
        return st.get("targets") or st["joints"]

    def tcp(self) -> np.ndarray:
        """Measured TCP pose (4x4) in the arm base frame."""
        return self.calib.fk(self.robot.status()["joints"])

    def move_to(
        self,
        xyz,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
        speed: float = DEFAULT_EE_SPEED,
        z_min: float | None = None,
    ) -> dict:
        z_min = self.z_min if z_min is None else z_min
        wps, durs = plan_ee_line(self.calib, self.targets(), xyz, pitch_deg, roll_deg, speed, z_min)
        return self.robot.trajectory(wps, durs, z_min=z_min)

    def move_by(self, dxyz, speed: float = DEFAULT_EE_SPEED, z_min: float | None = None) -> dict:
        """Relative straight-line move, holding the current orientation."""
        p = self.calib.fk(self.targets())[:3, 3]
        return self.move_to(p + np.asarray(dxyz, dtype=float), speed=speed, z_min=z_min)

    def pose(self, name: str, speed: float | None = None) -> dict:
        """Joint-space move to a named pose, refused unless the whole arm clears the floor
        all the way there."""
        if name not in NAMED_POSES:
            raise PlanError(f"unknown pose {name!r}; known: {sorted(NAMED_POSES)}")
        cur = self.targets()
        self.check_clear(cur, {**cur, **NAMED_POSES[name]}, f"move to {name}")
        return self.robot.joints(NAMED_POSES[name], speed=speed, z_min=self.z_min)

    def reset(self, to: str = "rest", speed: float = 15.0) -> None:
        """Back to a known pose from anywhere: lift the TCP straight up first if it's low (so
        the swing home can't drag the gripper across the floor or the block), then "ready",
        then (to="rest") the folded stow pose, every leg checked for floor clearance before
        anything moves. Slow joint speeds: an outstretched arm loads the shoulder, and a fast
        swing through such a pose trips the Pi's tracking-error abort. The gripper is left
        as it is."""
        if to not in ("rest", "ready"):
            raise PlanError("reset: 'to' must be 'rest' or 'ready'")
        cur = self.targets()
        if all(abs(cur[j] - v) < AT_POSE_UNITS for j, v in NAMED_POSES[to].items()):
            return  # already there: no lift-and-swing (it used to unfold and refold the arm)
        p = self.calib.fk(cur)[:3, 3]
        if p[2] < RESET_SAFE_Z_M:
            # Straight up, as high as the arm can go while holding its orientation (the
            # clearance check on the next leg decides whether that was high enough). From the
            # commanded pose, which may be below the floor guard after a contact.
            # Below where the jaw tips are now: the Pi checks the tips, not the TCP.
            z_min = min(self.z_min, p[2] - JAW_TIP_OFFSET_M - 0.01)
            # Pointing straight down far out, the wrist is near its limit and can't hold that
            # orientation while rising: then lift while tilting the gripper up and pulling in.
            lifts = [([p[0], p[1], z], None) for z in (RESET_SAFE_Z_M, 0.08, 0.06) if z > p[2]]
            lifts += [([p[0] - 0.04, p[1], RESET_SAFE_Z_M], 45.0), ([p[0] - 0.02, p[1], 0.06], 60.0)]
            for goal, pitch in lifts:
                try:
                    self.move_to(goal, pitch_deg=pitch, speed=0.015, z_min=z_min)
                    break
                except PlanError:
                    continue
        self.pose("ready", speed=speed)
        if to == "rest":
            self.pose("rest", speed=speed)
