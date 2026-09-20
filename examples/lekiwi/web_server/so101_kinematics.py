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

"""
Standalone forward/inverse kinematics for the SO101 arm's 5-joint chain
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll -> gripper_frame_link).

lerobot.model.kinematics.RobotKinematics already does this via the `placo` library, but
placo has no aarch64 wheels on PyPI (verified: `pip download --platform
manylinux2014_aarch64 placo` only resolves ancient 0.1.x versions, nothing installable) --
building it from source on a Raspberry Pi 4 is the same category of risk that caused the
torch SIGILL crash this web app's episode_recorder.py was rewritten to avoid. So this
reimplements plain closed-form FK + numeric damped-least-squares IK with zero dependencies
beyond numpy, which is already proven safe on this hardware.

The joint-chain constants below (origin xyz/rpy, in meters/radians, and joint limits) are
extracted from urdf/so101_new_calib.urdf's <joint> parent/child/origin/limit tags, in
kinematic order base -> tip. That URDF is gitignored (`*.urdf`) and not guaranteed to be
present wherever this module runs, so the numbers are hardcoded here rather than parsed
from it at import time.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# (xyz, rpy) fixed transform from the previous frame to this joint's own frame; the joint
# then rotates about its local +Z axis (true for every revolute joint in this chain).
_JOINT_ORIGINS: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),  # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),  # shoulder_lift
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),  # elbow_flex
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),  # wrist_flex
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),  # wrist_roll
]
# Fixed (non-actuated) transform from wrist_roll's child frame (gripper_link) to the TCP
# frame (gripper_frame_link -- the frame lerobot.model.kinematics targets by default too).
_TCP_ORIGIN: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (-0.0079, -0.000218121, -0.0981274),
    (0.0, 3.14159, 0.0),
)

# (lower, upper) in radians, from the URDF's <limit> tags -- IK output is clamped to these.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}


def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF's rpy is applied as extrinsic (fixed-axis) rotations: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    t = np.eye(4)
    t[:3, :3] = _rotz(yaw) @ _roty(pitch) @ _rotx(roll)
    t[:3, 3] = xyz
    return t


_ORIGIN_TRANSFORMS = [_transform(xyz, rpy) for xyz, rpy in _JOINT_ORIGINS]
_TCP_TRANSFORM = _transform(*_TCP_ORIGIN)


def forward_kinematics(joint_rad: np.ndarray) -> np.ndarray:
    """4x4 pose of the TCP (gripper_frame_link) in the arm's base frame."""
    t = np.eye(4)
    for origin_t, q in zip(_ORIGIN_TRANSFORMS, joint_rad):
        rz = np.eye(4)
        rz[:3, :3] = _rotz(q)
        t = t @ origin_t @ rz
    return t @ _TCP_TRANSFORM


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: 3x3 rotation matrix for a rotation vector (axis * angle, radians).
    Used to compose a small incremental rotation onto the end-effector's current orientation
    (see robot_bridge.py's cartesian rotation jog)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9:
        return np.eye(3)
    axis = rotvec / angle
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a quaternion in (x, y, z, w) order -- the order the WebXR
    Device API reports controller/pose orientations in."""
    x, y, z, w = quat_xyzw
    n = np.linalg.norm(quat_xyzw)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    """Inverse of rotvec_to_matrix: axis*angle (radians) for a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def _rotation_error(r_current: np.ndarray, r_target: np.ndarray) -> np.ndarray:
    """Small-angle axis-angle error rotating r_current onto r_target."""
    return matrix_to_rotvec(r_target @ r_current.T)


def _numeric_jacobian(joint_rad: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """6xN Jacobian (3 position rows + 3 orientation rows) via central differences."""
    n = len(joint_rad)
    jac = np.zeros((6, n))
    t0 = forward_kinematics(joint_rad)
    for i in range(n):
        perturbed = joint_rad.copy()
        perturbed[i] += eps
        t1 = forward_kinematics(perturbed)
        jac[:3, i] = (t1[:3, 3] - t0[:3, 3]) / eps
        jac[3:, i] = _rotation_error(t0[:3, :3], t1[:3, :3]) / eps
    return jac


def inverse_kinematics(
    current_joint_rad: np.ndarray,
    target_pose: np.ndarray,
    position_weight: float = 1.0,
    orientation_weight: float = 0.05,
    max_iters: int = 50,
    tol: float = 1e-5,
    damping: float = 0.05,
    max_step_rad: float = 0.2,
) -> np.ndarray:
    """Damped-least-squares numeric IK, seeded at `current_joint_rad`.

    5 joints can't hit an arbitrary 6D pose exactly (rank-deficient by one), hence the low
    default `orientation_weight`: position tracks closely, orientation settles wherever the
    remaining degree of freedom lands. `max_step_rad` caps how far a single solve can move
    a joint, so a target near a singularity produces a bounded (if imperfect) step rather
    than a wild swing on real hardware.
    """
    q = np.array(current_joint_rad, dtype=float).copy()
    weights = np.array([position_weight] * 3 + [orientation_weight] * 3)

    for _ in range(max_iters):
        t_current = forward_kinematics(q)
        pos_err = target_pose[:3, 3] - t_current[:3, 3]
        rot_err = _rotation_error(t_current[:3, :3], target_pose[:3, :3])
        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err) < tol:
            break

        jac = _numeric_jacobian(q) * weights[:, None]
        weighted_err = err * weights
        jjt = jac @ jac.T + (damping**2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, weighted_err)
        dq = np.clip(dq, -max_step_rad, max_step_rad)
        q = q + dq

    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[name]
        q[i] = np.clip(q[i], lo, hi)
    return q
