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
Bridges the (synchronous, blocking) LeKiwi robot control loop to the (async) FastAPI
server via a background thread and thread-safe shared state.

Per lekiwi_web_server.md's "Incremental Implementation" directive, this can drive either
the real robot or a MockLeKiwi that only logs actions and produces synthetic camera frames,
so the web app can be developed/tested end-to-end before hardware is involved.
"""

import logging
import math
import threading
import time
from typing import Protocol

import cv2
import numpy as np

import so101_kinematics as kin
from episode_recorder import EpisodeRecorder

logger = logging.getLogger(__name__)

ARM_JOINTS = [
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
]

# Matches LeKiwiClient's existing base speed presets (xy in m/s, theta in deg/s).
BASE_SPEEDS = {
    "slow": {"xy": 0.1, "theta": 30},
    "medium": {"xy": 0.2, "theta": 60},
    "fast": {"xy": 0.3, "theta": 90},
}

# Jog speed for arm joints, in normalized units/second (joints are -100..100, gripper 0..100).
JOG_SPEEDS = {"slow": 15.0, "medium": 30.0, "fast": 50.0}

# Jog speed for IK-driven cartesian jogging, in meters/second. The first real-hardware try
# used much more conservative numbers (0.02/0.04/0.07) since this control path was
# unverified; confirmed correct since (calibration fix + ready-pose ramp), raised here after
# that first try felt too slow even at "fast".
CARTESIAN_JOG_SPEEDS = {"slow": 0.05, "medium": 0.12, "fast": 0.22}

# Jog speed for IK-driven end-effector rotation, in radians/second. Raised (like
# CARTESIAN_JOG_SPEEDS before it) after the initial conservative values felt too slow once
# verified working.
CARTESIAN_ROT_JOG_SPEEDS = {"slow": 1.5, "medium": 3.0, "fast": 5.0}

# Orientation task weight for cartesian jogging's IK solve, passed to
# so101_kinematics.inverse_kinematics(). Higher than that function's own default (0.05,
# chosen back when nothing ever commanded rotation and orientation only needed to *not
# drift* while position moved) so a commanded roll/pitch/yaw actually gets tracked --
# still below position_weight's default of 1.0 since 5 joints can't hit an arbitrary 6D
# target exactly (rank-deficient by one DOF): position keeps priority when the two conflict.
CARTESIAN_JOG_ORIENTATION_WEIGHT = 0.3

# Motion scale for VR absolute-pose tracking: how much of the controller's actual hand
# motion (position delta and rotation angle, both scaled by the same factor) gets applied to
# the end-effector target. 1:1 (fast) felt too twitchy for fine control on the first real
# try, hence slow/medium options that deliberately under-track hand motion for more precise
# positioning -- unlike CARTESIAN_JOG_SPEEDS/CARTESIAN_ROT_JOG_SPEEDS this isn't a rate (the
# control itself is absolute-pose, not rate-based jogging), it's a unitless gain.
VR_MOTION_SCALE = {"slow": 0.3, "medium": 0.6, "fast": 1.0}

# Feetech STS3215 encoder resolution (ticks/revolution) -- used to convert between the
# calibrated normalized (-100..100) joint range and physical radians for IK. Matches the
# resolution implied by this robot's calibration file (base wheel motors calibrate over the
# full 0-4095 tick range).
MOTOR_RESOLUTION = 4096

# The 5 arm joints that make up so101_kinematics's chain, in the same order as
# so101_kinematics.JOINT_NAMES (i.e. base -> tip). Excludes arm_gripper, which doesn't
# affect end-effector pose and is still jogged directly.
IK_ARM_JOINTS = [f"arm_{name}" for name in kin.JOINT_NAMES]

# All 5 IK-chain joints at normalized 0 -- verified (see so101_kinematics.py's MuJoCo
# cross-check) to correspond to the arm reaching forward and slightly up, well clear of any
# joint limit. Cartesian jogging starts by ramping here rather than jogging immediately from
# wherever the arm was: the first real test of this feature started from ARM_NEUTRAL_POS (a
# deeply-folded stowed pose, near several joint limits), where even a correct IK solve needs
# disproportionately large joint swings for a small end-effector motion -- it looked like a
# bug but was really just a bad starting configuration.
CARTESIAN_READY_POSE = dict.fromkeys(IK_ARM_JOINTS, 0.0)
CARTESIAN_READY_RAMP_S = 1.5

# Per-joint valid range for clamping jog targets.
JOINT_RANGE = dict.fromkeys(
    [j for j in ARM_JOINTS if j != "arm_gripper"],
    (-100.0, 100.0),
)
JOINT_RANGE["arm_gripper"] = (0.0, 100.0)

# Neutral/rest pose in normalized units, matching ARM_NEUTRAL_POS_NORM in
# examples/lekiwi/gamepad_teleoperate.py (LeKiwi's default is use_degrees=False).
ARM_NEUTRAL_POS = {
    "arm_shoulder_pan": 0.0,
    "arm_shoulder_lift": -98.0,
    "arm_elbow_flex": 99.0,
    "arm_wrist_flex": 75.0,
    "arm_wrist_roll": 52.0,
    "arm_gripper": 2.0,
}

WATCHDOG_TIMEOUT_S = 0.4
# Matches examples/lekiwi/teleoperate.py's single combined read+write loop rate.
FPS = 30

# How long get_observation() must keep failing (e.g. a wedged USB camera -- see
# RobotBridge._maybe_recover_cameras()) before attempting a camera reconnect. Long enough
# not to trigger on a single transient blip (OpenCVCamera's own read loop already absorbs up
# to ~10 consecutive read errors on its own before it gives up), short enough that "a few
# frames freeze, then it recovers" doesn't turn into "stuck until a manual reboot".
CAMERA_WATCHDOG_TIMEOUT_S = 2.0
# Minimum time between reconnect attempts, so a camera that's genuinely gone (unplugged,
# truly dead) doesn't get hammered with repeated multi-second connect() retries forever.
CAMERA_RECONNECT_COOLDOWN_S = 8.0

# Only one control source drives the robot at a time. "gamepad" and "web" feed
# update_base()/update_jog() (jog deltas); "leader_keyboard" feeds update_base() (from
# keyboard key state, same as "web") and set_arm_absolute() (direct position mirror from a
# leader arm on the host PC -- see leader_keyboard_client.py); "vr" feeds update_vr_pose()
# (absolute end-effector pose with a clutch -- see that method and _apply_vr_target()).
CONTROL_MODES = ("gamepad", "web", "leader_keyboard", "vr")

# Rotates a WebXR-frame vector/rotation into the arm's base frame. WebXR's `local`/
# `local-floor` reference space is +X=right, +Y=up, +Z=toward the user (so -Z=forward); the
# arm's base frame (from forward_kinematics(zeros(5))'s reach direction) is +X=forward,
# +Y=left, +Z=up. This is a first guess at what "feels right" moving a hand -- like the
# gamepad's D-pad/stick sign conventions, expect to retune after trying it on the headset.
WEBXR_TO_ARM_FRAME = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

# Stall detection, per examples/lekiwi/torque_feedback.md: a motor is "stalled" when it's
# under significant load (Present_Load, raw 0-1000) while barely moving (Present_Velocity,
# raw magnitude) -- high load alone can just mean fast acceleration, so the speed gate
# guards against false positives on an unloaded but quickly-moving joint.
# Per-motor load thresholds match torque_feedback.md's recommended `per_motor_thresholds`.
STALL_LOAD_THRESHOLDS = {
    "arm_shoulder_pan": 0.3,
    "arm_shoulder_lift": 0.5,
    "arm_elbow_flex": 0.5,
    "arm_wrist_flex": 0.5,
    "arm_wrist_roll": 0.3,
    "arm_gripper": 0.2,
}
STALL_SPEED_THRESHOLD = 50  # raw Present_Velocity magnitude, matches torque_feedback.md's example

# How long a leader-arm mode entry takes to ramp from the arm's current pose up to the
# leader's live pose (see RobotBridge.set_arm_absolute()).
LEADER_CATCHUP_S = 1.5


def _stall_severity(joint: str, load: float, speed: float) -> float:
    """0.0 = not stalled; up to 1.0 = load at the motor's full-scale limit.

    Same normalisation as torque_feedback.md's Layer 2 (L_norm), reused here as a stall
    *severity* metric instead of a torque-limit scale -- it's the same "how far past the
    threshold is this load" question either way.
    """
    if speed > STALL_SPEED_THRESHOLD:
        return 0.0
    threshold = STALL_LOAD_THRESHOLDS[joint] * 1000
    if load <= threshold:
        return 0.0
    return min(1.0, (load - threshold) / (1000 - threshold))


class RobotLike(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def send_action(self, action: dict) -> dict: ...
    def get_observation(self) -> dict: ...
    def stop_base(self) -> None: ...
    action_features: dict
    observation_features: dict


class MockLeKiwi:
    """Stands in for `LeKiwi` so the web app can be built/tested without hardware."""

    def __init__(self):
        self._joint_pos = dict.fromkeys(ARM_JOINTS, 0.0)
        self._frame_counter = 0

    def connect(self) -> None:
        logger.info("MockLeKiwi connected (no hardware).")

    def disconnect(self) -> None:
        logger.info("MockLeKiwi disconnected.")

    def stop_base(self) -> None:
        pass

    @property
    def action_features(self) -> dict:
        return dict.fromkeys([f"{j}.pos" for j in ARM_JOINTS] + ["x.vel", "y.vel", "theta.vel"], float)

    @property
    def observation_features(self) -> dict:
        return {**self.action_features, "front": (480, 640, 3), "wrist": (480, 640, 3)}

    def send_action(self, action: dict) -> dict:
        for joint in ARM_JOINTS:
            key = f"{joint}.pos"
            if key in action:
                self._joint_pos[joint] = action[key]
        logger.debug("MockLeKiwi send_action: %s", action)
        return action

    def get_observation(self) -> dict:
        self._frame_counter += 1
        obs = {f"{joint}.pos": pos for joint, pos in self._joint_pos.items()}
        obs["x.vel"] = 0.0
        obs["y.vel"] = 0.0
        obs["theta.vel"] = 0.0
        # Fake a slowly-oscillating load with no real hardware, purely so the load/stall
        # readout is visible while developing the frontend with --mock.
        fake_load = abs((self._frame_counter * 7) % 1200 - 600)
        for joint in ARM_JOINTS:
            obs[f"{joint}.load"] = float(fake_load)
            obs[f"{joint}.speed"] = 0.0
        for cam_name in ("front", "wrist"):
            obs[cam_name] = self._synthetic_frame(cam_name)
        return obs

    def _synthetic_frame(self, cam_name: str) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (60, 40, 20) if cam_name == "front" else (20, 40, 60)
        t = self._frame_counter
        cv2.putText(
            frame, f"MOCK {cam_name}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
        )
        cv2.circle(frame, (320 + int(200 * np.sin(t / 20)), 240), 20, (0, 200, 255), -1)
        return frame


class RobotBridge:
    """Owns the robot connection and a background control-loop thread.

    The FastAPI layer only ever touches thread-safe methods here (`update_base`,
    `update_jog`, `get_jpeg_frame`, `get_joint_state`); all robot I/O happens on the
    background thread.
    """

    def __init__(self, robot: RobotLike, recorder: EpisodeRecorder | None = None):
        self._robot = robot
        self._recorder = recorder
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Playback: None means not replaying. While active, the control loop steps through
        # these recorded actions instead of computing one from the current control mode.
        self._playback_actions: list[dict] | None = None
        self._playback_index = 0
        self._playback_episode: int | None = None

        self._control_mode = "gamepad"

        self._desired_base = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
        self._desired_jogs: dict[str, dict] = {}  # joint -> {"dir": -1|0|1, "speed": str}
        # IK-driven cartesian jog: {"dx", "dy", "dz", "speed"}, each of dx/dy/dz -1..1. Mutually
        # exclusive with self._desired_jogs in practice (only one gamepad/web arm sub-mode is
        # ever live at a time), sharing the same jog watchdog (_last_jog_msg_time).
        self._desired_cartesian_jog: dict | None = None
        # Separate watchdog timestamps: arm jog traffic must never mask a stale base
        # command (or vice versa) — each input source goes stale independently.
        self._last_base_msg_time = 0.0
        self._last_jog_msg_time = 0.0

        self._joint_targets: dict[str, float] = dict.fromkeys(ARM_JOINTS, 0.0)
        self._latest_joint_state: dict[str, dict] = {
            joint: {"pos": 0.0, "load": 0.0, "speed": 0.0, "stalled": False, "severity": 0.0}
            for joint in ARM_JOINTS
        }
        self._latest_jpeg: dict[str, bytes] = {}

        # Leader-arm mirroring ramps in over LEADER_CATCHUP_S on entry to "leader_keyboard"
        # instead of snapping straight to the leader's (possibly very different) live pose --
        # see set_control_mode()/set_arm_absolute(). Deliberately NOT a blanket per-tick clamp
        # (like max_relative_target) since that would also cap legitimate fast hand motion
        # during normal teleoperation, not just the one-time mode-entry jump.
        self._leader_catchup_start: dict[str, float] = {}
        self._leader_catchup_until = 0.0

        # Cartesian-jog-entry ramp: mirrors the leader-catchup pattern above, but ramps
        # IK_ARM_JOINTS to CARTESIAN_READY_POSE on entering the arm's cartesian/IK jog
        # sub-mode (see set_arm_submode()) instead of jogging immediately from wherever the
        # arm was. While ramping, incoming cartesian jog deltas are ignored (see
        # _control_loop) so an automatic move and a live jog input can't fight each other.
        self._cartesian_ready_ramp_start: dict[str, float] = {}
        self._cartesian_ready_ramp_until = 0.0

        # VR (Quest) absolute-pose-with-clutch tracking -- see update_vr_pose()/
        # _apply_vr_target(). _vr_enabled is separate instance state (not part of
        # _desired_vr_target) purely to detect the clutch's rising edge across calls;
        # _desired_vr_target is the immutable-once-built snapshot _control_loop reads.
        self._vr_enabled = False
        self._vr_reference_ee_pose: np.ndarray | None = None
        self._vr_reference_controller_pose: np.ndarray | None = None
        self._desired_vr_target: dict | None = None
        self._last_vr_msg_time = 0.0

        # Camera-freeze recovery: a USB UVC camera can wedge (its background read thread
        # blocks inside the OS/driver forever, or keeps returning a frame whose timestamp
        # never advances) without ever raising from send_action() -- get_observation() is
        # the only thing that notices, via Camera.read_latest()'s max_age_ms staleness check
        # (LeKiwi.get_observation() reads motor state first, then cameras, so a stuck camera
        # also freezes the joint-state/load readout, not just video, even though motor
        # control itself keeps working since send_action() is a separate, unaffected call).
        # Previously the only fix was rebooting the Pi; see _maybe_recover_cameras().
        self._obs_failure_since: float | None = None
        self._last_camera_reconnect_attempt = 0.0
        self._camera_reconnect_in_progress = False

    def start(self) -> None:
        self._robot.connect()

        if self._recorder is not None:
            try:
                self._recorder.configure_features(self._robot.action_features, self._robot.observation_features)
            except Exception:
                logger.exception("Failed to configure episode recorder features -- recording disabled")
                self._recorder = None

        obs = self._robot.get_observation()
        for joint in ARM_JOINTS:
            key = f"{joint}.pos"
            if key in obs:
                self._joint_targets[joint] = obs[key]
                self._latest_joint_state[joint]["pos"] = obs[key]
        logger.info("RobotBridge starting with initial joint targets: %s", self._joint_targets)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._control_loop, name="robot_bridge_loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._robot.stop_base()
        finally:
            self._robot.disconnect()

    # --- called from the FastAPI event loop (thread-safe) ---

    def get_control_mode(self) -> str:
        with self._lock:
            return self._control_mode

    def set_control_mode(self, mode: str) -> None:
        if mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control mode: {mode!r}")
        with self._lock:
            self._control_mode = mode
            # Switching away from a source must not leave its last command latched in --
            # the next thing to touch the base/arm should be whatever mode is now active.
            self._desired_base = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
            self._desired_jogs.clear()
            self._desired_cartesian_jog = None
            self._cartesian_ready_ramp_until = 0.0
            self._cartesian_ready_ramp_start = {}
            self._desired_vr_target = None
            self._vr_enabled = False
            if mode == "leader_keyboard":
                self._leader_catchup_start = dict(self._joint_targets)
                self._leader_catchup_until = time.monotonic() + LEADER_CATCHUP_S
        logger.info("Control mode switched to: %s", mode)

    def update_base(self, x: float, y: float, theta: float, speed: str, source: str) -> None:
        with self._lock:
            if source != self._control_mode:
                return
            self._desired_base = {"x": x, "y": y, "theta": theta, "speed": speed}
            self._last_base_msg_time = time.monotonic()

    def update_jog(self, joint: str, direction: float, speed: str, source: str) -> None:
        """direction is -1..1: the web UI's hold-to-jog buttons only ever send -1/0/1,
        but the gamepad passes a continuous analog stick value through this same path."""
        if joint not in ARM_JOINTS:
            raise ValueError(f"Unknown joint: {joint}")
        with self._lock:
            if source != self._control_mode:
                return
            self._desired_jogs[joint] = {"dir": direction, "speed": speed}
            self._last_jog_msg_time = time.monotonic()

    def update_cartesian_jog(
        self,
        dx: float,
        dy: float,
        dz: float,
        droll: float,
        dpitch: float,
        dyaw: float,
        speed: str,
        source: str,
    ) -> None:
        """IK-driven end-effector jog: dx/dy/dz are -1..1 translation in the arm's base
        frame; droll/dpitch/dyaw are -1..1 rotation about the end-effector's own (body-frame)
        axes -- analogous to update_jog()'s per-joint direction but applied via
        so101_kinematics instead."""
        with self._lock:
            if source != self._control_mode:
                return
            self._desired_cartesian_jog = {
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "droll": droll,
                "dpitch": dpitch,
                "dyaw": dyaw,
                "speed": speed,
            }
            self._last_jog_msg_time = time.monotonic()

    def set_arm_submode(self, submode: str, source: str) -> None:
        """Called when a control source enters/leaves the arm's cartesian (IK) jog
        sub-mode. Entering starts a ramp to CARTESIAN_READY_POSE (see that constant's
        comment); leaving cancels any ramp still in progress."""
        if submode not in ("joint", "cartesian"):
            raise ValueError(f"Unknown arm submode: {submode!r}")
        with self._lock:
            if source != self._control_mode:
                return
            self._desired_cartesian_jog = None
            if submode == "cartesian":
                self._cartesian_ready_ramp_start = dict(self._joint_targets)
                self._cartesian_ready_ramp_until = time.monotonic() + CARTESIAN_READY_RAMP_S
            else:
                self._cartesian_ready_ramp_until = 0.0
                self._cartesian_ready_ramp_start = {}

    def update_vr_pose(
        self,
        position: list[float],
        quaternion: list[float],
        grip: bool,
        trigger: float,
        speed: str,
        source: str,
    ) -> None:
        """VR controller input: position ([x,y,z], meters) and quaternion ([x,y,z,w]) in the
        WebXR controller's own reference-space frame (converted to the arm's base frame here
        via WEBXR_TO_ARM_FRAME). grip is the clutch -- while held, the end-effector tracks
        the controller's motion *relative to where it was when grip was first pressed*
        (latched below), not its absolute pose; releasing freezes the arm in place, and
        squeezing again re-latches from wherever the controller physically is by then. This
        mirrors lerobot's own EEReferenceAndDelta pattern (robot_kinematic_processor.py),
        reimplemented here since that path pulls in placo/torch. trigger is 0..1, mapped
        directly to gripper position (0=open, 1=closed, i.e. squeeze to close). speed selects
        VR_MOTION_SCALE, applied to the tracked motion in _apply_vr_target()."""
        with self._lock:
            if source != self._control_mode:
                return

            pos_arm = WEBXR_TO_ARM_FRAME @ np.array(position, dtype=float)
            rot_arm = (
                WEBXR_TO_ARM_FRAME @ kin.quat_to_matrix(np.array(quaternion, dtype=float)) @ WEBXR_TO_ARM_FRAME.T
            )
            controller_pose = np.eye(4)
            controller_pose[:3, :3] = rot_arm
            controller_pose[:3, 3] = pos_arm

            if grip and not self._vr_enabled:
                current_rad = np.array(
                    [self._joint_norm_to_rad(j, self._joint_targets[j]) for j in IK_ARM_JOINTS]
                )
                self._vr_reference_ee_pose = kin.forward_kinematics(current_rad)
                self._vr_reference_controller_pose = controller_pose
            self._vr_enabled = grip

            lo, hi = JOINT_RANGE["arm_gripper"]
            gripper_norm = hi - max(0.0, min(1.0, trigger)) * (hi - lo)

            self._desired_vr_target = {
                "enabled": grip,
                "reference_ee_pose": self._vr_reference_ee_pose,
                "reference_controller_pose": self._vr_reference_controller_pose,
                "controller_pose": controller_pose,
                "gripper_norm": gripper_norm,
                "scale": VR_MOTION_SCALE.get(speed, VR_MOTION_SCALE["medium"]),
            }
            self._last_vr_msg_time = time.monotonic()

    def _apply_vr_target(self, vr_target: dict) -> None:
        """Applies a VR absolute-pose-with-clutch target (see update_vr_pose()), mutating
        self._joint_targets in place. Called from _control_loop only.

        The gripper tracks the trigger unconditionally -- it's an independent control from
        the grip/clutch, not gated by whether arm tracking is currently enabled."""
        lo, hi = JOINT_RANGE["arm_gripper"]
        self._joint_targets["arm_gripper"] = max(lo, min(hi, vr_target["gripper_norm"]))

        if not vr_target["enabled"] or vr_target["reference_ee_pose"] is None:
            return
        ref_ee = vr_target["reference_ee_pose"]
        ref_ctrl = vr_target["reference_controller_pose"]
        cur_ctrl = vr_target["controller_pose"]

        scale = vr_target["scale"]
        delta_pos = (cur_ctrl[:3, 3] - ref_ctrl[:3, 3]) * scale
        delta_rotvec = kin.matrix_to_rotvec(cur_ctrl[:3, :3] @ ref_ctrl[:3, :3].T) * scale
        delta_rot = kin.rotvec_to_matrix(delta_rotvec)

        target_pose = np.eye(4)
        target_pose[:3, 3] = ref_ee[:3, 3] + delta_pos
        target_pose[:3, :3] = delta_rot @ ref_ee[:3, :3]

        current_rad = np.array(
            [self._joint_norm_to_rad(j, self._joint_targets[j]) for j in IK_ARM_JOINTS]
        )
        solved_rad = kin.inverse_kinematics(
            current_rad, target_pose, orientation_weight=CARTESIAN_JOG_ORIENTATION_WEIGHT
        )
        for joint, rad in zip(IK_ARM_JOINTS, solved_rad):
            norm = self._joint_rad_to_norm(joint, rad)
            lo, hi = JOINT_RANGE[joint]
            self._joint_targets[joint] = max(lo, min(hi, norm))

    def _joint_norm_to_rad(self, joint: str, norm: float) -> float:
        """Converts a joint's normalized (-100..100) commanded position to a physical angle
        in radians, using this robot's own calibration (falls back to assuming the full
        -100..100 range spans +/-180 degrees when uncalibrated, e.g. MockLeKiwi).

        Zero must be the calibrated range's *midpoint* tick, not tick 0 -- matches how this
        codebase's own MotorNormMode.DEGREES defines "degrees" (see motors_bus.py's
        `_normalize`: `mid = (min_ + max_) / 2`). Missing that centering here originally
        put every joint's computed angle off by up to ~180 degrees, which is what made the
        arm's first real cartesian-jog attempt lurch to a wildly wrong pose: the IK solve's
        "current pose" (from forward_kinematics on the wrong angles) didn't correspond to
        the arm's actual physical pose at all.
        """
        calib = getattr(self._robot, "calibration", None)
        if calib and joint in calib:
            c = calib[joint]
            mid_ticks = (c.range_min + c.range_max) / 2.0
            ticks = c.range_min + (norm + 100.0) / 200.0 * (c.range_max - c.range_min)
            rad = (ticks - mid_ticks) * (2.0 * math.pi / MOTOR_RESOLUTION)
            return -rad if c.drive_mode else rad
        return math.radians(norm / 100.0 * 180.0)

    def _joint_rad_to_norm(self, joint: str, rad: float) -> float:
        calib = getattr(self._robot, "calibration", None)
        if calib and joint in calib:
            c = calib[joint]
            rad = -rad if c.drive_mode else rad
            mid_ticks = (c.range_min + c.range_max) / 2.0
            ticks = rad / (2.0 * math.pi / MOTOR_RESOLUTION) + mid_ticks
            return (ticks - c.range_min) / (c.range_max - c.range_min) * 200.0 - 100.0
        return math.degrees(rad) / 180.0 * 100.0

    def set_arm_absolute(self, positions: dict[str, float]) -> None:
        """Mirrors a leader arm's positions directly onto the arm's targets (no jog delta).

        Only takes effect in "leader_keyboard" mode. For LEADER_CATCHUP_S after entering that
        mode, targets ramp linearly from wherever the arm was to the leader's live pose instead
        of snapping there in one tick -- a leader/follower pose mismatch at the moment of the
        switch is expected, not a bug, so this is a one-time smoothing, not a standing limit on
        how fast normal teleoperation can move the arm afterward.
        """
        with self._lock:
            if self._control_mode != "leader_keyboard":
                return
            self._desired_jogs.clear()
            self._desired_cartesian_jog = None
            self._cartesian_ready_ramp_until = 0.0
            self._cartesian_ready_ramp_start = {}
            self._desired_vr_target = None
            self._vr_enabled = False

            now = time.monotonic()
            remaining = self._leader_catchup_until - now
            alpha = 1.0 if remaining <= 0 else 1.0 - (remaining / LEADER_CATCHUP_S)

            for joint, pos in positions.items():
                if joint not in ARM_JOINTS:
                    continue
                lo, hi = JOINT_RANGE[joint]
                pos = max(lo, min(hi, pos))
                if alpha < 1.0:
                    start = self._leader_catchup_start.get(joint, pos)
                    pos = start + (pos - start) * alpha
                self._joint_targets[joint] = pos

    def reset_arm(self) -> None:
        """Sends the arm to its neutral/rest pose (see ARM_NEUTRAL_POS)."""
        with self._lock:
            self._desired_jogs.clear()
            self._desired_cartesian_jog = None
            self._cartesian_ready_ramp_until = 0.0
            self._cartesian_ready_ramp_start = {}
            self._desired_vr_target = None
            self._vr_enabled = False
            for joint, pos in ARM_NEUTRAL_POS.items():
                self._joint_targets[joint] = pos

    def emergency_stop(self) -> None:
        """Halts all motion: zeroes base velocity and cancels any in-progress arm jog.

        Deliberately does NOT move the arm to any position (neutral or otherwise) -- an
        e-stop must never itself cause a large, sudden, potentially unsafe motion. Position
        control already holds the arm wherever it currently is with no action needed.
        """
        with self._lock:
            self._desired_base = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
            self._last_base_msg_time = time.monotonic()
            for joint in self._desired_jogs:
                self._desired_jogs[joint]["dir"] = 0.0
            self._desired_cartesian_jog = None
            self._cartesian_ready_ramp_until = 0.0
            self._cartesian_ready_ramp_start = {}
            self._desired_vr_target = None
            self._vr_enabled = False
            self._last_jog_msg_time = time.monotonic()

    def get_jpeg_frame(self, cam_name: str) -> bytes | None:
        with self._lock:
            return self._latest_jpeg.get(cam_name)

    def get_joint_state(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._latest_joint_state)

    # --- recording / playback (thread-safe; see episode_recorder.py for the dataset side) ---

    def start_recording(self, task: str) -> bool:
        if self._recorder is None:
            return False
        with self._lock:
            if self._playback_actions is not None:
                return False  # can't record a replay
        return self._recorder.start_recording(task)

    def stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.stop_recording()

    def discard_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.discard_recording()

    def upload_to_hub(self) -> None:
        if self._recorder is not None:
            self._recorder.upload_to_hub()

    def delete_episode(self, episode_index: int) -> bool:
        if self._recorder is None:
            return False
        with self._lock:
            if self._playback_episode == episode_index:
                return False  # don't delete out from under an active playback
        return self._recorder.delete_episode(episode_index)

    def start_playback(self, episode_index: int) -> str | None:
        """Returns None on success, or a user-facing reason string on failure -- distinct
        reasons ("busy" vs. "episode failed to load") matter for the error the operator
        sees; conflating them into a single bool made a real bug (a stale bounds check
        rejecting valid episodes after a delete) look identical to normal contention."""
        if self._recorder is None:
            return "Recording is disabled on this server."
        with self._lock:
            if self._recorder.is_recording or self._recorder.is_saving:
                return "A recording is in progress."
            if self._playback_actions is not None:
                return "Another playback is already in progress."
        # Loading the episode (parsing the video/parquet index) can take a moment -- do it
        # outside the lock so it doesn't stall update_base()/update_jog() callers meanwhile.
        actions = self._recorder.load_episode_actions(episode_index)
        if not actions:
            return f"Episode {episode_index} has no recorded frames or failed to load."
        with self._lock:
            self._playback_actions = actions
            self._playback_index = 0
            self._playback_episode = episode_index
        logger.info("Playback started: episode %d (%d frames)", episode_index, len(actions))
        return None

    def stop_playback(self) -> None:
        with self._lock:
            self._finish_playback_locked()

    def _finish_playback_locked(self) -> None:
        """Caller must hold self._lock. Ends playback and clears any input latched during
        it, so whichever control mode is active resumes from a clean, stopped state rather
        than a stale command from before/during the replay."""
        if self._playback_actions is None:
            return
        logger.info("Playback finished: episode %s", self._playback_episode)
        self._playback_actions = None
        self._playback_index = 0
        self._playback_episode = None
        self._desired_base = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
        self._desired_jogs.clear()
        self._desired_cartesian_jog = None
        self._cartesian_ready_ramp_until = 0.0
        self._cartesian_ready_ramp_start = {}
        self._desired_vr_target = None
        self._vr_enabled = False

    def get_recording_status(self) -> dict:
        status = (
            self._recorder.get_status()
            if self._recorder is not None
            else {
                "recording": False,
                "task": "",
                "elapsed_s": 0.0,
                "frame_count": 0,
                "episodes": [],
                "upload_status": "idle",
                "upload_message": "",
            }
        )
        with self._lock:
            status["playback_active"] = self._playback_actions is not None
            status["playback_episode"] = self._playback_episode
            status["playback_progress"] = (
                self._playback_index / len(self._playback_actions) if self._playback_actions else 0.0
            )
        return status

    # --- background thread ---

    def _apply_cartesian_jog(self, cart_jog: dict, period: float) -> None:
        """Moves the end-effector by a small cartesian delta via IK, mutating
        self._joint_targets for IK_ARM_JOINTS in place. Called from _control_loop only
        (not thread-safe on its own -- relies on the caller already being on that thread,
        same as the rest of the per-tick target math it sits alongside)."""
        speed = CARTESIAN_JOG_SPEEDS[cart_jog["speed"]]
        rot_speed = CARTESIAN_ROT_JOG_SPEEDS[cart_jog["speed"]]
        delta = np.array([cart_jog["dx"], cart_jog["dy"], cart_jog["dz"]]) * speed * period
        delta_rotvec = (
            np.array([cart_jog["droll"], cart_jog["dpitch"], cart_jog["dyaw"]]) * rot_speed * period
        )

        current_rad = np.array(
            [self._joint_norm_to_rad(j, self._joint_targets[j]) for j in IK_ARM_JOINTS]
        )
        current_pose = kin.forward_kinematics(current_rad)
        target_pose = current_pose.copy()
        target_pose[:3, 3] += delta
        # Body-frame (right-multiplied) rotation: roll/pitch/yaw are about the end-effector's
        # own axes, not the arm's base frame -- e.g. "roll" always spins the gripper about
        # its own pointing axis, whatever direction that currently is.
        target_pose[:3, :3] = current_pose[:3, :3] @ kin.rotvec_to_matrix(delta_rotvec)

        solved_rad = kin.inverse_kinematics(
            current_rad, target_pose, orientation_weight=CARTESIAN_JOG_ORIENTATION_WEIGHT
        )
        for joint, rad in zip(IK_ARM_JOINTS, solved_rad):
            norm = self._joint_rad_to_norm(joint, rad)
            lo, hi = JOINT_RANGE[joint]
            self._joint_targets[joint] = max(lo, min(hi, norm))

    def _maybe_recover_cameras(self, now: float) -> None:
        """Called from _control_loop after a get_observation() failure. Tracks how long
        failures have been continuous and, past CAMERA_WATCHDOG_TIMEOUT_S (with
        CAMERA_RECONNECT_COOLDOWN_S between attempts), kicks off a background reconnect of
        every camera -- see the __init__ comment above self._obs_failure_since for why a
        camera can wedge without send_action() ever noticing."""
        if self._obs_failure_since is None:
            self._obs_failure_since = now
            return
        if now - self._obs_failure_since < CAMERA_WATCHDOG_TIMEOUT_S:
            return
        if now - self._last_camera_reconnect_attempt < CAMERA_RECONNECT_COOLDOWN_S:
            return
        if self._camera_reconnect_in_progress:
            return

        cameras = getattr(self._robot, "cameras", None)
        if not cameras:
            return  # e.g. MockLeKiwi, or a robot type with no camera attribute at all

        self._last_camera_reconnect_attempt = now
        self._camera_reconnect_in_progress = True
        threading.Thread(
            target=self._reconnect_cameras_worker, args=(cameras,), name="camera_recovery", daemon=True
        ).start()

    def _reconnect_cameras_worker(self, cameras: dict) -> None:
        """Disconnects and reconnects every camera. Runs on its own thread -- Camera.connect()
        can block for several seconds (retry-with-backoff on a slow-to-settle USB device, see
        camera_opencv.py's connect()), which would otherwise stall send_action() and freeze
        arm/base motion too, not just the video, for the whole reconnect duration."""
        try:
            for cam_key, cam in cameras.items():
                try:
                    logger.warning(
                        "Camera %r: get_observation() has failed for >%.1fs, attempting reconnect",
                        cam_key,
                        CAMERA_WATCHDOG_TIMEOUT_S,
                    )
                    cam.disconnect()
                except Exception:
                    logger.exception("Error disconnecting camera %r during recovery", cam_key)
                try:
                    cam.connect()
                    logger.info("Camera %r reconnected", cam_key)
                except Exception:
                    logger.exception("Failed to reconnect camera %r", cam_key)
        finally:
            self._camera_reconnect_in_progress = False

    def _control_loop(self) -> None:
        # Single combined read+write loop at FPS, matching examples/lekiwi/teleoperate.py's
        # pattern (send_action + get_observation together every iteration, not split rates) --
        # split rates introduced extra latency/irregularity in send_action timing that felt
        # jerky, since get_observation (camera decode + motor bus read) could stall the loop
        # unpredictably relative to when it ran.
        period = 1.0 / FPS

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            with self._lock:
                playback_frame = None
                if self._playback_actions is not None:
                    if self._playback_index < len(self._playback_actions):
                        playback_frame = self._playback_actions[self._playback_index]
                        self._playback_index += 1
                    else:
                        self._finish_playback_locked()

            if playback_frame is not None:
                # Replaying a recorded episode: use its action verbatim instead of computing
                # one from whichever control mode is selected (that input is ignored for the
                # duration of playback, per RobotBridge's mode-gating design).
                action = dict(playback_frame)
                for joint in ARM_JOINTS:
                    key = f"{joint}.pos"
                    if key in action:
                        self._joint_targets[joint] = action[key]
            else:
                with self._lock:
                    base_stale = (loop_start - self._last_base_msg_time) > WATCHDOG_TIMEOUT_S
                    jogs_stale = (loop_start - self._last_jog_msg_time) > WATCHDOG_TIMEOUT_S
                    base_cmd = dict(self._desired_base)
                    jogs = dict(self._desired_jogs)
                    cart_jog = dict(self._desired_cartesian_jog) if self._desired_cartesian_jog else None
                    ramp_until = self._cartesian_ready_ramp_until
                    ramp_start = dict(self._cartesian_ready_ramp_start) if self._cartesian_ready_ramp_start else None
                    vr_stale = (loop_start - self._last_vr_msg_time) > WATCHDOG_TIMEOUT_S
                    vr_target = dict(self._desired_vr_target) if self._desired_vr_target else None

                if base_stale:
                    base_cmd = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
                if jogs_stale:
                    jogs = {}
                    cart_jog = None
                if vr_stale:
                    vr_target = None

                base_speed = BASE_SPEEDS[base_cmd["speed"]]
                action = {
                    "x.vel": base_cmd["x"] * base_speed["xy"],
                    "y.vel": base_cmd["y"] * base_speed["xy"],
                    "theta.vel": base_cmd["theta"] * base_speed["theta"],
                }

                for joint, jog in jogs.items():
                    if jog["dir"] == 0:
                        continue
                    delta = jog["dir"] * JOG_SPEEDS[jog["speed"]] * period
                    lo, hi = JOINT_RANGE[joint]
                    self._joint_targets[joint] = max(lo, min(hi, self._joint_targets[joint] + delta))

                if ramp_start is not None:
                    # Ramping into cartesian mode takes priority over any live jog input --
                    # see set_arm_submode()/CARTESIAN_READY_POSE. "Is a ramp in progress" is
                    # tracked by self._cartesian_ready_ramp_start being non-empty (cleared
                    # below once alpha reaches 1.0), not by comparing against ramp_until --
                    # that would skip the exact alpha=1.0 tick (the first tick whose
                    # loop_start lands past the deadline stops satisfying loop_start <
                    # ramp_until, so it would never run), leaving a small permanent residual
                    # instead of landing exactly on CARTESIAN_READY_POSE.
                    remaining = ramp_until - loop_start
                    alpha = 1.0 if remaining <= 0 else 1.0 - (remaining / CARTESIAN_READY_RAMP_S)
                    for joint in IK_ARM_JOINTS:
                        start = ramp_start.get(joint, self._joint_targets[joint])
                        end = CARTESIAN_READY_POSE[joint]
                        self._joint_targets[joint] = start + (end - start) * alpha
                    if alpha >= 1.0:
                        with self._lock:
                            self._cartesian_ready_ramp_start = {}
                elif cart_jog is not None and any(
                    cart_jog[k] for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
                ):
                    self._apply_cartesian_jog(cart_jog, period)
                elif vr_target is not None:
                    self._apply_vr_target(vr_target)

                for joint, target in self._joint_targets.items():
                    action[f"{joint}.pos"] = target

            try:
                self._robot.send_action(action)
            except Exception:
                logger.exception("send_action failed")

            try:
                obs = self._robot.get_observation()
                self._obs_failure_since = None
                new_joint_state = {}
                new_jpeg = {}
                for joint in ARM_JOINTS:
                    if f"{joint}.pos" not in obs:
                        continue
                    load = obs.get(f"{joint}.load", 0.0)
                    speed = obs.get(f"{joint}.speed", 0.0)
                    severity = _stall_severity(joint, load, speed)
                    new_joint_state[joint] = {
                        "pos": obs[f"{joint}.pos"],
                        "load": load,
                        "speed": speed,
                        "stalled": severity > 0.0,
                        "severity": severity,
                    }
                for cam_name in ("front", "wrist"):
                    frame = obs.get(cam_name)
                    if frame is not None:
                        # LeKiwi's cameras default to color_mode=RGB, but cv2.imencode
                        # expects BGR (OpenCV's native channel order) -- without this
                        # conversion, red and blue come out swapped in the JPEG.
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        ok, buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            new_jpeg[cam_name] = buf.tobytes()
                with self._lock:
                    self._latest_joint_state.update(new_joint_state)
                    self._latest_jpeg.update(new_jpeg)

                # Recording captures whatever's actually happening regardless of which
                # control mode is driving it -- skipped during playback (recording a replay
                # of itself isn't useful, and start_recording()/start_playback() already keep
                # the two mutually exclusive).
                if self._recorder is not None and playback_frame is None:
                    self._recorder.add_frame(obs, action)
            except Exception:
                logger.exception("get_observation failed")
                self._maybe_recover_cameras(loop_start)

            elapsed = time.monotonic() - loop_start
            time.sleep(max(period - elapsed, 0.0))
