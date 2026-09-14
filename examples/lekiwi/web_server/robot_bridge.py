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
import threading
import time
from typing import Protocol

import cv2
import numpy as np

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

# Only one control source drives the robot at a time. "gamepad" and "web" feed
# update_base()/update_jog() (jog deltas); "leader_keyboard" feeds update_base() (from
# keyboard key state, same as "web") and set_arm_absolute() (direct position mirror from a
# leader arm on the host PC -- see leader_keyboard_client.py).
CONTROL_MODES = ("gamepad", "web", "leader_keyboard")

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

    def __init__(self, robot: RobotLike):
        self._robot = robot
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._control_mode = "gamepad"

        self._desired_base = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
        self._desired_jogs: dict[str, dict] = {}  # joint -> {"dir": -1|0|1, "speed": str}
        # Separate watchdog timestamps: arm jog traffic must never mask a stale base
        # command (or vice versa) — each input source goes stale independently.
        self._last_base_msg_time = 0.0
        self._last_jog_msg_time = 0.0

        self._joint_targets: dict[str, float] = dict.fromkeys(ARM_JOINTS, 0.0)
        self._latest_joint_state: dict[str, dict] = {
            joint: {"pos": 0.0, "load": 0.0, "stalled": False, "severity": 0.0} for joint in ARM_JOINTS
        }
        self._latest_jpeg: dict[str, bytes] = {}

        # Leader-arm mirroring ramps in over LEADER_CATCHUP_S on entry to "leader_keyboard"
        # instead of snapping straight to the leader's (possibly very different) live pose --
        # see set_control_mode()/set_arm_absolute(). Deliberately NOT a blanket per-tick clamp
        # (like max_relative_target) since that would also cap legitimate fast hand motion
        # during normal teleoperation, not just the one-time mode-entry jump.
        self._leader_catchup_start: dict[str, float] = {}
        self._leader_catchup_until = 0.0

    def start(self) -> None:
        self._robot.connect()
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
            self._last_jog_msg_time = time.monotonic()

    def get_jpeg_frame(self, cam_name: str) -> bytes | None:
        with self._lock:
            return self._latest_jpeg.get(cam_name)

    def get_joint_state(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._latest_joint_state)

    # --- background thread ---

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
                base_stale = (loop_start - self._last_base_msg_time) > WATCHDOG_TIMEOUT_S
                jogs_stale = (loop_start - self._last_jog_msg_time) > WATCHDOG_TIMEOUT_S
                base_cmd = dict(self._desired_base)
                jogs = dict(self._desired_jogs)

            if base_stale:
                base_cmd = {"x": 0.0, "y": 0.0, "theta": 0.0, "speed": "medium"}
            if jogs_stale:
                jogs = {}

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

            for joint, target in self._joint_targets.items():
                action[f"{joint}.pos"] = target

            try:
                self._robot.send_action(action)
            except Exception:
                logger.exception("send_action failed")

            try:
                obs = self._robot.get_observation()
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
            except Exception:
                logger.exception("get_observation failed")

            elapsed = time.monotonic() - loop_start
            time.sleep(max(period - elapsed, 0.0))
