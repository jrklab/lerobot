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
gamepad_teleoperate.py  —  Control a LeKiwi robot arm with a USB gamepad.

Architecture
============
Direct joint velocity control.  Each control input integrates a delta onto
the current joint position each frame.  No IK / URDF required.

  GamepadReader (background thread, inputs lib)
       │
       ▼
  compute per-joint delta from axes/buttons × speed
       │
       ▼
  new_pos = obs_pos + delta   (clipped to joint limits)
       │
       ▼ + base velocities from D-pad / bumpers
  LeKiwiClient.send_action()

Gamepad mapping — SHANWAN Android Gamepad (USB or Bluetooth)
=============================================================
  Left  stick X  (ABS_X)   →  shoulder_pan   (left / right)
  Left  stick Y  (ABS_Y)   →  shoulder_lift  (up / down)
  Right stick Y  (ABS_Z)   →  elbow_flex     (up / down)
  Right stick X  (ABS_RZ)  →  wrist_flex     (left / right)
  LT (ABS_BRAKE, hold)     →  gripper open
  RT (ABS_GAS,   hold)     →  gripper close
  Y  (BTN_NORTH, hold)     →  wrist_roll +
  X  (BTN_WEST,  hold)     →  wrist_roll −
  D-pad up/down  (ABS_HAT0Y)     →  base forward / back
  D-pad left/right (ABS_HAT0X)   →  base rotate
  LB (BTN_TL)              →  base strafe left
  RB (BTN_TR)              →  base strafe right
  START  (BTN_START)       →  cycle speed (slow → medium → fast)
  SELECT (BTN_SELECT)      →  emergency stop (zero all)
  B      (BTN_SOUTH)       →  go to neutral pose

Note on SHANWAN button labelling
=================================
  Linux code  │  Physical label on SHANWAN pad
  ────────────┼──────────────────────────────
  BTN_SOUTH   │  B
  BTN_EAST    │  A
  BTN_NORTH   │  Y
  BTN_WEST    │  X

Dependencies
============
  uv add evdev

Bluetooth setup (Linux)
=======================
  bluetoothctl
    > power on
    > scan on
    > pair <MAC>
    > trust <MAC>
    > connect <MAC>
  Once connected, /dev/input/eventX is created automatically.
  The script auto-detects USB vs Bluetooth axis encoding.

Usage
=====
  # On LeKiwi host (Raspberry Pi):
  python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi

  # On host PC:
  python examples/lekiwi/gamepad_teleoperate.py

  # Discover your gamepad axis/button codes first:
  python examples/lekiwi/probe_gamepad.py
"""

import select
import threading
import time

import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep

try:
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    _RERUN_AVAILABLE = True
except ImportError:
    _RERUN_AVAILABLE = False

    def init_rerun(**kwargs):  # type: ignore[misc]
        print("[INFO] rerun-sdk not installed — live visualisation disabled.")

    def log_rerun_data(**kwargs):  # type: ignore[misc]
        pass


# ─── Configuration ──────────────────────────────────────────────────────────

# Frame-rate of the control loop
FPS = 30

# Joint position limits in DEGREES (use_degrees=True).
# Clip all computed targets within these bounds.
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "arm_shoulder_pan":  (-180.0, 180.0),
    "arm_shoulder_lift": (-120.0,  80.0),
    "arm_elbow_flex":    (  -5.0, 180.0),
    "arm_wrist_flex":    (-100.0, 100.0),
    "arm_wrist_roll":    (-180.0, 180.0),
    "arm_gripper":       (   0.0, 100.0),
}

# Joint position limits in NORMALIZED range [-100, 100] (use_degrees=False).
# Derived by scaling degree limits by 100/180; gripper keeps [0, 100].
JOINT_LIMITS_NORM: dict[str, tuple[float, float]] = {
    "arm_shoulder_pan":  (-100.0, 100.0),
    "arm_shoulder_lift": ( -100.0,  100.0),
    "arm_elbow_flex":    (  -100.0, 100.0),
    "arm_wrist_flex":    ( -100.0,  100.0),
    "arm_wrist_roll":    (-100.0, 100.0),
    "arm_gripper":       (   0.0, 100.0),
}

# Neutral/home joint angles in DEGREES — sent when B (BTN_SOUTH) is pressed.
ARM_NEUTRAL_POS_DEG: dict[str, float] = {
    "arm_shoulder_pan":  0.0,
    "arm_shoulder_lift": -30.0,
    "arm_elbow_flex":    60.0,
    "arm_wrist_flex":    -30.0,
    "arm_wrist_roll":    0.0,
    "arm_gripper":       50.0,
}

# Neutral/home joint angles in NORMALIZED range [-100, 100].
ARM_NEUTRAL_POS_NORM: dict[str, float] = {
    "arm_shoulder_pan":  0.0,
    "arm_shoulder_lift": -98.0,
    "arm_elbow_flex":    99.0,
    "arm_wrist_flex":    75.0,
    "arm_wrist_roll":    52.0,
    "arm_gripper":       2.0,
}

# Three speed levels — joint/wrist_roll/gripper are units per frame at full input.
# At FPS=30: 1 unit/frame = 30 units/s.
# Degree mode:      1 deg/frame  = 30 deg/s
# Normalized mode:  0.56/frame   ≈ 17 norm-units/s  (same physical rate)
SPEED_LEVELS_DEG = [
    # slow
    {"joint": 0.5,  "wrist_roll": 0.8,  "gripper": 0.8,  "xy": 0.10, "theta": 30.0},
    # medium  ← default
    {"joint": 1.0,  "wrist_roll": 1.5,  "gripper": 1.5,  "xy": 0.20, "theta": 60.0},
    # fast
    {"joint": 2.0,  "wrist_roll": 3.0,  "gripper": 3.0,  "xy": 0.35, "theta": 90.0},
]

# Normalized equivalents: joint speeds scaled by 100/180 ≈ 0.556
SPEED_LEVELS_NORM = [
    # slow
    {"joint": 0.5, "wrist_roll": 0.8, "gripper": 0.8,  "xy": 0.10, "theta": 30.0},
    # medium  ← default
    {"joint": 1.0, "wrist_roll": 1.6, "gripper": 1.6,  "xy": 0.20, "theta": 60.0},
    # fast
    {"joint": 2.0, "wrist_roll": 3.2, "gripper": 3.2,  "xy": 0.40, "theta": 90.0},
]


# ─── Gamepad reader ──────────────────────────────────────────────────────────


class GamepadReader:
    """
    Reads a USB or Bluetooth gamepad using the `evdev` library.

    Detects the gamepad automatically by scanning /dev/input/event* for a
    device that has joystick axes (ABS_X/Y) and gamepad buttons (BTN_SOUTH).
    Works for both USB and Bluetooth without any manual configuration —
    axis ranges are read directly from the device kernel metadata.

    Normalises sticks to [-1, 1] and triggers to [0, 1].
    """

    DEADZONE = 0.08

    AXIS_CODES    = {"ABS_X", "ABS_Y", "ABS_Z", "ABS_RZ"}
    TRIGGER_CODES = {"ABS_BRAKE", "ABS_GAS"}
    HAT_CODES     = {"ABS_HAT0X", "ABS_HAT0Y"}

    ALL_TRACKED_AXES = AXIS_CODES | TRIGGER_CODES | HAT_CODES

    def __init__(self, gamepad_index: int = 0):
        import evdev as _evdev

        self._evdev = _evdev
        self._gamepad_index = gamepad_index
        self._device = None
        # Maps axis name → (min, max) read from device absinfo.
        self._abs_ranges: dict[str, tuple[int, int]] = {}

        self._axes: dict[str, float] = {
            "ABS_X": 0.0,
            "ABS_Y": 0.0,
            "ABS_Z": 0.0,
            "ABS_RZ": 0.0,
            "ABS_BRAKE": 0.0,
            "ABS_GAS": 0.0,
            "ABS_HAT0X": 0.0,
            "ABS_HAT0Y": 0.0,
        }
        self._buttons: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ── device discovery ─────────────────────────────────────────────────────

    def _find_gamepads(self) -> list:
        """Return all event devices that look like gamepads."""
        ecodes = self._evdev.ecodes
        found = []
        for path in self._evdev.list_devices():
            try:
                dev = self._evdev.InputDevice(path)
            except Exception:
                continue
            caps = dev.capabilities()
            abs_codes  = {c for c, _ in caps.get(ecodes.EV_ABS, [])}
            key_codes  = {c for c in caps.get(ecodes.EV_KEY, [])}
            # Require at least two sticks + one gamepad face button
            has_sticks  = ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes
            has_buttons = ecodes.BTN_SOUTH in key_codes or ecodes.BTN_GAMEPAD in key_codes
            if has_sticks and has_buttons:
                found.append(dev)
            else:
                dev.close()
        return found

    # ── start / stop ─────────────────────────────────────────────────────────

    def start(self) -> None:
        gamepads = self._find_gamepads()
        if not gamepads:
            raise RuntimeError(
                "No gamepad detected (USB or Bluetooth).\n"
                "  USB : plug in the controller and try again.\n"
                "  BT  : pair the controller first (bluetoothctl), then re-run.\n"
                "Run `python examples/lekiwi/probe_gamepad.py` to list all input devices."
            )
        if self._gamepad_index >= len(gamepads):
            raise RuntimeError(
                f"gamepad_index={self._gamepad_index} but only {len(gamepads)} gamepad(s) found."
            )
        self._device = gamepads[self._gamepad_index]
        # Close extra gamepads we won't use
        for dev in gamepads:
            if dev is not self._device:
                dev.close()

        print(f"Gamepad detected: {self._device.name}  ({self._device.path})")
        self._read_abs_ranges()

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._device:
            self._device.close()

    # ── axis normalisation ───────────────────────────────────────────────────

    def _read_abs_ranges(self) -> None:
        """Populate _abs_ranges from the device's kernel absinfo."""
        ecodes = self._evdev.ecodes
        caps = self._device.capabilities()
        for code, absinfo in caps.get(ecodes.EV_ABS, []):
            names = ecodes.ABS.get(code, [])
            name = names[0] if isinstance(names, list) else names
            if name in self.ALL_TRACKED_AXES:
                self._abs_ranges[name] = (absinfo.min, absinfo.max)

    def _normalise_abs(self, name: str, raw: int) -> float:
        """Normalise to [-1,1] for sticks, [0,1] for triggers, raw for hats."""
        if name in self.HAT_CODES:
            return float(raw)
        lo, hi = self._abs_ranges.get(name, (0, 255))
        if name in self.TRIGGER_CODES:
            span = hi - lo
            return (raw - lo) / span if span > 0 else 0.0
        # Stick: map [lo, hi] → [-1, 1]
        mid  = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        return (raw - mid) / half if half > 0 else 0.0

    def _apply_deadzone(self, value: float) -> float:
        return value if abs(value) > self.DEADZONE else 0.0

    # ── poll loop ────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        ecodes = self._evdev.ecodes
        while self._running:
            # Wait up to 0.1s for data — avoids busy-looping and BlockingIOError
            r, _, _ = select.select([self._device.fd], [], [], 0.1)
            if not r:
                continue
            try:
                events = self._device.read()
            except Exception:
                time.sleep(0.01)
                continue
            with self._lock:
                for event in events:
                    if event.type == ecodes.EV_ABS:
                        names = ecodes.ABS.get(event.code, [])
                        name = names[0] if isinstance(names, list) else names
                        if name in self._axes:
                            self._axes[name] = self._normalise_abs(name, event.value)
                    elif event.type == ecodes.EV_KEY:
                        # BTN codes live in both ecodes.BTN and ecodes.KEY.
                        # A single code may have multiple aliases (e.g. BTN_NORTH == BTN_Y,
                        # BTN_WEST == BTN_X).  Store the event under ALL aliases so that
                        # callers can use whichever name they prefer.
                        raw = ecodes.BTN.get(event.code) or ecodes.KEY.get(event.code, [])
                        aliases = list(raw) if isinstance(raw, (list, tuple)) else ([raw] if raw else [])
                        for name in aliases:
                            if name:
                                self._buttons[name] = bool(event.value)

    # ── public API ───────────────────────────────────────────────────────────

    def get_axes(self) -> dict[str, float]:
        with self._lock:
            return {
                k: (v if k in self.HAT_CODES else self._apply_deadzone(v))
                for k, v in self._axes.items()
            }

    def get_buttons(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._buttons)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def clamp_joint(joint_name: str, value: float, limits: dict[str, tuple[float, float]]) -> float:
    lo, hi = limits[joint_name]
    return float(np.clip(value, lo, hi))


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    robot_config = LeKiwiClientConfig(remote_ip="raspberrypi.local", id="my_lekiwi")
    robot = LeKiwiClient(robot_config)

    # Select unit system based on robot config
    use_degrees = robot_config.use_degrees
    JOINT_LIMITS  = JOINT_LIMITS_DEG  if use_degrees else JOINT_LIMITS_NORM
    ARM_NEUTRAL_POS = ARM_NEUTRAL_POS_DEG if use_degrees else ARM_NEUTRAL_POS_NORM
    SPEED_LEVELS  = SPEED_LEVELS_DEG  if use_degrees else SPEED_LEVELS_NORM
    unit_label = "deg" if use_degrees else "norm[-100,100]"
    print(f"[CONFIG] use_degrees={use_degrees} — joint units: {unit_label}")

    gamepad = GamepadReader()
    gamepad.start()

    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("LeKiwi robot is not connected!")

    if _RERUN_AVAILABLE:
        init_rerun(session_name="lekiwi_gamepad_teleop")

    # ── State ────────────────────────────────────────────────────────────────
    speed_index = 1  # start at medium
    prev_start_btn = False
    prev_b_btn = False

    # Shadow of current arm joint positions (degrees).
    # Seeded from neutral; overwritten from first robot observation.
    joint_pos: dict[str, float] = dict(ARM_NEUTRAL_POS)

    print("\n" + "=" * 60)
    print("LeKiwi Gamepad Teleop — controls (SHANWAN Android Gamepad):")
    print("  Left  stick X     → shoulder_pan  (ABS_X)")
    print("  Left  stick Y     → shoulder_lift (ABS_Y)")
    print("  Right stick Y     → elbow_flex    (ABS_Z)")
    print("  Right stick X     → wrist_flex    (ABS_RZ)")
    print("  LT (ABS_BRAKE)    → gripper open")
    print("  RT (ABS_GAS)      → gripper close")
    print("  Y (BTN_NORTH)     → wrist_roll +")
    print("  X (BTN_WEST)      → wrist_roll −")
    print("  D-pad up/down     → base forward / back")
    print("  D-pad left/right  → base rotate")
    print("  LB / RB           → base strafe left / right")
    print("  START             → cycle speed (slow → medium → fast)")
    print("  SELECT            → emergency stop")
    print("  B (BTN_SOUTH)     → go to neutral pose")
    print()
    print("  SHANWAN: BTN_SOUTH=B  BTN_EAST=A  BTN_NORTH=Y  BTN_WEST=X")
    print()
    print("  Connection: USB or Bluetooth — axis range auto-detected.")
    print("=" * 60 + "\n")

    first_obs = True

    while True:
        t0 = time.perf_counter()

        # ── Sense ────────────────────────────────────────────────────────────
        obs = robot.get_observation()
        axes = gamepad.get_axes()
        buttons = gamepad.get_buttons()

        # Sync shadow from real observation on first frame.
        if first_obs:
            for joint in JOINT_LIMITS:
                obs_key = f"{joint}.pos"
                if obs_key in obs:
                    joint_pos[joint] = float(obs[obs_key])
            first_obs = False

        # ── Emergency stop ───────────────────────────────────────────────────
        if buttons.get("BTN_SELECT", False):
            stop_action = {k: 0.0 for k in robot._state_order}
            robot.send_action(stop_action)
            print("[STOP] SELECT pressed — all velocities zeroed.")
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            continue

        # ── Speed cycling ────────────────────────────────────────────────────
        start_pressed = buttons.get("BTN_START", False)
        if start_pressed and not prev_start_btn:
            speed_index = (speed_index + 1) % len(SPEED_LEVELS)
            names = ["slow", "medium", "fast"]
            print(f"[SPEED] {names[speed_index]}")
        prev_start_btn = start_pressed

        speed = SPEED_LEVELS[speed_index]

        # ── B: go to neutral pose ────────────────────────────────────────────
        b_pressed = buttons.get("BTN_SOUTH", False)
        if b_pressed and not prev_b_btn:
            joint_pos = dict(ARM_NEUTRAL_POS)
            print("[NEUTRAL] Moving to neutral pose.")
        prev_b_btn = b_pressed

        # ── Joint deltas ─────────────────────────────────────────────────────
        #
        # Stick sign conventions (SHANWAN):
        #   ABS_Y negative = pushed forward → shoulder_lift raises → positive delta
        #   ABS_Z negative = pushed up      → elbow_flex opens    → positive delta
        # Adjust the signs below to taste.

        lx = axes["ABS_X"]      # left  stick X,  right = +1
        ly = axes["ABS_Y"]      # left  stick Y,  forward = −1
        ry = axes["ABS_Z"]      # right stick Y,  up = −1
        rx = axes["ABS_RZ"]     # right stick X,  right = +1
        lt = axes["ABS_BRAKE"]  # 0 → 1  (open gripper)
        rt = axes["ABS_GAS"]    # 0 → 1  (close gripper)

        js = speed["joint"]
        wr = speed["wrist_roll"]
        gs = speed["gripper"]

        wrist_roll_dir = (
            (1.0 if buttons.get("BTN_NORTH", False) else 0.0)
            - (1.0 if buttons.get("BTN_WEST",  False) else 0.0)
        )

        deltas: dict[str, float] = {
            "arm_shoulder_pan":  lx  * js,
            "arm_shoulder_lift": -ly * js,   # forward → lift up
            "arm_elbow_flex":    -ry * js,   # stick up → flex open
            "arm_wrist_flex":    -rx  * js,
            "arm_wrist_roll":    wrist_roll_dir * wr,
            "arm_gripper":       (lt - rt) * gs,
        }

        for joint, delta in deltas.items():
            joint_pos[joint] = clamp_joint(joint, joint_pos[joint] + delta, JOINT_LIMITS)

        arm_action: dict[str, float] = {
            f"{joint}.pos": pos for joint, pos in joint_pos.items()
        }

        # ── Base action ──────────────────────────────────────────────────────
        xy_speed = speed["xy"]
        th_speed = speed["theta"]

        base_action: dict[str, float] = {
            "x.vel":     -axes["ABS_HAT0Y"] * xy_speed,
            "y.vel":     0.0,
            "theta.vel": -axes["ABS_HAT0X"] * th_speed,
        }
        if buttons.get("BTN_TL", False):
            base_action["y.vel"] += xy_speed   # LB → strafe left
        if buttons.get("BTN_TR", False):
            base_action["y.vel"] -= xy_speed   # RB → strafe right

        # ── Send ─────────────────────────────────────────────────────────────
        action = {**arm_action, **base_action}
        robot.send_action(action)

        # ── Visualise ────────────────────────────────────────────────────────
        try:
            log_rerun_data(observation=obs, action=action)
        except Exception:
            pass

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
