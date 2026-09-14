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
Bluetooth/USB gamepad input for LeKiwi, feeding the same RobotBridge used by the web app
(see lekiwi_web_server.md's "Input Multiplexer" -- gamepad and web UI both drive the same
underlying control loop rather than competing for the robot connection).

Axis/button reading is adapted from the original examples/lekiwi/gamepad_teleoperate.py
(SHANWAN Android Gamepad mapping), auto-detecting USB or Bluetooth via evdev the same way.
The specific controller this was verified against (via examples/lekiwi/probe_gamepad.py,
one button/axis at a time) reports different codes than that reference script assumed:

  Left stick:  ABS_X / ABS_Y        Right stick: ABS_RX / ABS_RY
  D-pad:       ABS_HAT0X / ABS_HAT0Y (not currently used)
  Y -> BTN_C   X -> BTN_NORTH   A -> BTN_B   B -> BTN_A
  LB -> BTN_WEST   RB -> BTN_Z   (confirmed via isolated one-at-a-time testing; an earlier,
    less isolated test had briefly suggested these were a "digital echo" of LT/RT -- they
    are not, a clean re-test showed LT/RT only ever produce ABS_Z/ABS_RZ, nothing else)
  LT -> ABS_Z (proportional)      RT -> ABS_RZ (proportional)
  A separate "-"/"+" button pair (not currently used) sends BTN_TL/BTN_TR.
  Select/Start -> confirmed dead buttons on this unit, no evdev event at all -- do not use.
  Home -> KEY_MENU (not currently used)

Mode toggle ("A" button / BTN_B): switches which mode the two analog sticks drive.
  - Base mode (default at startup): left stick = translate (x, y), right stick X = rotate.
  - Arm mode: left stick = shoulder_pan/lift, right stick = elbow_flex/wrist_flex (unchanged
    from the original script). Only one mode is ever "live" at a time, matching the base/arm
    tab split already used in the web UI -- switching modes releases whichever was active.

Other buttons (active in both modes): LB cycles speed slow/medium/fast, RB is emergency
stop (zero all motion; does NOT move the arm to any position), B resets the arm to its
neutral pose. Select/Start would have been the more conventional choice for speed/e-stop,
but they're dead on this controller.
Arm-mode-only: LT/RT (gripper), Y/X (wrist_roll).
"""

import logging
import os
import select
import threading
import time

from robot_bridge import ARM_JOINTS, RobotBridge

logger = logging.getLogger(__name__)

DEADZONE = 0.08
POLL_HZ = 30
RECONNECT_INTERVAL_S = 2.0
# How often to check that the device's /dev/input node still exists (catches a Bluetooth
# drop that doesn't cleanly raise an OSError on the next read/select call). Idle gamepad
# input (sticks centered, no buttons held) produces zero events -- that's normal, not a
# disconnect signal, so this must be an independent liveness check, not an event timeout.
LIVENESS_CHECK_INTERVAL_S = 1.0

SPEED_NAMES = ["slow", "medium", "fast"]

# Verified against the actual paired controller via examples/lekiwi/probe_gamepad.py --
# this gamepad's HID descriptor differs from the SHANWAN reference mapping in
# gamepad_teleoperate.py's docstring (right stick axes and face-button codes are
# different, and LT/RT are ABS_Z/ABS_RZ here rather than ABS_BRAKE/ABS_GAS).
#   Left stick:  ABS_X / ABS_Y
#   Right stick: ABS_RX / ABS_RY
#   Triggers:    ABS_Z (one trigger), ABS_RZ (the other) -- proportional, 0..1023
#   Y -> BTN_C   X -> BTN_NORTH   A -> BTN_B   B -> BTN_A
AXIS_CODES = {"ABS_X", "ABS_Y", "ABS_RX", "ABS_RY"}
TRIGGER_CODES = {"ABS_Z", "ABS_RZ"}
ALL_TRACKED_AXES = AXIS_CODES | TRIGGER_CODES


class GamepadInput:
    def __init__(self, bridge: RobotBridge):
        self._bridge = bridge
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._evdev = None
        self._device = None
        self._abs_ranges: dict[str, tuple[int, int]] = {}
        self._axes: dict[str, float] = dict.fromkeys(ALL_TRACKED_AXES, 0.0)
        self._buttons: dict[str, bool] = {}

        self._mode = "base"  # "base" | "arm"
        self._speed_index = 1  # start at medium
        self._prev_buttons: dict[str, bool] = {}
        self._active_base = False
        self._active_jogs: dict[str, bool] = dict.fromkeys(ARM_JOINTS, False)

    def start(self) -> None:
        try:
            import evdev

            self._evdev = evdev
        except ImportError:
            logger.warning("evdev not available -- gamepad input disabled.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gamepad_input", daemon=True)
        self._thread.start()

    def is_connected(self) -> bool:
        """Thread-safe enough for a status readout: a plain reference read/write of
        self._device, no torn-write risk under the GIL for this use case."""
        return self._device is not None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._device is not None:
            self._device.close()

    # --- device discovery ---

    def _find_gamepad(self):
        ecodes = self._evdev.ecodes
        for path in self._evdev.list_devices():
            try:
                dev = self._evdev.InputDevice(path)
            except Exception:
                continue
            caps = dev.capabilities()
            abs_codes = {c for c, _ in caps.get(ecodes.EV_ABS, [])}
            key_codes = set(caps.get(ecodes.EV_KEY, []))
            has_sticks = ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes
            has_buttons = ecodes.BTN_SOUTH in key_codes or ecodes.BTN_GAMEPAD in key_codes
            if has_sticks and has_buttons:
                return dev
            dev.close()
        return None

    def _read_abs_ranges(self) -> None:
        ecodes = self._evdev.ecodes
        caps = self._device.capabilities()
        for code, absinfo in caps.get(ecodes.EV_ABS, []):
            names = ecodes.ABS.get(code, [])
            name = names[0] if isinstance(names, list) else names
            if name in ALL_TRACKED_AXES:
                self._abs_ranges[name] = (absinfo.min, absinfo.max)

    def _normalize(self, name: str, raw: int) -> float:
        lo, hi = self._abs_ranges.get(name, (0, 255))
        if name in TRIGGER_CODES:
            span = hi - lo
            return (raw - lo) / span if span > 0 else 0.0
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        return (raw - mid) / half if half > 0 else 0.0

    def _deadzone(self, value: float) -> float:
        return value if abs(value) > DEADZONE else 0.0

    # --- main loop ---

    def _run(self) -> None:
        period = 1.0 / POLL_HZ
        last_reconnect_attempt = 0.0
        last_liveness_check = 0.0

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            if self._device is None:
                if loop_start - last_reconnect_attempt >= RECONNECT_INTERVAL_S:
                    last_reconnect_attempt = loop_start
                    dev = self._find_gamepad()
                    if dev is not None:
                        self._device = dev
                        self._read_abs_ranges()
                        logger.info("Gamepad connected: %s (%s)", dev.name, dev.path)
                time.sleep(period)
                continue

            # A gamepad genuinely emits zero events while idle (sticks centered, no
            # buttons held) -- that's normal, not a disconnect. The only reliable
            # disconnect signals are a read/select error, or the kernel having removed
            # the input node (which it does when a Bluetooth HID device drops).
            if loop_start - last_liveness_check > LIVENESS_CHECK_INTERVAL_S:
                last_liveness_check = loop_start
                if not os.path.exists(self._device.path):
                    logger.warning("Gamepad input node disappeared; assuming disconnected.")
                    self._handle_disconnect()
                    continue

            try:
                r, _, _ = select.select([self._device.fd], [], [], period)
                if r:
                    for event in self._device.read():
                        self._handle_event(event)
            except Exception as e:
                logger.warning("Gamepad disconnected (%s); will keep retrying.", e)
                self._handle_disconnect()
                continue

            self._dispatch()

            elapsed = time.monotonic() - loop_start
            time.sleep(max(period - elapsed, 0.0))

    def _handle_disconnect(self) -> None:
        self._release_all()
        try:
            self._device.close()
        except Exception:
            pass
        self._device = None

    def _handle_event(self, event) -> None:
        ecodes = self._evdev.ecodes
        if event.type == ecodes.EV_ABS:
            names = ecodes.ABS.get(event.code, [])
            name = names[0] if isinstance(names, list) else names
            if name in self._axes:
                self._axes[name] = self._normalize(name, event.value)
        elif event.type == ecodes.EV_KEY:
            raw = ecodes.BTN.get(event.code) or ecodes.KEY.get(event.code, [])
            aliases = list(raw) if isinstance(raw, (list, tuple)) else ([raw] if raw else [])
            for name in aliases:
                if name:
                    self._buttons[name] = bool(event.value)

    def _pressed_edge(self, name: str) -> bool:
        """True only on the press transition (not while held), so toggles fire once."""
        now = self._buttons.get(name, False)
        was = self._prev_buttons.get(name, False)
        self._prev_buttons[name] = now
        return now and not was

    def _release_all(self) -> None:
        """Called on disconnect: send one final stop so a dropped connection can't leave
        the robot driving/jogging forever."""
        self._bridge.emergency_stop()
        self._active_base = False
        self._active_jogs = dict.fromkeys(ARM_JOINTS, False)

    def _dispatch(self) -> None:
        # A -> BTN_B (mode toggle)
        if self._pressed_edge("BTN_B"):
            self._mode = "arm" if self._mode == "base" else "base"
            logger.info("Gamepad mode: %s", self._mode.upper())
            # Release whatever the other mode was driving.
            self._bridge.update_base(0.0, 0.0, 0.0, SPEED_NAMES[self._speed_index])
            for joint in ARM_JOINTS:
                self._bridge.update_jog(joint, 0.0, SPEED_NAMES[self._speed_index])
            self._active_base = False
            self._active_jogs = dict.fromkeys(ARM_JOINTS, False)

        # This controller's Select/Start buttons are dummies (confirmed: no evdev event at
        # all), so speed-cycle/e-stop live on LB/RB instead. Verified via isolated testing
        # that BTN_WEST/BTN_Z are LB/RB's own codes, not tied to the LT/RT triggers (which
        # only ever produce ABS_Z/ABS_RZ) -- safe to use without risk of an accidental
        # e-stop firing whenever the gripper trigger is pulled.
        # LB -> BTN_WEST (cycle speed)
        if self._pressed_edge("BTN_WEST"):
            self._speed_index = (self._speed_index + 1) % len(SPEED_NAMES)
            logger.info("Gamepad speed: %s", SPEED_NAMES[self._speed_index])

        # B -> BTN_A (reset arm to neutral)
        if self._pressed_edge("BTN_A"):
            logger.info("Gamepad: reset arm to neutral.")
            self._bridge.reset_arm()

        # RB -> BTN_Z (emergency stop, held)
        if self._buttons.get("BTN_Z", False):
            self._bridge.emergency_stop()
            self._active_base = False
            self._active_jogs = dict.fromkeys(ARM_JOINTS, False)
            return

        speed = SPEED_NAMES[self._speed_index]
        lx, ly = self._deadzone(self._axes["ABS_X"]), self._deadzone(self._axes["ABS_Y"])
        rx, ry = self._deadzone(self._axes["ABS_RX"]), self._deadzone(self._axes["ABS_RY"])

        if self._mode == "base":
            # y.vel is negative for rightward strafe (same convention as the web app's
            # joystick: `y: -translate.x`) -- lx is positive when pushed right, so negate.
            self._dispatch_base(x=-ly, y=-lx, theta=-rx, speed=speed)
        else:
            self._dispatch_arm(lx=lx, ly=ly, rx=rx, ry=ry, speed=speed)

    def _dispatch_base(self, x: float, y: float, theta: float, speed: str) -> None:
        is_active = bool(x or y or theta)
        if is_active or self._active_base:
            self._bridge.update_base(x, y, theta, speed)
        self._active_base = is_active

    def _dispatch_arm(self, lx: float, ly: float, rx: float, ry: float, speed: str) -> None:
        lt = self._axes.get("ABS_Z", 0.0)
        rt = self._axes.get("ABS_RZ", 0.0)
        # Y -> BTN_C (wrist_roll+), X -> BTN_NORTH (wrist_roll-)
        wrist_roll_dir = (1.0 if self._buttons.get("BTN_C", False) else 0.0) - (
            1.0 if self._buttons.get("BTN_NORTH", False) else 0.0
        )

        # Sign conventions match the original gamepad_teleoperate.py mapping.
        dirs = {
            "arm_shoulder_pan": lx,
            "arm_shoulder_lift": -ly,
            "arm_elbow_flex": -ry,
            "arm_wrist_flex": -rx,
            "arm_wrist_roll": wrist_roll_dir,
            "arm_gripper": lt - rt,
        }
        for joint, direction in dirs.items():
            is_active = direction != 0.0
            if is_active or self._active_jogs[joint]:
                self._bridge.update_jog(joint, direction, speed)
            self._active_jogs[joint] = is_active
