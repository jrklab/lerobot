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
Companion client for the LeKiwi web app's "leader + keyboard" control mode.

Unlike the gamepad (which is read in-process on the Pi, over Bluetooth), the leader arm and
keyboard belong to a professional operator sitting at their own workstation -- so this runs
on THAT machine (the host PC, not the Pi) and talks to the already-running web server over
its existing WebSocket endpoint, exactly like a browser tab would. It sends two message
types the server only honors while the web app's control-mode selector is set to
"leader_keyboard": `arm_pos` (absolute leader positions, mirrored directly onto the arm) and
`base` with `source: "leader_keyboard"` (WASD/ZX-derived base velocity, same schema as the
web app's on-page joystick).

You still need to select "Leader + Keyboard" in the web app's mode selector for this script's
commands to take effect -- it does not switch modes itself, but it does print a line whenever
the server confirms that mode is (or stops being) active, so it's obvious whether anything
you do here currently has an effect.

Torque feedback (leader arm resists when the follower stalls, same as examples/lekiwi/
teleoperate.py) is available -- press 'b' to toggle it on/off, starts disabled.

Usage:
  uv run python examples/lekiwi/web_server/leader_keyboard_client.py \
      --server ws://raspberrypi.local:8000/ws/control \
      --leader-port /dev/ttyUSB0 --leader-id leader_arm_1
"""

import argparse
import asyncio
import contextlib
import json
import logging
import time

import websockets

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.teleoperators.torque_feedback import TorqueFeedbackConfig, map_load_to_torque_limit


# force=True: lerobot.utils.import_utils's `_pynput_available = is_package_available("pynput")`
# (evaluated at import time, before this line runs, since it's pulled in transitively via
# KeyboardTeleop) calls logging.debug() internally, which is the *first* bare logging.xxx()
# call in the process -- Python auto-configures the root logger to level WARNING right there
# if nothing's configured yet, which makes a later plain basicConfig() a silent no-op (per
# the logging docs). Without force=True, every logger.info() below (including the startup
# and mode-change lines) gets swallowed with no error, even though the script is running fine.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
)
logger = logging.getLogger(__name__)

FPS = 30
SPEED_NAMES = ["slow", "medium", "fast"]
ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Matches LeKiwiConfig.teleop_keys' defaults (examples/lekiwi/teleoperate.py's reference
# keyboard-base mapping), so this feels the same as the existing ZMQ-based teleop script.
TELEOP_KEYS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "rotate_left": "z",
    "rotate_right": "x",
    "speed_up": "r",
    "speed_down": "f",
}

# Testing max resistance: per_motor_scales raised to 1.0 (ceiling) across the board, up from
# examples/lekiwi/teleoperate.py's recommended starting point (0.5 / 0.3 for gripper).
TORQUE_FEEDBACK_CONFIG = TorqueFeedbackConfig(
    enabled=False,  # starts disabled; toggle with 'b'
    global_scale_factor=1.0,
    per_motor_scales={
        "shoulder_pan": 1.0,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 1.0,
        "wrist_roll": 1.0,
        "gripper": 1.0,
    },
    per_motor_thresholds={
        "shoulder_pan": 0.3,
        "shoulder_lift": 0.5,
        "elbow_flex": 0.5,
        "wrist_flex": 0.5,
        "wrist_roll": 0.3,
        "gripper": 0.2,
    },
    speed_threshold=50,
)


class _SpeedCycle:
    """Edge-detects the speed_up/speed_down keys so holding one doesn't cycle every tick."""

    def __init__(self):
        self.index = SPEED_NAMES.index("medium")
        self._prev_up = False
        self._prev_down = False

    def update(self, pressed_keys: dict) -> str:
        up = TELEOP_KEYS["speed_up"] in pressed_keys
        down = TELEOP_KEYS["speed_down"] in pressed_keys
        if up and not self._prev_up:
            self.index = min(self.index + 1, len(SPEED_NAMES) - 1)
            logger.info("Base speed: %s", SPEED_NAMES[self.index])
        if down and not self._prev_down:
            self.index = max(self.index - 1, 0)
            logger.info("Base speed: %s", SPEED_NAMES[self.index])
        self._prev_up, self._prev_down = up, down
        return SPEED_NAMES[self.index]


def _keyboard_to_base(pressed_keys: dict) -> tuple[float, float, float]:
    """Normalized -1..1 per axis, same sign convention as the web app's joystick/gamepad:
    y positive = left strafe, theta positive = turn left (CCW)."""
    x = y = theta = 0.0
    if TELEOP_KEYS["forward"] in pressed_keys:
        x += 1.0
    if TELEOP_KEYS["backward"] in pressed_keys:
        x -= 1.0
    if TELEOP_KEYS["left"] in pressed_keys:
        y += 1.0
    if TELEOP_KEYS["right"] in pressed_keys:
        y -= 1.0
    if TELEOP_KEYS["rotate_left"] in pressed_keys:
        theta += 1.0
    if TELEOP_KEYS["rotate_right"] in pressed_keys:
        theta -= 1.0
    return x, y, theta


class _ServerState:
    """Latest info from the server's periodic `state` broadcasts (see server.py's
    `_status_pusher`), kept up to date by `_receive_loop` and read by the control loop."""

    def __init__(self):
        self.joints: dict[str, dict] = {}
        self.control_mode: str | None = None


async def _receive_loop(ws, state: _ServerState) -> None:
    """Reads every incoming message so the connection doesn't back up, and prints a line
    whenever the server-confirmed control mode changes (so it's obvious whether this
    script's commands are currently taking effect)."""
    with contextlib.suppress(websockets.ConnectionClosed):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "state":
                continue
            state.joints = msg.get("joints", {})
            mode = msg.get("control_mode")
            if mode is not None and mode != state.control_mode:
                state.control_mode = mode
                if mode == "leader_keyboard":
                    logger.info("Control mode is now ACTIVE (leader_keyboard) -- driving the robot.")
                else:
                    logger.info("Control mode is now '%s' -- this script's commands are ignored.", mode)


class _StallHold:
    """Experimental: on top of the usual Torque_Limit capping, snaps each stalled joint's
    leader `Goal_Position` to the follower's position at the moment of the stall (not the
    leader's own position -- the follower may lag behind the leader, so its actual stuck
    point is the more meaningful target). Snapshotted once per stall onset (the 0 -> active
    transition), not re-snapshotted every tick while still stalled, so the hold point
    doesn't chase any further creep; released (no special action needed -- send_feedback
    already disables torque once torque_limit returns to 0) and re-armed fresh next stall.
    """

    def __init__(self):
        self._holding: set[str] = set()

    def update(self, leader: SO101Leader, torque_limits: dict[str, float], state: _ServerState) -> None:
        for motor, limit in torque_limits.items():
            if limit > 0:
                if motor not in self._holding:
                    info = state.joints.get(f"arm_{motor}")
                    if info is not None:
                        q = info["pos"]
                        leader.bus.write("Goal_Position", motor, q)
                        logger.info("Stall hold engaged: %s -> follower's stalled position %.1f", motor, q)
                    self._holding.add(motor)
            elif motor in self._holding:
                logger.info("Stall hold released: %s", motor)
                self._holding.discard(motor)

    def reset(self) -> None:
        self._holding.clear()


def _apply_torque_feedback(leader: SO101Leader, state: _ServerState, stall_hold: _StallHold) -> None:
    load_dict = {}
    speed_dict = {}
    for joint, info in state.joints.items():
        motor = joint.removeprefix("arm_")
        if motor in ARM_MOTORS and "load" in info and "speed" in info:
            load_dict[motor] = info["load"]
            speed_dict[motor] = info["speed"]
    if not load_dict:
        return
    torque_limits = map_load_to_torque_limit(
        load_dict, TORQUE_FEEDBACK_CONFIG, list(load_dict.keys()), speed_dict=speed_dict
    )
    stall_hold.update(leader, torque_limits, state)
    leader.send_feedback({f"{motor}.torque": val for motor, val in torque_limits.items()})


async def _control_loop(ws, leader: SO101Leader, keyboard: KeyboardTeleop, state: _ServerState) -> None:
    speed_cycle = _SpeedCycle()
    stall_hold = _StallHold()
    prev_b_pressed = False
    period = 1.0 / FPS
    while True:
        loop_start = time.monotonic()

        leader_action = leader.get_action()  # {"shoulder_pan.pos": ..., ...}
        positions = {f"arm_{k.removesuffix('.pos')}": v for k, v in leader_action.items()}
        await ws.send(json.dumps({"type": "arm_pos", "positions": positions}))

        pressed_keys = keyboard.get_action()
        speed = speed_cycle.update(pressed_keys)
        x, y, theta = _keyboard_to_base(pressed_keys)
        await ws.send(
            json.dumps(
                {"type": "base", "x": x, "y": y, "theta": theta, "speed": speed, "source": "leader_keyboard"}
            )
        )

        b_pressed = "b" in pressed_keys
        if b_pressed and not prev_b_pressed:
            TORQUE_FEEDBACK_CONFIG.enabled = not TORQUE_FEEDBACK_CONFIG.enabled
            if TORQUE_FEEDBACK_CONFIG.enabled:
                logger.info("Torque feedback: ENABLED")
            else:
                logger.info("Torque feedback: DISABLED")
                leader.send_feedback({f"{m}.torque": 0 for m in ARM_MOTORS})
                stall_hold.reset()
        prev_b_pressed = b_pressed

        if TORQUE_FEEDBACK_CONFIG.enabled:
            _apply_torque_feedback(leader, state, stall_hold)

        elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(period - elapsed, 0.0))


async def main_async(args: argparse.Namespace) -> None:
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="leader_keyboard_client"))
    leader.connect()
    keyboard.connect()
    if not keyboard.is_connected:
        raise RuntimeError(
            "Keyboard listener failed to start (needs a graphical session -- pynput requires "
            "DISPLAY on Linux). Run this script on a desktop, not a headless machine."
        )
    logger.info("Leader arm + keyboard connected.")
    logger.info("Select 'Leader arm + keyboard' in the web app's Control selector to drive the robot.")
    logger.info("Press 'b' to toggle torque feedback ON/OFF (starts disabled).")

    state = _ServerState()
    try:
        while True:
            try:
                async with websockets.connect(args.server) as ws:
                    logger.info("Connected to %s", args.server)
                    receive_task = asyncio.create_task(_receive_loop(ws, state))
                    try:
                        await _control_loop(ws, leader, keyboard, state)
                    finally:
                        receive_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await receive_task
            except (websockets.ConnectionClosed, OSError) as e:
                logger.warning("Connection to %s lost (%s); retrying in 1s...", args.server, e)
                await asyncio.sleep(1.0)
    finally:
        leader.disconnect()
        keyboard.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--server", required=True, help="Web app WS URL, e.g. ws://raspberrypi.local:8000/ws/control"
    )
    parser.add_argument("--leader-port", required=True, help="Serial port for the leader arm (on this PC).")
    parser.add_argument("--leader-id", required=True, help="Leader arm calibration id.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
