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
commands to take effect -- it does not switch modes itself.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FPS = 30
SPEED_NAMES = ["slow", "medium", "fast"]

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


async def _drain_incoming(ws) -> None:
    """We don't need the server's `state` broadcasts, but must keep reading so the
    connection doesn't back up."""
    with contextlib.suppress(websockets.ConnectionClosed):
        async for _ in ws:
            pass


async def _control_loop(ws, leader: SO101Leader, keyboard: KeyboardTeleop) -> None:
    speed_cycle = _SpeedCycle()
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
    logger.info("Leader arm + keyboard connected. Select 'Leader + Keyboard' in the web app to drive.")

    try:
        while True:
            try:
                async with websockets.connect(args.server) as ws:
                    logger.info("Connected to %s", args.server)
                    drain_task = asyncio.create_task(_drain_incoming(ws))
                    try:
                        await _control_loop(ws, leader, keyboard)
                    finally:
                        drain_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await drain_task
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
