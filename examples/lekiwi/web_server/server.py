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
LeKiwi web control server (Phase 2 of examples/lekiwi/lekiwi_web_server.md).

Serves a vanilla HTML/JS/WebSocket control UI (base joystick + 6 explicit per-joint arm
jog controls) and drives the robot through a background thread (see robot_bridge.py).

Run against real hardware (on the Pi):
    uv run python examples/lekiwi/web_server/server.py --robot-id kiwi_sn_0

Run mocked, for frontend/UI development without hardware (host PC or Pi):
    uv run python examples/lekiwi/web_server/server.py --mock
"""

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# Allow running this script directly (`python server.py` or `python examples/.../server.py`)
# regardless of the current working directory, since robot_bridge.py is a plain sibling
# module rather than part of the installed lerobot package.
sys.path.insert(0, str(Path(__file__).parent))

from gamepad_input import GamepadInput  # noqa: E402
from robot_bridge import ARM_JOINTS, BASE_SPEEDS, JOG_SPEEDS, MockLeKiwi, RobotBridge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="LeKiwi Web Control")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

bridge: RobotBridge | None = None
gamepad: GamepadInput | None = None


@app.get("/")
async def index():
    return StreamingResponse(open(STATIC_DIR / "index.html", "rb"), media_type="text/html")


@app.get("/video/{cam_name}")
async def video_feed(cam_name: str):
    if cam_name not in ("front", "wrist"):
        return StreamingResponse(iter([b""]), status_code=404)

    async def mjpeg_stream():
        boundary = b"--frame"
        while True:
            frame = bridge.get_jpeg_frame(cam_name) if bridge else None
            if frame is not None:
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(1 / 15)

    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/control")
async def ws_control(websocket: WebSocket):
    await websocket.accept()
    client_tag = str(websocket.client)
    logger.info("Client connected: %s", client_tag)
    try:
        status_task = asyncio.create_task(_status_pusher(websocket))
        try:
            # Diagnostics: the client sends at a fixed ~33ms cadence (30Hz); if messages
            # are actually arriving less regularly than that for a given client, that's
            # direct evidence of network/browser-side delay rather than a robot-side issue.
            last_recv = time.monotonic()
            gap_count = 0
            gap_total_ms = 0.0
            gap_max_ms = 0.0
            while True:
                raw = await websocket.receive_text()
                now = time.monotonic()
                gap_ms = (now - last_recv) * 1000
                last_recv = now
                if gap_ms > 100:  # more than ~3x the expected 33ms interval
                    gap_count += 1
                    gap_total_ms += gap_ms
                    gap_max_ms = max(gap_max_ms, gap_ms)
                    if gap_count % 10 == 1:
                        logger.warning(
                            "%s: WS message gap %.0fms (>%d such gaps so far, worst %.0fms, avg %.0fms)",
                            client_tag,
                            gap_ms,
                            gap_count,
                            gap_max_ms,
                            gap_total_ms / gap_count,
                        )
                _handle_message(raw)
        finally:
            status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await status_task
    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", websocket.client)


async def _status_pusher(websocket: WebSocket) -> None:
    """Periodically pushes current arm joint state back to the client (for a live readout)."""
    while True:
        if bridge is not None:
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "state",
                            "joints": bridge.get_joint_state(),
                            "gamepad_connected": gamepad.is_connected() if gamepad is not None else False,
                        }
                    )
                )
            except Exception:
                return
        await asyncio.sleep(0.2)


def _handle_message(raw: str) -> None:
    if bridge is None:
        return
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Dropping malformed WS message: %s", raw)
        return

    msg_type = msg.get("type")
    if msg_type == "base":
        speed = msg.get("speed", "medium")
        if speed not in BASE_SPEEDS:
            speed = "medium"
        bridge.update_base(
            x=float(msg.get("x", 0.0)),
            y=float(msg.get("y", 0.0)),
            theta=float(msg.get("theta", 0.0)),
            speed=speed,
        )
    elif msg_type == "jog":
        joint = msg.get("joint")
        if joint not in ARM_JOINTS:
            logger.warning("Dropping jog message for unknown joint: %s", joint)
            return
        speed = msg.get("speed", "medium")
        if speed not in JOG_SPEEDS:
            speed = "medium"
        direction = int(msg.get("dir", 0))
        bridge.update_jog(joint, direction, speed)
    elif msg_type == "reset_arm":
        logger.info("Resetting arm to neutral pose.")
        bridge.reset_arm()
    else:
        logger.warning("Dropping WS message with unknown type: %s", msg_type)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use a mocked robot (no hardware required).")
    parser.add_argument("--robot-id", default="kiwi_sn_0", help="LeKiwi robot id (calibration file name).")
    parser.add_argument("--port-name", default="/dev/ttyUSB0", help="Serial port for the motor bus.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the web server.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the web server.")
    parser.add_argument(
        "--no-gamepad",
        action="store_true",
        help="Disable Bluetooth/USB gamepad input (auto-detected and connected by default).",
    )
    args = parser.parse_args()

    global bridge, gamepad
    if args.mock:
        logger.info("Starting with MockLeKiwi (no hardware).")
        robot = MockLeKiwi()
    else:
        from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
        from lerobot.robots.lekiwi.lekiwi import LeKiwi

        robot = LeKiwi(LeKiwiConfig(port=args.port_name, id=args.robot_id))

    bridge = RobotBridge(robot)
    bridge.start()

    if not args.no_gamepad:
        gamepad = GamepadInput(bridge)
        gamepad.start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if gamepad is not None:
            gamepad.stop()
        bridge.stop()


if __name__ == "__main__":
    main()
