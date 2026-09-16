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
import socket
import sys
import threading
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

from episode_recorder import EpisodeRecorder  # noqa: E402
from gamepad_input import GamepadInput  # noqa: E402
from robot_bridge import ARM_JOINTS, BASE_SPEEDS, FPS, JOG_SPEEDS, MockLeKiwi, RobotBridge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="LeKiwi Web Control")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

bridge: RobotBridge | None = None
gamepad: GamepadInput | None = None

# Uploading to the Hub only works when the Pi has a route to the internet (only true when
# Ethernet is plugged into the host -- the Pi's own hotspot mode, used for cordless ground
# testing, has no upstream route at all). Checked periodically in the background rather than
# on every status push, since a real check involves a network round-trip/timeout.
_internet_reachable = False


def _internet_check_loop() -> None:
    global _internet_reachable
    while True:
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=3):
                _internet_reachable = True
        except OSError:
            _internet_reachable = False
        time.sleep(15)


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
                reply = _handle_message(raw)
                if reply is not None:
                    await websocket.send_text(json.dumps(reply))
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
                            "control_mode": bridge.get_control_mode(),
                            "recording": bridge.get_recording_status(),
                            "internet_reachable": _internet_reachable,
                        }
                    )
                )
            except Exception:
                return
        await asyncio.sleep(0.2)


def _handle_message(raw: str) -> dict | None:
    """Returns an optional message to send back to just this client (e.g. an error the
    sender should see immediately, distinct from the broadcast periodic state message)."""
    if bridge is None:
        return None
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
        # "source" distinguishes the on-page joystick ("web") from the leader/keyboard
        # companion script's WASD-derived base commands ("leader_keyboard") -- either way
        # RobotBridge only applies it if that source is the currently active control mode.
        source = msg.get("source", "web")
        if source not in ("web", "leader_keyboard"):
            source = "web"
        bridge.update_base(
            x=float(msg.get("x", 0.0)),
            y=float(msg.get("y", 0.0)),
            theta=float(msg.get("theta", 0.0)),
            speed=speed,
            source=source,
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
        bridge.update_jog(joint, direction, speed, source="web")
    elif msg_type == "arm_pos":
        positions = msg.get("positions")
        if not isinstance(positions, dict):
            logger.warning("Dropping malformed arm_pos message: %s", raw)
            return
        try:
            bridge.set_arm_absolute({k: float(v) for k, v in positions.items()})
        except (TypeError, ValueError):
            logger.warning("Dropping arm_pos message with non-numeric positions: %s", raw)
    elif msg_type == "set_mode":
        mode = msg.get("mode")
        try:
            bridge.set_control_mode(mode)
        except ValueError:
            logger.warning("Dropping set_mode message with unknown mode: %s", mode)
    elif msg_type == "reset_arm":
        logger.info("Resetting arm to neutral pose.")
        bridge.reset_arm()
    elif msg_type == "start_recording":
        task = str(msg.get("task", "")).strip()
        if not task:
            logger.warning("Dropping start_recording message with empty task.")
            return {"type": "error", "message": "Enter a task description before recording."}
        if not bridge.start_recording(task):
            logger.warning("start_recording refused (already recording, playback active, or disabled).")
            return {
                "type": "error",
                "message": "Couldn't start recording -- still saving the previous episode, or a "
                "playback is in progress. Try again in a moment.",
            }
    elif msg_type == "stop_recording":
        bridge.stop_recording()
    elif msg_type == "discard_recording":
        bridge.discard_recording()
    elif msg_type == "start_playback":
        try:
            episode = int(msg.get("episode"))
        except (TypeError, ValueError):
            logger.warning("Dropping start_playback message with invalid episode: %s", raw)
            return {"type": "error", "message": "Invalid episode."}
        if not bridge.start_playback(episode):
            logger.warning("start_playback refused for episode %d.", episode)
            return {
                "type": "error",
                "message": "Couldn't start playback -- a recording or another playback is in progress.",
            }
    elif msg_type == "stop_playback":
        bridge.stop_playback()
    elif msg_type == "upload_to_hub":
        bridge.upload_to_hub()
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
    parser.add_argument(
        "--repo-id",
        default="jrkhf/lekiwi_recordings",
        help="Hugging Face dataset repo id episodes are recorded into (one dataset per deployment).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local directory for the recorded dataset. Defaults to $HF_LEROBOT_HOME/<repo-id>.",
    )
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Disable episode recording/playback (enabled by default).",
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

    recorder = None
    if not args.no_recording:
        recorder = EpisodeRecorder(repo_id=args.repo_id, root=args.dataset_root, fps=FPS)

    bridge = RobotBridge(robot, recorder=recorder)
    bridge.start()

    if not args.no_gamepad:
        gamepad = GamepadInput(bridge)
        gamepad.start()

    threading.Thread(target=_internet_check_loop, daemon=True).start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if gamepad is not None:
            gamepad.stop()
        bridge.stop()


if __name__ == "__main__":
    main()
