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

"""Client for the LeKiwi web server's auto-mode HTTP API (web_server/auto_control.py).

Each primitive is submitted, then waited on until it is done, failed or aborted. A background
thread sends heartbeats while the client is enabled; if this process stalls or dies, the Pi
stops the robot on its own within auto_control.HEARTBEAT_TIMEOUT_S.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://10.55.0.102:8000"
HEARTBEAT_PERIOD_S = 0.2


class RobotError(RuntimeError):
    """A primitive was refused, failed, or aborted; .result has the Pi's record."""

    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result or {}


class OperatorTookOver(RobotError):
    """Auto mode was left on the robot side (gamepad, e-stop, web UI): never resume on our own."""


class Robot:
    def __init__(self, url: str = DEFAULT_URL):
        self.url = url.rstrip("/")
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        # Pi clock minus Mac clock. The Pi has no RTC and often no internet, so after a reboot
        # its wall clock can be hours off; every Pi timestamp (frame capture, command finish)
        # is converted through this, refreshed on each status() call.
        self._clock_offset: float | None = None

    # --- transport ---

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 5.0):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                return r.headers, payload
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                reason = json.loads(payload).get("reason", payload.decode())
            except ValueError:
                reason = payload.decode(errors="replace")
            raise RobotError(f"{method} {path}: HTTP {e.code}: {reason}") from None

    def _json(self, method: str, path: str, body: dict | None = None, timeout: float = 5.0) -> dict:
        _, payload = self._request(method, path, body, timeout)
        return json.loads(payload)

    # --- session ---

    def enable(self) -> None:
        """Switch the robot into auto mode and start heartbeats."""
        self._json("POST", "/auto/enable")
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        self._hb_thread.start()
        logger.info("auto mode enabled")

    def disable(self) -> None:
        self._hb_stop.set()
        try:
            self._json("POST", "/auto/disable")
        except (RobotError, OSError) as e:
            logger.warning("disable failed: %s", e)

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(HEARTBEAT_PERIOD_S):
            try:
                self._json("POST", "/auto/heartbeat", timeout=1.0)
            except (RobotError, OSError) as e:
                logger.warning("heartbeat failed: %s", e)

    def __enter__(self) -> Robot:
        self.enable()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.stop()
        except (RobotError, OSError):
            pass
        self.disable()

    # --- state ---

    def status(self) -> dict:
        t0 = time.time()
        st = self._json("GET", "/auto/status")
        t1 = time.time()
        self._clock_offset = st["time"] - (t0 + t1) / 2
        return st

    def frame(self, cam: str, after: float = 0.0) -> tuple[bytes, float]:
        """JPEG bytes from 'front' or 'wrist' captured after `after` (Mac wall-clock time,
        e.g. time.time() for "a fresh frame"). Returns (jpeg, capture time on the Mac clock)."""
        if self._clock_offset is None:
            self.status()
        off = self._clock_offset
        after_pi = after + off if after > 0 else 0.0
        headers, jpeg = self._request("GET", f"/auto/frame/{cam}?after={after_pi:.4f}", timeout=3.0)
        return jpeg, float(headers.get("X-Capture-Time", "0")) - off

    def stop(self) -> None:
        self._json("POST", "/auto/stop")

    # --- primitives (each blocks until it finishes; raises RobotError unless it's "done") ---

    def run(self, cmd: dict, timeout: float = 60.0) -> dict:
        reply = self._json("POST", "/auto/cmd", cmd, timeout=10.0)
        cmd_id = reply["id"]
        deadline = time.monotonic() + timeout
        while True:
            st = self.status()
            c = st.get("command") or {}
            if st["mode"] != "auto":
                raise OperatorTookOver(f"robot left auto mode (now {st['mode']}): operator took over", c)
            if c.get("id") == cmd_id and c["state"] != "running":
                if c["state"] != "done":
                    raise RobotError(f"{cmd['cmd']} {c['state']}: {c['reason']}", c)
                return c
            if time.monotonic() > deadline:
                self.stop()
                raise RobotError(f"{cmd['cmd']}: no result within {timeout:.0f} s, stopped")
            time.sleep(0.05)

    def calibration(self) -> dict:
        return self._json("GET", "/auto/calibration")

    def trajectory(self, waypoints: list[dict], durations: list[float], z_min: float | None = None) -> dict:
        total = sum(durations)
        return self.run(
            {"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min},
            timeout=total + 15.0,
        )

    def joints(self, targets: dict[str, float], speed: float | None = None, z_min: float | None = None) -> dict:
        return self.run({"cmd": "joints", "targets": targets, "speed": speed, "z_min": z_min})

    def gripper(self, pos: float, speed: float | None = None) -> dict:
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed})

    def base(self, dx: float = 0.0, dy: float = 0.0, dtheta_deg: float = 0.0, speed: float | None = None) -> dict:
        return self.run({"cmd": "base", "dx": dx, "dy": dy, "dtheta_deg": dtheta_deg, "speed": speed})
