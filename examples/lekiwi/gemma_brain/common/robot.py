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

import http.client
import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Where the LeKiwi web service can be, tried in this order (see find_url): the Ethernet cable
# to this Mac (fixed addresses), the robot's own Wi-Fi hotspot LeKiwi-RoboNet (the Pi is its
# gateway), the Pi's name on a shared Wi-Fi network.
CANDIDATE_URLS = ("http://10.55.0.102:8000", "http://10.42.0.1:8000", "http://raspberrypi.local:8000")
DEFAULT_URL = CANDIDATE_URLS[0]


def find_url(candidates=CANDIDATE_URLS, timeout: float = 1.5) -> str:
    """The first candidate whose /auto/status answers; raises RobotError if none does."""
    for url in candidates:
        try:
            with urllib.request.urlopen(url + "/auto/status", timeout=timeout) as r:
                if r.status == 200:
                    return url
        except OSError:
            continue
    raise RobotError(f"no LeKiwi web service answered at {', '.join(candidates)}")
# Heartbeats every 0.1 s on one kept-open connection (no TCP handshake per beat over a laggy
# Wi-Fi link): a single late reply no longer comes near the Pi's 1.5 s timeout.
# The auto interface's joint positions (web_server/joint_angles.py): degrees, since Oct 29.
JOINT_UNITS = "deg"
HEARTBEAT_PERIOD_S = 0.1
HEARTBEAT_TIMEOUT_S = 1.0
ENABLE_TIMEOUT_S = 3.0  # convention: auto mode's first control step takes a few of its ticks


class RobotError(RuntimeError):
    """A primitive was refused, failed, or aborted; .result has the Pi's record."""

    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result or {}


class BaseLocked(RobotError):
    """A base move was asked for while the base is locked (Robot(base_enabled=False))."""


class OperatorTookOver(RobotError):
    """Auto mode was left on the robot side (gamepad, e-stop, web UI): never resume on our own."""


class Robot:
    def __init__(self, url: str | None = None, base_enabled: bool = True, require_calibrated: bool = True):
        """url None: find the robot on whichever link is up (find_url). require_calibrated: enable()
        refuses an arm on a provisional calibration (its motors don't match the saved file: tasks
        would run on the wrong zeros); only the calibration itself turns it off."""
        self.url = (url or find_url()).rstrip("/")
        # False: every base (wheel) command is refused here, before it reaches the robot, e.g.
        # while the robot is tethered to power and must not drive.
        self.base_enabled = base_enabled
        self.require_calibrated = require_calibrated
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
        """Switch the robot into auto mode and start heartbeats. Refused, before anything
        changes, when the Pi's auto interface isn't in degrees (JOINT_UNITS): one on the version
        before Oct 29 would read this planner's degrees as its normalized units."""
        st = self._json("GET", "/auto/status")
        units = st.get("joint_units")
        if self.require_calibrated and st.get("arm_calibration") == "provisional":
            raise RobotError("the arm is on a provisional calibration (its motors don't match the saved file): "
                             "run calibration/calibrate_all.py first")
        if units != JOINT_UNITS:
            raise RobotError(f"the Pi's auto interface reports joint units {units!r}, this planner sends "
                             f"{JOINT_UNITS!r}: deploy web_server to the Pi first")
        self._json("POST", "/auto/enable")
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        self._hb_thread.start()
        # Auto mode holds no joint targets until its first control step; a command sent before
        # that is refused "no joint state yet" (Oct 29: the first one after a service restart).
        deadline = time.monotonic() + ENABLE_TIMEOUT_S
        while not self._json("GET", "/auto/status").get("targets"):
            if time.monotonic() > deadline:
                raise RobotError(f"auto mode has no joint state {ENABLE_TIMEOUT_S:.0f} s after enable")
            time.sleep(0.05)
        logger.info("auto mode enabled")

    def disable(self) -> None:
        self._hb_stop.set()
        try:
            self._json("POST", "/auto/disable")
        except (RobotError, OSError) as e:
            logger.warning("disable failed: %s", e)

    def _heartbeat_loop(self) -> None:
        host = self.url.split("://", 1)[1]
        conn = None
        while not self._hb_stop.wait(HEARTBEAT_PERIOD_S):
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(host, timeout=HEARTBEAT_TIMEOUT_S)
                conn.request("POST", "/auto/heartbeat", body=b"", headers={"Content-Length": "0"})
                conn.getresponse().read()
            except (OSError, http.client.HTTPException) as e:
                logger.warning("heartbeat failed: %s", e)
                if conn is not None:
                    conn.close()
                conn = None
        if conn is not None:
            conn.close()

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
            try:
                st = self.status()
            except OSError as e:  # a Wi-Fi stall: keep asking until the deadline
                logger.warning("status failed while waiting for %s: %s", cmd["cmd"], e)
                if time.monotonic() > deadline:
                    raise RobotError(f"{cmd['cmd']}: robot unreachable for {timeout:.0f} s") from None
                time.sleep(0.2)
                continue
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

    def write_calibration(self, joints: dict[str, dict], save: bool, relax_s: float = 0.0) -> dict:
        """The arm's motor calibration ({joint: {homing_offset, range_min, range_max}}, ticks),
        written by the Pi between two control ticks, the arm's torque briefly off (it should be
        resting); save: to its calibration file too (calibration/self_calibrate.py); relax_s: the
        torque off that long first, for the arm to settle where it lies."""
        body = {"joints": joints, "save": save, **({"relax_s": relax_s} if relax_s else {})}
        return self._json("POST", "/auto/calibration", body, timeout=10.0 + relax_s)

    def trajectory(
        self, waypoints: list[dict], durations: list[float], z_min: float | None = None, contact_stop: bool = False
    ) -> dict:
        """contact_stop: the Pi stops the move itself where the TCP meets something (it watches
        measured vs commanded TCP height every tick) and reports where (result["contact"])."""
        total = sum(durations)
        cmd = {"cmd": "trajectory", "waypoints": waypoints, "durations": durations, "z_min": z_min, "units": JOINT_UNITS}
        if contact_stop:
            cmd["contact_stop"] = True
        return self.run(cmd, timeout=total + 15.0)

    def joints(self, targets: dict[str, float], speed: float | None = None, z_min: float | None = None) -> dict:
        """targets in degrees (common/arm.py), speed in deg/s."""
        return self.run({"cmd": "joints", "targets": targets, "speed": speed, "z_min": z_min, "units": JOINT_UNITS})

    def gripper(self, pos: float, speed: float | None = None) -> dict:
        """pos: degrees open from closed (0), speed in deg/s."""
        return self.run({"cmd": "gripper", "pos": pos, "speed": speed, "units": JOINT_UNITS})

    def base(self, dx: float = 0.0, dy: float = 0.0, dtheta_deg: float = 0.0, speed: float | None = None,
             theta_speed: float | None = None) -> dict:
        """speed: m/s for driving, theta_speed: deg/s for turning (None: the Pi's defaults)."""
        if not self.base_enabled:
            raise BaseLocked(f"base move refused (base locked): dx={dx} dy={dy} dtheta={dtheta_deg}")
        try:
            return self._base(dx, dy, dtheta_deg, speed, theta_speed)
        except RobotError as e:
            moved = (e.result.get("result") or {}).get("moved")
            if "heartbeat lost" not in str(e) or not moved:
                raise
            # Cut off by a link stall: finish the rest of the move, once (in the robot's frame at
            # the stop, which for these short moves is close enough to the start's).
            rest = (dx - moved[0], dy - moved[1], dtheta_deg - moved[2])
            logger.warning("base move cut off by a heartbeat gap after %s; finishing %s", moved, rest)
            if abs(rest[0]) < 0.01 and abs(rest[1]) < 0.01 and abs(rest[2]) < 2.0:
                return {**e.result, "state": "done", "reason": "finished (close enough after a link stall)"}
            return self._base(*rest, speed, theta_speed)

    def _base(self, dx: float, dy: float, dtheta_deg: float, speed: float | None,
              theta_speed: float | None = None) -> dict:
        cmd = {"cmd": "base", "dx": dx, "dy": dy, "dtheta_deg": dtheta_deg, "speed": speed}
        if theta_speed is not None:
            cmd["theta_speed"] = theta_speed
        return self.run(cmd)
