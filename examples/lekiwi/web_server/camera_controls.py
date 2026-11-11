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

"""The cameras' image controls (exposure, brightness, ...) set from the web page, through
v4l2-ctl, saved on the Pi and applied again at startup and after a camera reconnects.

Nothing set them before (Nov 11): both Innomaker U20CAM cameras ran on their defaults, auto
exposure in aperture-priority mode, which exposes for the whole frame. In daylight the wrist
camera, mostly seeing the light floor, blew the top of a tennis ball out to white; the brain
then read the ball as a 27-36 mm long thing and gripped it like a brick (Nov 8 runs).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# The controls the page offers, in the order it shows them; anything else v4l2-ctl lists is left
# alone. auto_exposure first: exposure_time_absolute only takes in manual mode (1).
CONTROLS = (
    "auto_exposure",
    "exposure_time_absolute",
    "exposure_dynamic_framerate",
    "brightness",
    "gain",
    "backlight_compensation",
    "white_balance_automatic",
    "power_line_frequency",
)
STORE = Path.home() / ".config" / "lekiwi" / "camera_controls.json"
V4L2_TIMEOUT_S = 5.0

# "  brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0 flags=inactive"
_CTRL = re.compile(r"^\s*(\w+) 0x[0-9a-f]+ \((\w+)\)\s*:\s*(.*)$")
_MENU = re.compile(r"^\s+(\d+): (.+)$")


def parse_ctrls(text: str) -> dict[str, dict]:
    """v4l2-ctl --list-ctrls-menus output as {name: {type, value, min, max, step, default,
    inactive, menu}}; only the CONTROLS."""
    out: dict[str, dict] = {}
    last = None
    for line in text.splitlines():
        m = _CTRL.match(line)
        if m:
            name, kind, rest = m.groups()
            fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
            last = None
            if name not in CONTROLS:
                continue
            c = {
                "type": kind,
                "value": int(fields["value"]),
                "default": int(fields["default"]),
                "inactive": "flags=inactive" in rest,
            }
            if kind == "bool":
                c.update(min=0, max=1, step=1)
            else:
                c.update(min=int(fields["min"]), max=int(fields["max"]), step=int(fields.get("step", 1)))
            if kind == "menu":
                c["menu"] = {}
            out[name] = last = c
            continue
        m = _MENU.match(line)
        if m and last is not None and "menu" in last:
            last["menu"][int(m.group(1))] = m.group(2).strip()
    return {n: out[n] for n in CONTROLS if n in out}  # the page's order, exposure first


class CameraControls:
    """The cameras' controls by camera name ("front", "wrist") and device path."""

    def __init__(self, devices: dict[str, str], store: Path = STORE, run=subprocess.run):
        self.devices, self.store, self._run = dict(devices), store, run

    def _v4l2(self, cam: str, *args: str) -> str:
        r = self._run(
            ["v4l2-ctl", "-d", self.devices[cam], *args],
            capture_output=True,
            text=True,
            timeout=V4L2_TIMEOUT_S,
        )
        if r.returncode != 0:
            raise RuntimeError(f"v4l2-ctl on the {cam} camera: {(r.stderr or r.stdout).strip()}")
        return r.stdout

    def read(self, cam: str) -> dict[str, dict]:
        return parse_ctrls(self._v4l2(cam, "--list-ctrls-menus"))

    def _saved(self) -> dict[str, dict[str, int]]:
        try:
            return json.loads(self.store.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, saved: dict) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps(saved, indent=1, sort_keys=True))

    def _write(self, cam: str, values: dict[str, int], now: dict[str, dict]) -> dict[str, int]:
        """Each value clamped to the camera's range (a menu's to its entries), in CONTROLS
        order, one v4l2-ctl call each: a failure names its control."""
        done = {}
        for name in CONTROLS:
            if name not in values or name not in now:
                continue
            c = now[name]
            v = min(max(int(values[name]), c["min"]), c["max"])
            if c.get("menu") and v not in c["menu"]:
                v = c["value"]  # not one of its entries (auto_exposure has 1 and 3): unchanged
            self._v4l2(cam, f"--set-ctrl={name}={v}")
            done[name] = v
        return done

    def set(self, cam: str, values: dict) -> dict[str, dict]:
        """Set some controls, keep them for the next start; the controls as they now read."""
        unknown = set(values) - set(CONTROLS)
        if unknown:
            raise ValueError(f"not a camera control here: {', '.join(sorted(unknown))}")
        done = self._write(cam, values, self.read(cam))
        saved = self._saved()
        saved[cam] = {**saved.get(cam, {}), **done}
        self._save(saved)
        logger.info("camera %s: set %s", cam, done)
        return self.read(cam)

    def reset(self, cam: str) -> dict[str, dict]:
        """Every control back to the camera's default, nothing kept."""
        now = self.read(cam)
        self._write(cam, {n: c["default"] for n, c in now.items()}, now)
        saved = self._saved()
        saved.pop(cam, None)
        self._save(saved)
        logger.info("camera %s: controls back to their defaults", cam)
        return self.read(cam)

    def apply_saved(self) -> None:
        """The kept controls, on every camera that has some (at startup, after a reconnect).
        A camera that fails is logged; the rest still get theirs."""
        for cam, values in self._saved().items():
            if cam not in self.devices:
                continue
            try:
                self._write(cam, values, self.read(cam))
                logger.info("camera %s: kept controls applied %s", cam, values)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
                logger.warning("camera %s: kept controls not applied: %s", cam, e)
