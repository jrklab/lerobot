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

"""camera_controls.py against a fake v4l2-ctl that answers like the wrist camera did (Nov 11)."""

import re
import subprocess

import pytest
from camera_controls import CONTROLS, CameraControls, parse_ctrls

# v4l2-ctl --list-ctrls-menus on the wrist camera (Innomaker U20CAM-1080p), Nov 11.
LISTING = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                       contrast 0x00980901 (int)    : min=0 max=64 step=1 default=32 value=32
                     saturation 0x00980902 (int)    : min=0 max=128 step=1 default=64 value=64
                            hue 0x00980903 (int)    : min=-40 max=40 step=1 default=0 value=0
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                          gamma 0x00980910 (int)    : min=72 max=500 step=1 default=100 value=100
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1 (50 Hz)
				0: Disabled
				1: 50 Hz
				2: 60 Hz
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=4600 flags=inactive
                      sharpness 0x0098091b (int)    : min=0 max=6 step=1 default=3 value=3
         backlight_compensation 0x0098091c (int)    : min=0 max=160 step=1 default=12 value=12

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
				1: Manual Mode
				3: Aperture Priority Mode
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=60 flags=inactive
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
"""


class FakeV4l2:
    """Keeps each control's value; --list-ctrls-menus answers LISTING with them, --set-ctrl
    changes one. The exposure time is inactive unless auto_exposure is manual (1)."""

    def __init__(self):
        self.values = {n: c["value"] for n, c in parse_ctrls(LISTING).items()}
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        arg = cmd[3]
        if arg.startswith("--set-ctrl="):
            name, v = arg.removeprefix("--set-ctrl=").split("=")
            if name == "exposure_time_absolute" and self.values["auto_exposure"] != 1:
                # the camera's own refusal of an inactive control (Nov 11, on the Pi)
                return subprocess.CompletedProcess(cmd, 255, "", "Error setting controls: Permission denied")
            self.values[name] = int(v)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        def line(m):
            name = m.group(1)
            text = re.sub(r"value=-?\d+", f"value={self.values.get(name, 0)}", m.group(0))
            text = text.replace(" flags=inactive", "")
            if name == "exposure_time_absolute" and self.values["auto_exposure"] != 1:
                text += " flags=inactive"
            return text

        out = re.sub(r"^\s*(\w+) 0x.*$", line, LISTING, flags=re.M)
        return subprocess.CompletedProcess(cmd, 0, out, "")


@pytest.fixture
def cams(tmp_path):
    fake = FakeV4l2()
    return CameraControls({"wrist": "/dev/wrist"}, store=tmp_path / "cc.json", run=fake), fake


def test_the_listing_reads_as_the_camera_has_it():
    c = parse_ctrls(LISTING)
    assert list(c) == list(CONTROLS)  # in the page's order, not the camera's listing
    assert c["auto_exposure"]["value"] == 3 and c["auto_exposure"]["menu"] == {
        1: "Manual Mode",
        3: "Aperture Priority Mode",
    }
    assert c["exposure_time_absolute"] == {
        "type": "int",
        "value": 60,
        "default": 157,
        "inactive": True,
        "min": 1,
        "max": 5000,
        "step": 1,
    }
    assert c["brightness"]["min"] == -64 and c["white_balance_automatic"]["type"] == "bool"
    assert c["power_line_frequency"]["menu"][2] == "60 Hz"


def test_manual_exposure_is_set_before_its_time_and_kept(cams):
    cc, fake = cams
    out = cc.set("wrist", {"exposure_time_absolute": 40, "auto_exposure": 1})
    sets = [c[3] for c in fake.calls if c[3].startswith("--set-ctrl")]
    assert sets == ["--set-ctrl=auto_exposure=1", "--set-ctrl=exposure_time_absolute=40"]
    assert out["exposure_time_absolute"]["value"] == 40 and not out["exposure_time_absolute"]["inactive"]
    # kept: a new process (the service restarting) applies it to a camera back on its defaults
    fresh = FakeV4l2()
    CameraControls({"wrist": "/dev/wrist"}, store=cc.store, run=fresh).apply_saved()
    assert fresh.values["auto_exposure"] == 1 and fresh.values["exposure_time_absolute"] == 40


def test_the_exposure_time_under_auto_exposure_is_left_alone(cams):
    cc, fake = cams
    cc.set("wrist", {"exposure_time_absolute": 40, "brightness": 5})
    assert fake.values["exposure_time_absolute"] == 60 and fake.values["brightness"] == 5


def test_values_are_kept_inside_the_cameras_range(cams):
    cc, fake = cams
    cc.set("wrist", {"brightness": -500, "gain": 900, "auto_exposure": 2})
    assert fake.values["brightness"] == -64 and fake.values["gain"] == 100
    assert fake.values["auto_exposure"] == 3  # 2 isn't one of its modes: left as it was


def test_an_unknown_control_is_refused_before_anything_is_set(cams):
    cc, fake = cams
    with pytest.raises(ValueError):
        cc.set("wrist", {"brightness": 5, "zoom_absolute": 3})
    assert not any(c[3].startswith("--set-ctrl") for c in fake.calls)


def test_reset_puts_the_defaults_back_and_forgets_them(cams):
    # Nov 11: under auto exposure the reset also set the exposure time, and the camera refused
    # it ("Permission denied"); manual mode first, so the reset has both modes to undo.
    cc, fake = cams
    cc.set(
        "wrist",
        {"brightness": -20, "power_line_frequency": 2, "auto_exposure": 1, "exposure_time_absolute": 30},
    )
    cc.reset("wrist")
    assert fake.values["brightness"] == 0 and fake.values["power_line_frequency"] == 1
    assert fake.values["auto_exposure"] == 3  # back to auto; the time then is the camera's own
    cc.reset("wrist")  # again, already under auto exposure: nothing refused
    fresh = FakeV4l2()
    fresh.values["brightness"] = 7
    CameraControls({"wrist": "/dev/wrist"}, store=cc.store, run=fresh).apply_saved()
    assert fresh.values["brightness"] == 7  # nothing kept for it any more


def test_a_failing_camera_at_startup_is_logged_not_raised(tmp_path):
    def broken(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "Cannot open device /dev/wrist")

    store = tmp_path / "cc.json"
    store.write_text('{"wrist": {"brightness": -10}}')
    CameraControls({"wrist": "/dev/wrist"}, store=store, run=broken).apply_saved()


def test_a_camera_back_on_its_defaults_gets_its_kept_controls_again(cams, caplog):
    # A replug or a USB drop the camera watchdog doesn't see: the camera comes back on its defaults.
    cc, fake = cams
    cc.set("wrist", {"power_line_frequency": 2, "brightness": -10})
    assert cc.check() == {}  # as kept: nothing to do
    fake.values.update(power_line_frequency=1, brightness=0)  # back on its defaults
    assert cc.check() == {"wrist": {"power_line_frequency": 2, "brightness": -10}}
    assert fake.values["power_line_frequency"] == 2 and fake.values["brightness"] == -10


def test_an_unplugged_camera_is_noted_once_and_put_right_when_back(cams, caplog):
    cc, fake = cams
    cc.set("wrist", {"brightness": -10})
    real = cc._run
    cc._run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Cannot open device")
    assert cc.check() == {} and cc.check() == {}
    assert sum("unplugged" in r.message for r in caplog.records) == 1
    fake.values["brightness"] = 0
    cc._run = real
    assert cc.check() == {"wrist": {"brightness": -10}}


def test_an_inactive_kept_control_is_not_a_loss(cams):
    # The exposure time kept from manual mode, now under auto exposure: inactive, not refought.
    cc, fake = cams
    cc.set("wrist", {"auto_exposure": 1, "exposure_time_absolute": 40})
    cc.set("wrist", {"auto_exposure": 3})
    fake.values["exposure_time_absolute"] = 157
    assert cc.check() == {}
