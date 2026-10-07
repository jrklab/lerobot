# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig


_CAMERA_BY_PATH = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:{port}:1.0-video-index0"


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": OpenCVCameraConfig(
            # Stable USB-topology-keyed path (physical port 1.2), not a raw /dev/videoN index
            # -- both cameras are the same model and report the identical USB serial number,
            # so /dev/v4l/by-id/ can't tell them apart (it silently collides/aliases), unlike
            # the motor bus's /dev/serial/by-id/ path elsewhere in this project. by-path is
            # keyed by physical port position instead, which stays stable across a USB
            # disconnect/reconnect re-enumeration event (confirmed to happen on this hardware
            # -- see index_or_path="/dev/video0" below/dmesg history) even though the raw
            # /dev/videoN index it points to can change.
            index_or_path=_CAMERA_BY_PATH.format(port="1.2"),
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
            warmup_s=5,
            # Right after boot (or a fresh host restart) these USB UVC cameras can take up to
            # ~90s to reliably deliver frames while the driver/hardware settles, then work fine.
            # Retry instead of failing the whole host process over a transient boot-time hiccup.
            connect_retry_timeout_s=120,
            connect_retry_interval_s=3,
        ),
        "wrist": OpenCVCameraConfig(
            # Physical port 1.1 -- was hardcoded as /dev/video0 until a real USB
            # disconnect/reconnect re-enumerated it to /dev/video1, which then made every
            # connect attempt fail against the now-stale /dev/video0 path until fixed here.
            index_or_path=_CAMERA_BY_PATH.format(port="1.1"),
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            warmup_s=5,
            connect_retry_timeout_s=120,
            connect_retry_interval_s=3,
        ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    port: str = "/dev/ttyUSB0"  # port to connect to the bus
    id: str = "kiwi_sn_0"  # robot id

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@dataclass
class LeKiwiHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 3000

    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Must match the `use_degrees` setting on the LeKiwi server (lekiwi_host).
    # False → actions in normalized range [-100, 100] (default).
    # True  → actions in physical degrees.
    use_degrees: bool = False

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
