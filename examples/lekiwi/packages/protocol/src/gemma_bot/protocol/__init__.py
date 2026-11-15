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

"""The protocol between the brain and an executor: HTTP + JSON, JPEG frames.

The brain plans and sees; an executor moves one robot's motors and refuses what isn't safe. They
only meet here, so either can run on another computer. Paths are versioned (/v1); an executor
also serves the same handlers at the paths before /v1 (/auto/...) while older brains exist.

    GET  /v1/describe                    the robot (Description): joints, gripper, base, cameras
    POST /v1/session/enable              take control (auto mode); heartbeats keep it
    POST /v1/session/disable             give it back (the gamepad)
    POST /v1/session/heartbeat           more often than heartbeat_timeout_s, or the robot stops
    POST /v1/cmd                         one command: joints, trajectory, gripper, base, probe
    POST /v1/stop                        stop whatever runs
    GET  /v1/status                      joints, targets, the running command, gripper hold, odom
    GET  /v1/frame/{role}?after=t        one JPEG captured after the executor's time t
    GET  /v1/calibration                 the arm's joint calibration (degrees from it)
    POST /v1/calibration                 write it (the arm calibrating itself)
    GET  /v1/camera/{role}/controls      a camera's image controls; POST sets, .../reset resets
"""

from gemma_bot.protocol.describe import Base, Camera, Description, Gripper, Joint

VERSION = 1
PREFIX = f"/v{VERSION}"

DESCRIBE = f"{PREFIX}/describe"
ENABLE = f"{PREFIX}/session/enable"
DISABLE = f"{PREFIX}/session/disable"
HEARTBEAT = f"{PREFIX}/session/heartbeat"
CMD = f"{PREFIX}/cmd"
STOP = f"{PREFIX}/stop"
STATUS = f"{PREFIX}/status"
CALIBRATION = f"{PREFIX}/calibration"


def frame(role: str) -> str:
    return f"{PREFIX}/frame/{role}"


def camera_controls(role: str) -> str:
    return f"{PREFIX}/camera/{role}/controls"


# The paths before /v1, by their /v1 path: what an executor still serves for an older brain.
LEGACY = {
    ENABLE: "/auto/enable",
    DISABLE: "/auto/disable",
    HEARTBEAT: "/auto/heartbeat",
    CMD: "/auto/cmd",
    STOP: "/auto/stop",
    STATUS: "/auto/status",
    CALIBRATION: "/auto/calibration",
    frame("{cam_name}"): "/auto/frame/{cam_name}",
    camera_controls("{cam_name}"): "/camera/{cam_name}/controls",
    camera_controls("{cam_name}") + "/reset": "/camera/{cam_name}/controls/reset",
}

__all__ = [
    "Base",
    "Camera",
    "Description",
    "Gripper",
    "Joint",
    "VERSION",
    "LEGACY",
    "DESCRIBE",
    "ENABLE",
    "DISABLE",
    "HEARTBEAT",
    "CMD",
    "STOP",
    "STATUS",
    "CALIBRATION",
    "frame",
    "camera_controls",
]
