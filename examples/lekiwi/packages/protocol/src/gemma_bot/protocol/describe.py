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

"""The robot an executor drives, as it describes itself at /v1/describe: what the brain may ask
of it. Skills check this (has a base? a wrist camera? how many arm joints?) instead of the robot's
name. Plain dataclasses, JSON both ways (to_dict / from_dict)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Joint:
    name: str
    min_deg: float  # what a command may reach: the calibrated range within the arm model's limits
    max_deg: float


@dataclass
class Gripper:
    name: str
    min_deg: float  # degrees of opening, 0 = closed
    max_deg: float


@dataclass
class Base:
    kind: str  # "omni" (drives any direction) or "diff" (forward and turn)
    max_speed_mps: float
    max_turn_dps: float
    max_move_m: float  # per axis, in one command
    max_turn_deg: float  # in one command
    pos_tol_m: float  # a drive ends within this of its goal
    turn_tol_deg: float  # a turn ends within this of its goal


@dataclass
class Camera:
    role: str  # "front", "scene", "wrist": what it looks at; also its name in /v1/frame/{role}
    mount: str  # "body" (fixed on the robot), "arm" (moves with the gripper), "world" (fixed in the room)
    controls: bool = False  # image controls at /v1/camera/{role}/controls


@dataclass
class Description:
    protocol: int
    embodiment: str  # "lekiwi", "so101_desk": which robot, for logs and configs, never for skills
    joint_units: str  # "deg"
    arm_model: str  # the arm's kinematic model, e.g. "so101_new_calib"
    joints: list[Joint]  # the arm's joints, base to tip, without the gripper
    max_joint_speed_dps: float
    heartbeat_timeout_s: float  # the robot stops when the brain is silent this long
    gripper: Gripper | None = None
    base: Base | None = None
    cameras: list[Camera] = field(default_factory=list)
    calibration: str | None = None  # "calibrated", "provisional", or None (no motors: a mock)

    def camera(self, role: str) -> Camera | None:
        return next((c for c in self.cameras if c.role == role), None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Description:
        """Fields this version doesn't know (from a newer executor) are left out."""
        d = _known(cls, d)
        d["joints"] = [Joint(**_known(Joint, j)) for j in d["joints"]]
        d["gripper"] = Gripper(**_known(Gripper, d["gripper"])) if d.get("gripper") else None
        d["base"] = Base(**_known(Base, d["base"])) if d.get("base") else None
        d["cameras"] = [Camera(**_known(Camera, c)) for c in d.get("cameras", [])]
        return cls(**d)


def _known(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}
