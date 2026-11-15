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

from __future__ import annotations

import importlib.util
import subprocess
import sys
from enum import Enum
from typing import Any, TypedDict

import numpy as np


def _probe_torch_available() -> bool:
    """Check if torch can actually be imported without crashing (SIGILL-safe)."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            timeout=15,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


_torch_available = _probe_torch_available()
if _torch_available:
    import torch
else:
    torch = None  # type: ignore[assignment]


class TransitionKey(str, Enum):
    """Keys for accessing EnvTransition dictionary components."""

    # TODO(Steven): Use consts
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


PolicyAction = torch.Tensor if _torch_available else Any
RobotAction = dict[str, Any]
EnvAction = np.ndarray
RobotObservation = dict[str, Any]


EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: RobotObservation | None,
        TransitionKey.ACTION.value: Any,
        TransitionKey.REWARD.value: Any,
        TransitionKey.DONE.value: Any,
        TransitionKey.TRUNCATED.value: Any,
        TransitionKey.INFO.value: dict[str, Any] | None,
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,
    },
)
