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

"""A run's log: events as JSON lines (runs/<time>-<what>/log.jsonl, also printed) and the
camera frames next to them, numbered in order."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class Log:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.f = (out / "log.jsonl").open("a")
        self.n = 0

    def ev(self, kind: str, **kw) -> None:
        rec = {"t": round(time.time(), 3), "ev": kind, **kw}
        self.f.write(json.dumps(rec, default=_jsonable) + "\n")
        self.f.flush()
        print(f"[{kind}] " + " ".join(f"{k}={_short(v)}" for k, v in kw.items()))

    def frame(self, tag: str, jpeg: bytes) -> str:
        self.n += 1
        name = f"{self.n:03d}_{tag}.jpg"
        (self.out / name).write_bytes(jpeg)
        return name


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.round(4).tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return str(v)


def _short(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) <= 4:
        return "[" + ", ".join(f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x) for x in v) + "]"
    return str(v)
