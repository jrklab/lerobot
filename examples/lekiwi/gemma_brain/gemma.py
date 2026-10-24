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

"""Gemma 4 calls through a local Ollama server, each returned with its timing so every
decision can be logged next to the frame it was made on.

Perception questions are short and direct, asked with the image first, thinking off and
greedy decoding: the model answers what it sees, and code does the geometry.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:e4b"


@dataclass
class Answer:
    text: str
    value: object = None
    ms: float = 0.0
    prompt_ms: float = 0.0
    out_tokens: int = 0
    raw: dict = field(default_factory=dict, repr=False)


class Gemma:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
        max_tokens: int = 200,
        timeout: float = 60.0,
    ) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens},
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            self.url + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def ask_image(self, jpeg: bytes, question: str, max_tokens: int = 60) -> Answer:
        t = time.monotonic()
        r = self.chat(
            [{"role": "user", "content": question, "images": [base64.b64encode(jpeg).decode()]}],
            max_tokens=max_tokens,
        )
        return Answer(
            text=r["message"]["content"].strip(),
            ms=(time.monotonic() - t) * 1000,
            prompt_ms=r.get("prompt_eval_duration", 0) / 1e6,
            out_tokens=r.get("eval_count", 0),
            raw=r,
        )

    def box(self, jpeg: bytes, what: str) -> Answer:
        """Bounding box of `what` as (x0, y0, x1, y1) in 0..1 image fractions, or None."""
        q = (
            f"Find {what} in this image. Reply with JSON only: "
            '{"box_2d": [ymin, xmin, ymax, xmax]} with coordinates normalized to 0-1000, '
            f'or {{"box_2d": null}} if there is no {what}.'
        )
        a = self.ask_image(jpeg, q)
        a.value = parse_box(a.text)
        return a

    def yes_no(self, jpeg: bytes, question: str) -> Answer:
        a = self.ask_image(jpeg, question + " Answer with one word: yes or no.", max_tokens=4)
        word = a.text.strip().strip(".!").lower()
        a.value = True if word.startswith("yes") else False if word.startswith("no") else None
        return a


def parse_box(text: str):
    """Gemma's {"box_2d": [ymin, xmin, ymax, xmax]} (0-1000) -> (x0, y0, x1, y1) in 0..1."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        box = json.loads(m.group(0)).get("box_2d")
    except (ValueError, AttributeError):
        return None
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return None
    y0, x0, y1, x1 = (max(0.0, min(1.0, v / 1000.0)) for v in box)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
