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

"""Offline checks for demo_extract's keyframe segmentation on a synthetic pick-and-place.

    .venv/bin/python test_demo_extract.py
"""

import sys

import numpy as np

from demo_extract import find_close, find_keyframes

rng = np.random.default_rng(0)


def demo(close_to=7.0, uneven=False):
    """A synthetic episode: hover pause, descend, pre-close pause, close, lift, pause, carry,
    pre-release pause, open. Returns joints (T, 6), z (T,), and the true frames."""
    segs = [("pause", 30, 0.12), ("move", 40, 0.02), ("pause", 20, 0.02), ("close", 20, 0.02),
            ("pause", 10, 0.02), ("move", 30, 0.11), ("pause", 20, 0.11), ("move", 50, 0.13),
            ("pause", 20, 0.13), ("open", 10, 0.13), ("pause", 10, 0.13)]
    joints, z, g, zc, true, t = [], [], 40.0, 0.12, {}, 0
    q = np.zeros(5)
    for kind, n, z_end in segs:
        for k in range(n):
            if kind == "move":
                q = q + 1.5  # clearly moving
                zc = zc + (z_end - zc) / (n - k)
            else:
                q = q + rng.normal(0, 0.05, 5)  # sensor noise while still
            if kind == "close":
                step = (40.0 - close_to) / n
                g = g - (0 if uneven and k in (6, 7) else step) - (step * 2 if uneven and k == 8 else 0)
                g = max(g, close_to)
            if kind == "open":
                g = g + 2.5
            joints.append([*q, g + rng.normal(0, 0.02)])
            z.append(zc)
        true[kind + f"_{len(true)}"] = (t, t + n - 1)
        t += n
    return np.array(joints), np.array(z), true


def test_keyframes_of_a_pick_and_place():
    joints, z, true = demo()
    kf = find_keyframes(joints, z)
    assert kf.held
    assert 88 <= kf.close_start <= 91 and 108 <= kf.close_end <= 112, kf  # the close is frames 90-109
    assert kf.pre_close == kf.close_start, kf  # the pre-close pause (70-89) runs up to the close
    assert kf.hover is not None and kf.hover <= 30 and z[kf.hover] > 0.1, kf  # the hover pause (0-29)
    assert kf.lift is not None and 150 <= kf.lift <= 152 and z[kf.lift] - z[kf.close_end] > 0.08, kf
    assert 238 <= kf.pre_release <= 239, kf  # the gripper opens at frame 240


def test_uneven_close_is_one_close():
    joints, z, _ = demo(uneven=True)
    s, e = find_close(joints[:, 5])
    assert joints[s, 5] - joints[e, 5] > 30, (s, e)


def test_close_on_nothing_is_not_held():
    joints, z, _ = demo(close_to=2.0)
    assert not find_keyframes(joints, z).held


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
    sys.exit(0)
