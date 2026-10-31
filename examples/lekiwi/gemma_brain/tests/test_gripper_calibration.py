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

"""calibration/calibrate_gripper.py's close lags: what a stall-lag threshold must sit between."""

from calibration.calibrate_gripper import close_lags


def _close(stop_deg, lag=1.5, speed=39.0, dt=0.05, n=30):
    """A close from 40 deg at `speed`, the jaws `lag` behind their target, stopped at `stop_deg`."""
    trace = []
    for k in range(n):
        tgt = max(0.0, 40.0 - speed * dt * k)
        trace.append({"t": k * dt, "joints": {"arm_gripper": max(stop_deg, tgt + lag)}, "targets": {"arm_gripper": tgt}})
    return trace


def test_a_close_on_an_object_reports_its_free_and_stall_lags():
    lags = close_lags(_close(stop_deg=22.0)[:15])  # the Pi ends the close soon after the stall
    assert lags["free_lag_max_deg"] == 1.5 and lags["stall_lag_deg"] > 4.0


def test_too_short_a_trace_reports_nothing():
    assert close_lags(_close(22.0)[:2]) == {}
