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

"""Is the link to the robot good enough to drive it? Sends heartbeats the way the brain does
(every 0.1 s, one kept-open connection) for a while and reports the reply times. Nothing moves.

    .venv/bin/python link_test.py [--seconds 30] [--robot http://10.42.0.1:8000]

The Pi stops any motion after HEARTBEAT_TIMEOUT_S (1.5 s) without a heartbeat: any gap near
that would cut a move short.
"""

import argparse
import http.client
import statistics
import time

from robot import find_url

PI_TIMEOUT_S = 1.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--robot", default=None)
    args = ap.parse_args()
    url = args.robot or find_url()
    host = url.split("://", 1)[1]
    print(f"robot at {url}; sending heartbeats for {args.seconds:.0f} s ...")
    conn, rtts, fails, last_ok, gaps = None, [], 0, time.monotonic(), []
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        t = time.monotonic()
        try:
            if conn is None:
                conn = http.client.HTTPConnection(host, timeout=1.0)
            conn.request("POST", "/auto/heartbeat", body=b"", headers={"Content-Length": "0"})
            conn.getresponse().read()
            now = time.monotonic()
            rtts.append(1000 * (now - t))
            gaps.append(now - last_ok)
            last_ok = now
        except (OSError, http.client.HTTPException):
            fails += 1
            if conn is not None:
                conn.close()
            conn = None
        time.sleep(max(0.0, 0.1 - (time.monotonic() - t)))
    rtts.sort()
    worst_gap = max(gaps) if gaps else float("inf")
    print(f"replies: {len(rtts)} ok, {fails} failed")
    if rtts:
        print(f"reply time: median {statistics.median(rtts):.0f} ms, 95% under {rtts[int(0.95 * (len(rtts) - 1))]:.0f} ms, worst {rtts[-1]:.0f} ms")
    print(f"longest time without a heartbeat getting through: {worst_gap:.2f} s "
          f"(the Pi stops motion at {PI_TIMEOUT_S} s)")
    verdict = "GOOD" if worst_gap < 0.5 else "OK, but marginal" if worst_gap < PI_TIMEOUT_S else "NOT GOOD ENOUGH"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
