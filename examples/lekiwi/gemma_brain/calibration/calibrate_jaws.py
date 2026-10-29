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

# ruff: noqa: E402

"""Measure where the fixed jaw sits in the wrist image (calib.json "wrist_jaws"). The fixed jaw
is rigid with the wrist camera, so its tip row and inner edge are the same in every frame; this
takes the median over saved wrist frames from runs/ (no robot motion) and reports the spread.
Frames where a blue brick touches the jaw are outliers and fall away in the median.

    python calibration/calibrate_jaws.py [--runs 'runs/2026102*'] [--write]
"""

import argparse
import datetime
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from common.vision import jaw_geometry

CALIB = ROOT / "data" / "calib.json"
# frames taken with the arm at rest or just before closing: the jaws are in clear view
TAGS = ("plan_wrist", "pre_close")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "2026*"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    files = sorted(f for tag in TAGS for f in glob.glob(f"{args.runs}/*_{tag}*.jpg"))
    found = [g for f in files if (g := jaw_geometry(Path(f).read_bytes()))]
    if len(found) < 10:
        raise SystemExit(f"only {len(found)} frames with the jaws found in {len(files)}; need 10")
    value, spread = {}, {}
    for key in ("fixed_tip_v", "fixed_inner_u"):
        xs = [g[key] for g in found]
        med = statistics.median(xs)
        value[key] = round(float(med), 4)
        spread[key] = round(float(statistics.median(abs(x - med) for x in xs)), 4)
    print(f"{len(found)}/{len(files)} frames: {value}  median abs deviation {spread}")
    if args.write:
        calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
        calib["wrist_jaws"] = {
            "value": value,
            "unit": "wrist image fraction (u right, v down)",
            "meaning": "fixed jaw (right) in the wrist image: its tip row and its inner edge column just below the tip",
            "method": f"median of jaw_geometry over {len(found)} saved {'/'.join(TAGS)} frames",
            "spread_mad": spread,
            "date": datetime.date.today().isoformat(),
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")


if __name__ == "__main__":
    main()
