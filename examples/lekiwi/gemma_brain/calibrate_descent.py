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

"""Fit how a floor point moves in the wrist image as the gripper goes straight down
(calib.json "wrist_descent"), from the logged picks in runs/ (no robot motion).

With the jaw tips h above the floor, a floor point's image offset from the point the descent
heads for (p0) scales as 1 / (depth_m + h), depth_m being the camera's height above the tips.
Every logged descent between two looks with no sideways move in between is one sample:
  - depth_m from how much the brick's box grows (its width ratio) between the two heights;
  - p0, with depth_m fixed, by least squares on the box edges (edges cut by the frame skipped);
  - the scatter of the last descent (3 cm look -> just before closing) across the jaws (u):
    how far the brick lands from where the model put it, in mm at the floor. The arm's sag
    differs with reach and roll, so this is noise the final check must allow for.

    python calibrate_descent.py [--runs 'runs/2026*'] [--write]
"""

import argparse
import datetime
import glob
import json
from pathlib import Path

import numpy as np

CALIB = Path(__file__).with_name("calib.json")
MOVES = ("servo_step", "along_step", "base", "lineup_arm", "roll_step")


def descents(log: Path) -> list[dict]:
    """Consecutive looks at the brick (servo and pre-close boxes) with only a straight descent
    between them: [{"a": (h, box), "b": (h, box), "last": bool}]."""
    looks, floor = [], None
    for line in log.open():
        e = json.loads(line)
        ev, tag = e.get("ev"), str(e.get("tag", ""))
        if ev in ("box", "snap") and e.get("box") and tag.startswith(("servo_h", "pre_close_check")):
            if looks and looks[-1]["tag"] == tag:
                looks[-1]["box"] = e["box"]  # the snapped box replaces Gemma's
            else:
                h = looks[-1].get("pre_close_h") if tag.startswith("pre_close") and looks else None
                looks.append({"tag": tag, "box": e["box"], "h": h, "moved": False})
        elif ev == "servo" and looks:
            looks[-1]["h"] = e["h"]
        elif ev in MOVES and looks:
            looks[-1]["moved"] = True
        elif ev == "floor_predicted":
            floor = e["z"]
        elif ev == "pre_close" and looks and floor is not None and "z_meas" in e:
            looks[-1]["pre_close_h"] = e["z_meas"] - floor
    out = []
    for a, b in zip(looks, looks[1:], strict=False):
        if a["moved"] or a["h"] is None or b["h"] is None or abs(a["h"] - b["h"]) < 0.01:
            continue
        out.append({"a": (a["h"], a["box"]), "b": (b["h"], b["box"]),
                    "last": a["tag"].startswith("servo_h3") and b["tag"].startswith("pre_close")})
    return out


def inside(v: float) -> bool:
    return 0.01 < v < 0.99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/2026*")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    pairs = [d for log in sorted(glob.glob(f"{args.runs}/log.jsonl")) for d in descents(Path(log))]
    if len(pairs) < 20:
        raise SystemExit(f"only {len(pairs)} logged descents; need 20")

    # depth from the width ratio: wb / wa = (D + ha) / (D + hb)
    ds = []
    for p in pairs:
        (ha, ba), (hb, bb) = p["a"], p["b"]
        if all(inside(v) for v in (ba[0], ba[2], bb[0], bb[2])):
            s = (bb[2] - bb[0]) / (ba[2] - ba[0])
            if abs(s - 1) > 0.05:
                ds.append((ha - s * hb) / (s - 1))
    depth = float(np.median(ds))

    # p0: edge_b - edge_a * s = p0 * (1 - s), s = (D + ha) / (D + hb)
    rows, ys, w = [], [], []
    for p in pairs:
        (ha, ba), (hb, bb) = p["a"], p["b"]
        s = (depth + ha) / (depth + hb)
        for k, axis in ((0, 0), (2, 0), (1, 1), (3, 1)):
            if inside(ba[k]) and inside(bb[k]):
                r = np.zeros(2)
                r[axis] = 1 - s
                rows.append(r)
                ys.append(bb[k] - ba[k] * s)
                w.append(640.0 if axis == 0 else 480.0)
    a, y, w = np.array(rows), np.array(ys), np.array(w)
    p0 = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
    res_px = np.abs(a @ p0 - y) * w

    # last descent: right edge (the fixed jaw's side) against the model, mm at the floor
    jac = np.array(json.loads(CALIB.read_text())["wrist_image_jacobian"]["value"])
    u_per_m_at_j = float(np.linalg.norm(jac[0]))  # measured with the tips 0.099 m up
    last_mm = []
    for p in pairs:
        (ha, ba), (hb, bb) = p["a"], p["b"]
        if p["last"] and inside(ba[2]) and inside(bb[2]):
            pred = p0[0] + (ba[2] - p0[0]) * (depth + ha) / (depth + hb)
            u_per_m = u_per_m_at_j * (0.099 + depth) / (max(hb, 0.0) + depth)
            last_mm.append(1000 * (bb[2] - pred) / u_per_m)

    value = {"depth_m": round(depth, 4), "p0": [round(float(v), 4) for v in p0]}
    print(f"{len(pairs)} descents: {value}; depth from {len(ds)} width ratios "
          f"(quartiles {np.percentile(ds, [25, 75]).round(3).tolist()}), edge residual median {np.median(res_px):.1f} px")
    print(f"last descent, right edge vs model: {len(last_mm)} picks, mm {sorted(round(v, 1) for v in last_mm)}")
    if args.write:
        calib = json.loads(CALIB.read_text())
        calib["wrist_descent"] = {
            "value": value,
            "unit": "depth: metres from the camera to the jaw tips along the descent; p0: wrist image fraction (u right, v down)",
            "meaning": "a floor point's image offset from p0 scales as 1 / (depth_m + h), h = jaw tips above the floor, TCP pointing down",
            "method": f"calibrate_descent.py over {len(pairs)} logged straight descents: depth from box width ratios, p0 least squares on box edges",
            "edge_residual_median_px": round(float(np.median(res_px)), 1),
            "last_descent_right_edge_mm": [round(v, 1) for v in last_mm],
            "date": datetime.date.today().isoformat(),
        }
        CALIB.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"wrote {CALIB}")


if __name__ == "__main__":
    main()
