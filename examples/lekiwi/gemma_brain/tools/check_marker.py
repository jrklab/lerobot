"""Where the front camera measures the container markers (brain/markers.py), to check against a
tape measure before the place relies on them. Reads frames only; nothing moves.

    .venv/bin/python tools/check_marker.py            # one frame
    .venv/bin/python tools/check_marker.py --n 10     # ten, with their spread

For each marker seen: its middle in the arm frame (x forward, y left, z up; floor ~-0.10), its
distance from the camera, which way its face turns, its middle's height above the floor and
the height of the top of the face it's on (the container's).
The frame is saved next to the run logs with the markers outlined."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.cameras import IMAGE_PX  # noqa: E402
from brain.geometry import Geometry  # noqa: E402
from brain.markers import face_top_z, find_markers  # noqa: E402
from common.robot import Robot  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "check"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1, help="frames to measure")
args = ap.parse_args()

geom = Geometry.load()
robot = Robot()
seen: dict[int, list[np.ndarray]] = {}
img = None
for _ in range(args.n):
    jpeg, _ = robot.frame("front", after=time.time())
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    for m in find_markers(img, geom.cams.front, geom.cams.T_base_cam):
        top = face_top_z(img, m, geom.cams.front, geom.cams.T_base_cam)
        top = np.nan if top is None else top - geom.floor_seen_z
        seen.setdefault(m.id, []).append(np.r_[m.centre, m.distance_m, m.heading_deg(), top])
        cv2.polylines(img, [(m.corners * IMAGE_PX).astype(np.int32)], True, (0, 0, 255), 2)
if img is not None:
    cv2.imwrite(str(OUT / "marker_front.jpg"), img)
if not seen:
    print("no container marker in the front camera")
for mid, rows in sorted(seen.items()):
    a = np.array(rows)
    x, y, z, d, hd, top = a.mean(axis=0)
    spread = a.std(axis=0) if len(a) > 1 else np.zeros(6)
    print(f"marker {mid}: seen {len(a)}/{args.n}  middle x {x:.3f} y {y:.3f} m (arm frame), "
          f"{100 * (z - geom.floor_seen_z):.1f} cm above the floor; {100 * d:.1f} cm from the camera; "
          f"face turned {hd:+.1f} deg; face top {100 * top:.1f} cm above the floor" + (f"  (spread: {1000 * spread[:3].max():.1f} mm, {spread[4]:.1f} deg)" if len(a) > 1 else ""))
print("frame:", OUT / "marker_front.jpg")
