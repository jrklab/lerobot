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

"""Step 4: the parameters of a pick (and place) from one teleoperated demo, next to v1's
hand-set constants. Offline: nothing on the robot changes.

    .venv/bin/python calibration/demo_extract.py demos/<dataset> --episode 0 --target "red block" [--place-target cup]

The dataset is the web app's recording (LeRobot v3 layout, copied from the Pi's
~/.cache/huggingface/lerobot/<repo_id>). Keyframes come from the joint states alone: the
close (gripper falling, then stalling short of empty), the pause before it, the hover above
it, the lift after it, and the release (gripper opening). Heights come from forward
kinematics with the robot's calibration (demos/calibration.json, from /auto/calibration).
The wrist frames at the pre-close and pre-release keyframes are boxed by the model (26B on
--vision-url) for where the object sits in the image. Writes <dataset>/extract-ep<N>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # gemma_brain: common/, brain/, data/, runs/
sys.path.insert(0, str(ROOT))

from common.arm import Calibration

JOINTS = ["arm_shoulder_pan", "arm_shoulder_lift", "arm_elbow_flex", "arm_wrist_flex", "arm_wrist_roll", "arm_gripper"]
# Segmentation, in the recording's own units (normalized joints at 30 Hz): generic, not
# robot-specific -- they say what counts as "still" and as "closing" in a human demo.
STILL_UNITS = 0.4  # max joint change per frame for the arm to count as paused
PAUSE_FRAMES = 6  # stillness at least this long (0.2 s) splits motions
# A deliberate pause (the procedure asks for ~1 s) marks the hover and lift keyframes: a slow
# human descent or lift has many 0.2-0.6 s still moments (Oct 27 demo).
KEY_PAUSE_FRAMES = 24
CLOSE_DROP = 8.0  # the gripper falls at least this much in one close
GRIPPER_EMPTY = 3.5  # v2 params.GRIPPER_EMPTY_MAX: closing below this held nothing
OPEN_RISE = 3.0  # the gripper opening this far above its closed width is a release
LIFT_MIN_M = 0.02  # the lift keyframe: the first pause at least this far above the close


@dataclass
class Keyframes:
    close_start: int  # the last frame before the gripper starts closing
    close_end: int  # the gripper stops (stalled on the object)
    pre_close: int  # the last frame of the pause before the close
    hover: int | None  # the last deliberate pause before the close that is clearly above it
    lift: int | None  # the highest deliberate pause between the close and the release
    pre_release: int | None  # the last frame before the gripper opens again
    held: bool


def pauses(joints: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame ranges, end inclusive, where no arm joint moves more than
    STILL_UNITS per frame for at least PAUSE_FRAMES frames."""
    still = np.r_[True, np.abs(np.diff(joints[:, :5], axis=0)).max(axis=1) <= STILL_UNITS]
    out, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= PAUSE_FRAMES:
                out.append((start, i - 1))
            start = None
    return out


def find_close(g: np.ndarray) -> tuple[int, int]:
    """(start, end) of the first close: the gripper falls at least CLOSE_DROP, end = where it
    stops falling (stalled on the object, or empty). A human closes unevenly, so a brief
    flat spot followed by more fall within 5 frames still counts as the same close."""
    i = 1
    while i < len(g):
        if g[i] >= g[i - 1] - 0.2:
            i += 1
            continue
        j = i
        while j + 1 < len(g):
            ahead = g[j + 1 : j + 6]
            if g[j + 1] < g[j] - 0.05 or ahead.min() < g[j] - 0.5:
                j += 1
            else:
                break
        if g[i - 1] - g[j] >= CLOSE_DROP:
            return i - 1, j
        i = j + 1
    raise ValueError("no gripper close in this episode")


def find_keyframes(joints: np.ndarray, z: np.ndarray) -> Keyframes:
    """Keyframes of the first grasp in an episode. joints: (T, 6) normalized, gripper last;
    z: (T,) TCP height (m)."""
    g = joints[:, 5]
    close_start, close_end = find_close(g)
    width = g[close_end]
    ps = pauses(joints)
    # The arm stays still while the gripper closes, so the pause before the close runs on into
    # it: clip each pause to the frame before the gripper starts moving.
    before = [(p0, min(p1, close_start)) for p0, p1 in ps if p0 < close_start]
    pre_close = before[-1][1] if before else close_start
    hover = next((p1 for p0, p1 in reversed(before)
                  if p1 - p0 + 1 >= KEY_PAUSE_FRAMES and z[p1] - z[close_end] >= LIFT_MIN_M), None)
    rel = np.nonzero(g[close_end:] > width + OPEN_RISE)[0]
    pre_release = None
    if len(rel):
        r = close_end + int(rel[0])
        while r > close_end and g[r - 1] > width + 0.3:  # back to where the opening began
            r -= 1
        pre_release = r - 1
    end = pre_release if pre_release is not None else len(g) - 1
    # The top of the lift: the highest deliberate pause between the close and the release (a
    # demo can pause partway up first: Oct 27, 1 s at 2.7 cm before 9.9 cm).
    # The pause the release comes out of isn't the lift (the target can sit higher than it).
    ups = [p0 for p0, p1 in ps if close_end < p0 <= end and p1 - p0 + 1 >= KEY_PAUSE_FRAMES
           and z[p0] - z[close_end] >= LIFT_MIN_M and not (pre_release is not None and p0 <= pre_release <= p1)]
    lift = max(ups, key=lambda f: z[f]) if ups else None
    return Keyframes(close_start, close_end, pre_close, hover, lift, pre_release, held=width > GRIPPER_EMPTY)


def load_episode(root: Path, episode: int) -> pd.DataFrame:
    df = pd.read_parquet(root / "data" / "chunk-000" / f"file-{episode:03d}.parquet")
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["observation.state"]["names"]
    state = np.stack(df["observation.state"].values)
    for k, n in enumerate(names):
        df[n] = state[:, k]
    return df


def wrist_frame(root: Path, df: pd.DataFrame, episode: int, row: int) -> tuple[bytes, int] | None:
    """JPEG of the wrist camera at `row` (or the nearest earlier row that has a frame), via
    the recorder's video_frame column; without it, assumes one video frame per row."""
    col = "observation.images.wrist.video_frame"
    if col in df:
        frames = df[col].values[: row + 1]
        hits = np.nonzero(frames >= 0)[0]
        if not len(hits):
            return None
        k = int(frames[hits[-1]])
    else:
        print("warning: no video_frame column; assuming one wrist frame per row")
        k = row
    cap = cv2.VideoCapture(str(root / "videos" / "observation.images.wrist" / "chunk-000" / f"file-{episode:03d}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, bgr = cap.read()
    if not ok:
        return None
    return cv2.imencode(".jpg", bgr)[1].tobytes(), k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--target", required=True, help="what was picked, e.g. 'red block'")
    ap.add_argument("--place-target", default=None, help="where it was put, e.g. 'cup'")
    ap.add_argument("--calibration", type=Path, default=ROOT / "demos" / "calibration.json")
    ap.add_argument("--vision-url", default="http://127.0.0.1:8081")
    args = ap.parse_args()

    df = load_episode(args.dataset, args.episode)
    c = json.loads(args.calibration.read_text())
    calib = Calibration(c["joints"], c.get("motor_resolution", 4096))
    joints = df[[f"{j}.pos" for j in JOINTS]].values
    tcp = np.stack([calib.fk(dict(zip(JOINTS, row, strict=True))) for row in joints])
    z = tcp[:, 2, 3]
    kf = find_keyframes(joints, z)

    from brain import params as prm  # the values the v2 skills use now, for the comparison
    from common.gemma import LlamaCpp
    from common.vision import brick_angle, snap_box

    # Next to the other committed demo data, where brain/params.py reads it (the raw recording
    # under demos/ is git-ignored).
    out_dir = ROOT / "data" / "demos" / args.dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"episode": args.episode, "frames": len(df), "keyframes": kf.__dict__}
    g = joints[:, 5]
    out["open_width"] = float(np.median(g[max(kf.close_start - PAUSE_FRAMES, 0) : kf.close_start + 1]))
    out["closed_width"] = float(g[kf.close_end])
    out["close_z_m"] = float(z[kf.close_end])
    out["floor_z_m_est"] = float(z[kf.close_end] - prm.GRASP_LIFT_M.value)  # the demo closes near the floor
    out["descend_m"] = float(z[kf.hover] - z[kf.close_end]) if kf.hover is not None else None
    out["lift_m"] = float(z[kf.lift] - z[kf.close_end]) if kf.lift is not None else None
    out["roll_deg_at_close"] = math.degrees(calib.to_rad("arm_wrist_roll", joints[kf.close_end, 4]))
    base = np.abs(df[["x.vel", "y.vel"]].values).max(axis=1) > 0.005
    out["base_moved_between_close_and_release"] = bool(base[kf.close_end : kf.pre_release or len(df)].any())
    if kf.pre_release is not None:
        out["release_z_above_floor_m"] = float(z[kf.pre_release] - out["floor_z_m_est"])

    model = LlamaCpp(args.vision_url)
    got = wrist_frame(args.dataset, df, args.episode, kf.pre_close)
    if got:
        jpeg, k = got
        (out_dir / f"ep{args.episode}_pre_close.jpg").write_bytes(jpeg)
        a = model.box(jpeg, f"the {args.target} on the floor")
        if a.value is not None:
            box, how = snap_box(jpeg, a.value, what=args.target)
            ang = brick_angle(jpeg, box)
            out["pre_close"] = {"video_frame": k, "box": [round(v, 4) for v in box], "snap": how,
                                "centre": [round((box[0] + box[2]) / 2, 4), round((box[1] + box[3]) / 2, 4)]}
    if kf.hover is not None:
        got = wrist_frame(args.dataset, df, args.episode, kf.hover)
        if got:
            jpeg, k = got
            (out_dir / f"ep{args.episode}_hover.jpg").write_bytes(jpeg)
            a = model.box(jpeg, f"the {args.target} on the floor")
            if a.value is not None:
                box, _ = snap_box(jpeg, a.value, what=args.target)
                ang = brick_angle(jpeg, box)
                roll = math.degrees(calib.to_rad("arm_wrist_roll", joints[kf.hover, 4]))
                out["hover_view"] = {"video_frame": k, "box": [round(v, 4) for v in box],
                                     "brick_angle_deg": None if ang is None else round(ang["angle_deg"], 1), "aspect": None if ang is None else round(ang["aspect"], 2), "roll_deg": round(roll, 1)}
                if ang:
                    # The wrist turns between the hover and the close by this much, on top of
                    # aligning to the brick.
                    out["roll_change_hover_to_close_deg"] = round(out["roll_deg_at_close"] - roll, 1)
    if kf.pre_release is not None and args.place_target:
        # Right before the release the cup fills the wrist view (the model finds no "cup"):
        # use the last deliberate pause above it, before lowering in.
        from calibration.demo_extract import KEY_PAUSE_FRAMES as KP  # noqa: E402
        above = [p1 for p0, p1 in pauses(joints) if kf.close_end < p1 < kf.pre_release and p1 - p0 + 1 >= KP
                 and z[p1] - z[kf.pre_release] >= LIFT_MIN_M and p1 != kf.lift and p0 != kf.lift]
        view = above[-1] if above else kf.pre_release
        out["above_target_z_above_floor_m"] = round(float(z[view] - out["floor_z_m_est"]), 3)
        got = wrist_frame(args.dataset, df, args.episode, view)
        if got:
            jpeg, k = got
            (out_dir / f"ep{args.episode}_above_target.jpg").write_bytes(jpeg)
            a = model.box(jpeg, f"the {args.place_target}")
            if a.value is not None:
                b = a.value
                out["pre_release"] = {"video_frame": k, "box": [round(v, 4) for v in b],
                                      "centre": [round((b[0] + b[2]) / 2, 4), round((b[1] + b[3]) / 2, 4)]}

    rows = [
        ("open width before the close", f"{out['open_width']:.1f}", f"OPEN_WIDTH {prm.OPEN_WIDTH.value:.1f}"),
        ("jaw width at the close", f"{out['closed_width']:.1f}", f"empty below {prm.GRIPPER_EMPTY_MAX.value}; 2x4 seen 6.7-7.0"),
        ("hover -> close descend", _m(out["descend_m"]),
         f"hover {prm.HOVER_HEIGHT_M.value} m, last look {prm.FINAL_LOOK_HEIGHT_M.value} m above the floor"),
        ("lift after the close", _m(out["lift_m"]), f"LIFT_M {prm.LIFT_M.value:.3f}"),
        ("release height above floor", _m(out.get("release_z_above_floor_m")),
         f"RELEASE_ABOVE_FLOOR_M {prm.RELEASE_ABOVE_FLOOR_M.value:.3f}"),
    ]
    print(f"\nepisode {args.episode}: {len(df)} frames, keyframes {kf}")
    print(f"{'quantity':45s} {'demo':>16s}   v2 now")
    for name, demo, v1 in rows:
        print(f"{name:45s} {demo:>16s}   {v1}")
    path = out_dir / f"extract-ep{args.episode}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwritten {path}")


def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f} m"


if __name__ == "__main__":
    main()
