"""How the wrist camera's brightness setting changes what the brain sees: the data for an automatic
brightness (Nov 11: the owner found the brightness critical for a pick, and set it by hand for
daylight and for indoor light; on Sunday Nov 8 daylight blew a tennis ball's top out to white and
it was read as a 27-36 mm "brick").

The arm goes to the pick's hover over the reach point (10 cm up, the gripper straight down: where
the wrist camera looks for an object before a grasp), then for each brightness setting the wrist
camera's frames are measured and the main model (26B) asked for each object, as the pick asks:

    image     mean level, share of clipped (white) and of black pixels, over the frame and inside
              each object's box
    detection found or not, the box, and the outline's aspect as the grasp plan measures it
              (vision.brick_angle on the model's box with its sampled colour: round under 1.4)

The camera's own auto exposure stays on; only its brightness bias steps through --values. The
setting it had is put back at the end, and the arm folds.

Put the objects on the floor under the gripper first (they show in the wrist camera at the hover),
note the light, and run (the robot must be in a state to move the arm):

    .venv/bin/python tools/wrist_brightness_sweep.py --light day "tennis ball" "red brick"

Results: runs/brightness/<time>-<light>/: the frames, sweep.jsonl (one row per frame and object),
and a summary printed per setting.
"""

import argparse
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain import actions, params as prm  # noqa: E402
from brain.geometry import Geometry  # noqa: E402
from brain.primitives import Skills  # noqa: E402
from common.arm import Arm  # noqa: E402
from common.gemma import LlamaCpp  # noqa: E402
from common.robot import Robot  # noqa: E402
from common.vision import ELONGATED_ASPECT, brick_angle, sample_look  # noqa: E402

PLANNER_URL = "http://127.0.0.1:8081"  # the main model, as plan_brain.py
SETTLE_S = 2.0  # after a brightness change, for the camera's own auto exposure to settle
CLIP = 250  # a channel at or above this is clipped (white)
BLACK = 15  # a pixel this dark or darker in every channel is black


def controls(robot: Robot, values: dict | None = None) -> dict:
    """The wrist camera's controls (GET), or set some (POST): the web server's camera endpoint."""
    url = robot.url + "/camera/wrist/controls"
    data = None if values is None else json.dumps(values).encode()
    req = urllib.request.Request(
        url, data=data, method="GET" if data is None else "POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        reply = json.load(r)
    if not reply.get("ok"):
        raise RuntimeError(f"wrist camera controls: {reply.get('reason')}")
    return reply["controls"]


def stats(img: np.ndarray, box=None) -> dict:
    """Mean grey level, clipped and black shares, over the frame or inside a box (fractions)."""
    if box is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(box, (w, h, w, h), strict=True))
        img = img[max(0, y0) : max(y0 + 1, y1), max(0, x0) : max(x0 + 1, x1)]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "mean": round(float(grey.mean()), 1),
        "clipped": round(float((img.max(axis=2) >= CLIP).mean()), 4),
        "black": round(float((img.max(axis=2) <= BLACK).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "objects", nargs="+", help='what is under the gripper, as a task names it, e.g. "tennis ball"'
    )
    ap.add_argument("--light", required=True, help="a label for the light: day, indoor, evening ...")
    ap.add_argument(
        "--values",
        type=int,
        nargs="+",
        default=[-64, -48, -32, -16, 0, 16, 32, 48, 64],
        help="the brightness settings to step through",
    )
    ap.add_argument("--frames", type=int, default=3, help="frames measured per setting")
    ap.add_argument("--robot", default=None)
    ap.add_argument(
        "--no-move", action="store_true", help="the arm is already where the wrist camera should look"
    )
    args = ap.parse_args()

    out = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "brightness"
        / (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"-{args.light}")
    )
    out.mkdir(parents=True, exist_ok=True)
    main_model = LlamaCpp(PLANNER_URL)
    robot = Robot(args.robot)
    before = {k: v["value"] for k, v in controls(robot).items()}
    print(f"wrist camera before: {before}\nsaving to {out}")
    rows = []
    with robot:
        arm = Arm(robot)
        sk = Skills(robot, arm, Geometry.load(), None, main=main_model)
        try:
            if not args.no_move:
                r = actions.hover_at(sk, prm.HOVER_HEIGHT_M)
                if not r.ok:
                    sys.exit(f"couldn't go to the hover: {r.reason}")
            for value in args.values:
                controls(robot, {"brightness": value})
                time.sleep(SETTLE_S)
                for k in range(args.frames):
                    jpeg = sk.frame("wrist")
                    name = f"b{value:+03d}_{k}.jpg"
                    (out / name).write_bytes(jpeg)
                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    frame = stats(img)
                    for obj in args.objects:
                        a = main_model.box(jpeg, f"the {obj} on the floor")
                        row = {
                            "light": args.light,
                            "brightness": value,
                            "frame": name,
                            "obj": obj,
                            "image": frame,
                            "found": a.value is not None,
                        }
                        if a.value is not None:
                            box = [round(float(v), 4) for v in a.value]
                            m = brick_angle(jpeg, box, look=sample_look(jpeg, box))
                            row.update(
                                box=box,
                                inside=stats(img, box),
                                aspect=None if m is None else m["aspect"],
                                round=None if m is None else m["aspect"] < ELONGATED_ASPECT,
                            )
                        rows.append(row)
                        with open(out / "sweep.jsonl", "a") as f:
                            f.write(json.dumps(row) + "\n")
                print(
                    f"brightness {value:+3d}: frame mean {frame['mean']:5.1f}, clipped {100 * frame['clipped']:4.1f}%"
                )
        finally:
            controls(robot, {"brightness": before["brightness"]})
            print(f"wrist camera brightness back to {before['brightness']}")
            if not args.no_move:
                arm.reset()

    print(
        f"\nper setting and object: found, aspect (round under {ELONGATED_ASPECT:.1f}), clipped inside its box"
    )
    for value in args.values:
        for obj in args.objects:
            rs = [r for r in rows if r["brightness"] == value and r["obj"] == obj]
            found = [r for r in rs if r["found"]]
            aspects = [r["aspect"] for r in found if r.get("aspect") is not None]
            clip = [r["inside"]["clipped"] for r in found]
            print(
                f"  {value:+3d} {obj:14s} found {len(found)}/{len(rs)}"
                + (f", aspect {min(aspects):.2f}-{max(aspects):.2f}" if aspects else "")
                + (f", clipped {100 * max(clip):.0f}%" if clip else "")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
