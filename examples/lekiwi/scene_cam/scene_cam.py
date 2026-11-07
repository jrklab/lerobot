"""The scene camera: a USB camera on the Mac that watches the robot for Claude (not used by the
robot). Saves its latest frame to scene_cam/latest.jpg once a second. Run it from Terminal, which
has the camera permission (Claude Code's background sessions can't get it):

    cd ~/Work/lerobot && uv run python examples/lekiwi/scene_cam/scene_cam.py
"""

import os
import sys
import time
from pathlib import Path

import cv2

OUT = Path(__file__).with_name("latest.jpg")
PERIOD_S = 1.0


def open_camera(index: int | None) -> cv2.VideoCapture:
    """The given index, else the first camera that isn't the FaceTime one (index 0 on a MacBook)."""
    for i in [index] if index is not None else [1, 2, 0]:
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened() and cap.read()[0]:
            print(f"scene camera: index {i}")
            return cap
        cap.release()
    sys.exit("no camera opened: check System Settings > Privacy & Security > Camera for Terminal")


def main() -> None:
    cap = open_camera(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    tmp = OUT.with_suffix(".tmp.jpg")
    while True:
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(tmp), frame)
            os.replace(tmp, OUT)  # whole frames only, for a reader at any moment
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
