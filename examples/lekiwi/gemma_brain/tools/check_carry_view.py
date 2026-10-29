"""Does the arm, carried at the hover height over the reach point, show in the front camera?
One front frame folded, one at the hover; both saved next to this script. Folds at the end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain import actions, params as prm  # noqa: E402
from brain.geometry import Geometry  # noqa: E402
from brain.primitives import Skills  # noqa: E402
from common.arm import Arm  # noqa: E402
from common.robot import Robot  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "runs" / "check"  # the frames go here
HERE.mkdir(parents=True, exist_ok=True)
robot = Robot()
with robot:
    arm = Arm(robot)
    sk = Skills(robot, arm, Geometry.load())
    try:
        (HERE / "carry_folded_front.jpg").write_bytes(sk.frame("front"))
        print("hover", actions.hover_at(sk, prm.HOVER_HEIGHT_M))
        (HERE / "carry_hover_front.jpg").write_bytes(sk.frame("front"))
    finally:
        arm.reset()
print("saved", HERE / "carry_folded_front.jpg", HERE / "carry_hover_front.jpg")
