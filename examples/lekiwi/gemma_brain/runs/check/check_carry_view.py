"""Does the arm, carried at the hover height over the reach point, show in the front camera?
One front frame folded, one at the hover; both saved next to this script. Folds at the end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arm import Arm  # noqa: E402
from robot import Robot  # noqa: E402
from v2 import actions, params as prm  # noqa: E402
from v2.geometry import Geometry  # noqa: E402
from v2.skills import Skills2  # noqa: E402

HERE = Path(__file__).resolve().parent
robot = Robot()
with robot:
    arm = Arm(robot)
    sk = Skills2(robot, arm, Geometry.load())
    try:
        (HERE / "carry_folded_front.jpg").write_bytes(sk.frame("front"))
        print("hover", actions.hover_at(sk, prm.HOVER_HEIGHT_M))
        (HERE / "carry_hover_front.jpg").write_bytes(sk.frame("front"))
    finally:
        arm.reset()
print("saved", HERE / "carry_folded_front.jpg", HERE / "carry_hover_front.jpg")
