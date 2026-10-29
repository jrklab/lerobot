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

"""v2 layer 1, the primitives: the only code that drives motors and cameras (design doc
"v2 skills: three layers"). Every value comes from geometry.py (CALIB) or params.py (DEMO,
SPEC, RULE); the actions and chains above only compose these.

    move(part, ...)   arm: vertical straight-line move to a height over x, y, or by a delta,
                      optionally stopping at contact; base: a plain turn or a plain drive
    grip(open|close)  returns whether something is held
    find(obj, cam)    the object's box in a fresh frame, checked by colour; the fast model while
                      tracking an object already found, the main model otherwise or as backup
    servo(obj, cam, goal, part)  find -> goal.error -> goal.step -> move, until goal.done

Arm moves keep the gripper vertical (pitch 90 in the arm model with the homing-tick joint
zero). The Arm planner refuses joint targets outside the recorded range and paths below the
floor clearance before anything moves, and the Pi enforces z_min on every tick.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from arm import (
    FLOOR_CLEARANCE_M,
    JAW_TIP_OFFSET_M,
    JOINT_LIMIT_MARGIN,
    JOINT_RANGE,
    NAMED_POSES,
    Arm,
    PlanError,
    plan_ee_line,
)
from robot import RobotError
from vision import colour_matches, measured_angle_of, snap_box

from v2 import params as prm
from v2.geometry import Geometry
from v2.goals import Detection

VERTICAL = 90.0  # convention: the arm model's pitch for a gripper pointing straight down


@dataclass
class Result:
    ok: bool
    reason: str = ""
    data: dict = field(default_factory=dict)


def has_colour(jpeg: bytes, box, obj: str) -> bool:
    """The box holds the colour `obj` names (vision.colour_matches), and enough coloured pixels
    to tell: v1's check passes a box with too few, which let a box on bare floor through."""
    ok, why = colour_matches(jpeg, box, obj)
    return ok and not why.startswith("too few")


class Skills2:
    _last_moved: str | None = None  # "arm" or "base": what moved last, for the settle before a frame

    def __init__(self, robot, arm: Arm, geom: Geometry, log=None, main=None, fast=None):
        self.robot, self.arm, self.geom, self.log = robot, arm, geom, log
        self.main, self.fast = main, fast  # box models: the main one (26B) and the fast one (E2B)
        self.roll = 0.0
        self.seen: dict[str, Detection] = {}  # the last detection of each object, per camera
        self.heights: dict[str, float] = {}  # object heights measured side-on by the front camera
        self._roll_range = None
        # The Arm's whole-arm floor clearance from the measured floor (arm.DEFAULT_FLOOR_Z_M,
        # -0.100, predates the joint-zero fix; the measured floor is ~-0.126 at the jaw tips).
        self.arm.floor_z = float(geom.floor_coef[0]) - JAW_TIP_OFFSET_M
        self.arm.z_min = self.arm.floor_z + FLOOR_CLEARANCE_M + JAW_TIP_OFFSET_M

    def ev(self, name: str, **kw) -> None:
        if self.log is not None:
            self.log.ev(name, **kw)

    def tcp(self) -> np.ndarray:
        """The measured TCP (forward kinematics of the measured joints)."""
        return self.arm.tcp()[:3, 3]

    def cmd_tcp(self) -> np.ndarray:
        """The commanded TCP: where moves start from. The measured one hangs ~1 cm off it at
        reach (Oct 27: a land planned from it moved 9 mm sideways and tripped the contact stop)."""
        return self.arm.calib.fk(self.arm.targets())[:3, 3]

    def height(self) -> float:
        """Jaw tips above the floor under the gripper (vertical gripper: TCP above floor TCP)."""
        p = self.tcp()
        return float(p[2] - self.geom.floor(p[0], p[1]))

    # --- move ---

    def move(self, part: str = "arm", *, xy=None, height: float | None = None, delta=None,
             roll: float | None = None, until_contact: bool = False, speed: float | None = None) -> Result:
        """One plain move.

        arm: to `height` (jaw tips above the floor) over TCP `xy` (default: where it is), or by
        `delta` (dx, dy, dz in metres); `roll` turns the wrist; until_contact: the Pi stops the
        move where the TCP meets something (data["contact_z"]), searching at most
        LAND_SEARCH_M past the goal. Refused, with nothing sent, out of reach or too low.
        base: `delta` = (dx, dy) in metres or `roll` = a turn in degrees, never both: a
        combined turn and strafe left the body unturned (Oct 27).
        """
        if part == "base":
            return self._move_base(delta, roll)
        p = self.cmd_tcp()
        roll = self.roll if roll is None else roll
        if delta is not None:
            goal = p + np.asarray(delta, float)
        else:
            x, y = (p[0], p[1]) if xy is None else xy
            goal = np.array([x, y, self.geom.floor(x, y) + (p[2] - self.geom.floor(p[0], p[1]) if height is None else height)])
        floor = self.geom.floor(goal[0], goal[1])
        # The Pi's floor guard applies to the jaw tips, JAW_TIP_OFFSET_M below the TCP: a move whose
        # tips start or end within the free-move clearance gets its own guard just under it (a lift
        # from the floor was refused by the default one, which sits above its start: Oct 27 teach).
        clear = FLOOR_CLEARANCE_M + JAW_TIP_OFFSET_M
        h_from, h_to = p[2] - self.geom.floor(p[0], p[1]), goal[2] - floor
        # On the line counts as low: the split below ends a move there, and the free-move guard
        # sits above the jaw tips at the line (Oct 28: all 5 descents refused by the Pi).
        low = min(h_from, h_to) < clear or bool(np.isclose(min(h_from, h_to), clear))
        if low and not until_contact and speed is None and max(h_from, h_to) > clear:
            # Only the part below the clearance at the near speed: a 13 cm lift from the floor
            # took 13 s, a 10 -> 3 cm descent 7 s, all at 1 cm/s (Oct 28 trials).
            mid = p + (goal - p) * (clear - h_from) / (h_to - h_from)
            down = h_to < h_from
            r = self.move("arm", delta=tuple(mid - p), roll=roll,
                          speed=prm.FREE_SPEED_MPS.value if down else prm.NEAR_SPEED_MPS.value)
            if not r.ok:
                return r
            return self.move("arm", delta=tuple(goal - self.cmd_tcp()), roll=roll,
                             speed=prm.NEAR_SPEED_MPS.value if down else prm.FREE_SPEED_MPS.value)
        speed = speed or (prm.NEAR_SPEED_MPS.value if low else prm.FREE_SPEED_MPS.value)
        search = prm.LAND_SEARCH_M.value if until_contact else 0.0
        # Low and contact-stop moves get a guard just under them; the Pi checks the jaw tips, so
        # their offset too. A contact-stop move without one got the Pi's built-in default, which
        # predates the joint-zero fix (-0.050, the floor is ~-0.126): refused (Oct 27 place-in).
        z_min = (min(goal[2], p[2]) - search - JAW_TIP_OFFSET_M - prm.LOW_MOVE_GUARD_M.value) if low or until_contact else None
        try:
            if until_contact:
                wps, durs = plan_ee_line(self.arm.calib, self.arm.targets(), goal.tolist(), VERTICAL, roll, speed,
                                         z_min=goal[2] - search)
                res = self.robot.trajectory(wps, durs, z_min=z_min, contact_stop=True).get("result", {})
            else:
                self.arm.move_to(goal.tolist(), pitch_deg=VERTICAL, roll_deg=roll, speed=speed, z_min=z_min)
                res = {}
        except PlanError as e:
            return Result(False, f"can't move there: {e}", {"goal": goal.round(4).tolist()})
        except RobotError as e:  # refused or stopped on the Pi: a failed step, not a crash
            return Result(False, f"the robot refused or stopped the move: {e}", {"goal": goal.round(4).tolist()})
        self.roll = roll
        self._last_moved = "arm"
        data = {"goal": goal.round(4).tolist(), "floor": round(floor, 4), "contact": False}
        if res.get("contact"):
            data.update(contact=True, contact_z=float(res["z_meas"]), detector=res.get("how"))
        self.ev("move", part="arm", **data)
        return Result(True, data=data)

    def _move_base(self, delta, turn) -> Result:
        if (delta is None) == (turn is None):
            return Result(False, "base: give a drive (dx, dy) or a turn, one at a time")
        # The Pi ends a base move once within its tolerance of the goal, so a move asks for that
        # much more, in the same direction, to end where it was meant to.
        if turn is not None:
            turn = float(turn)
            res = self.robot.base(dtheta_deg=turn + math.copysign(prm.BASE_TURN_STOP_SHORT_DEG.value, turn))
        else:
            d = np.asarray(delta, float)
            n = float(np.linalg.norm(d))
            d = d * (n + prm.BASE_STOP_SHORT_M.value) / n if n > 0 else d
            res = self.robot.base(dx=float(d[0]), dy=float(d[1]))
        self._last_moved = "base"
        moved = res.get("result", {}).get("moved")
        self.ev("move", part="base", delta=None if delta is None else list(delta), turn=turn, moved=moved)
        return Result(True, data={"moved": moved})

    @property
    def roll_range(self) -> tuple[float, float]:
        """(min, max) wrist roll keeping the roll joint within ROLL_FROM_REST_DEG of its rest
        angle and inside its recorded range (owner's rule, Oct 27), from the calibration."""
        if self._roll_range is None:
            cal, joint = self.arm.calib, "arm_wrist_roll"

            def deg(norm):
                return math.degrees(cal.to_rad(joint, norm))

            def joint_at(roll):
                x = prm.REACH_POINT_X_M.value  # where the roll is used: over the reach point
                wps, _ = plan_ee_line(cal, dict(NAMED_POSES["ready"]), [x, 0.0, self.geom.floor(x, 0.0) + prm.HOVER_HEIGHT_M.value],
                                      VERTICAL, roll, prm.FREE_SPEED_MPS.value, -1.0)
                return deg(wps[-1][joint])

            j0 = joint_at(0.0)
            toward_less = joint_at(-90.0) < j0  # convention: a quarter turn shows which way the joint goes

            def roll_for(target):
                """The roll whose joint angle is `target`, found by halving along the planner's own
                IK: the joint isn't linear in the roll (a two-point slope put a -160 deg roll at
                98.5 joint units, the planner at 100.5: refused, Oct 28 full task). It returns
                the inner side, a roll the planner accepts."""
                inner, outer = 0.0, -180.0 if (target < j0) == toward_less else 180.0  # convention: half a turn either way
                for _ in range(12):  # convention: halvings: 180 deg / 2^12, under 0.05 deg
                    mid = (inner + outer) / 2
                    try:
                        reached = (joint_at(mid) - target) * (target - j0) >= 0
                    except PlanError:  # past the joint's limit
                        reached = True
                    inner, outer = (inner, mid) if reached else (mid, outer)
                return inner

            rest = deg(NAMED_POSES["rest"][joint])
            # The ends the planner accepts: its joint range, JOINT_LIMIT_MARGIN inside (Oct 28: the
            # full -100..100 ends made a roll clamped to the range's end a refused move).
            jlo, jhi = JOINT_RANGE[joint]
            lo, hi = sorted((deg(jlo + JOINT_LIMIT_MARGIN), deg(jhi - JOINT_LIMIT_MARGIN)))
            span = prm.ROLL_FROM_REST_DEG.value
            jmin, jmax = max(rest - span, lo), min(rest + span, hi)
            self._roll_range = tuple(sorted((roll_for(jmin), roll_for(jmax))))
        return self._roll_range

    def _settle(self) -> None:
        """Wait before a frame until the arm has stopped, at most LOOK_SETTLE_S: stopped when the
        measured TCP moved less than one encoder tick's worth between two reads. After a base
        move the full wait: nothing measures the body's coasting."""
        end = time.monotonic() + prm.LOOK_SETTLE_S.value
        t0, waited = time.monotonic(), "full"
        if self._last_moved == "arm":
            tick = 2 * math.pi / self.arm.calib.resolution  # one encoder step, rad
            prev = self.tcp()
            while time.monotonic() < end:
                cur = self.tcp()
                if np.linalg.norm(cur - prev) <= tick * np.linalg.norm(cur):  # the tick's TCP motion at this reach
                    waited = "still"
                    break
                prev = cur
        if waited == "full":
            time.sleep(max(end - time.monotonic(), 0.0))
        self.ev("settle", after=self._last_moved, how=waited, s=round(time.monotonic() - t0, 3))

    # --- find ---

    def frame(self, cam: str, tag: str = "look") -> bytes | None:
        """A frame captured after this call, or None. A failed request re-measures the Mac-Pi
        clock offset and tries once more: the Pi has no time source and its clock jumped ~36 s
        mid-run (Oct 27 trials), so "a frame after now" asked for one from its future (HTTP 504)."""
        self._settle()
        try:
            jpeg, _ = self.robot.frame(cam, after=time.time())
        except RobotError as e:
            self.ev("frame_retry", cam=cam, error=str(e))
            self.robot._clock_offset = None  # the next request measures it again
            try:
                jpeg, _ = self.robot.frame(cam, after=time.time())
            except RobotError as e2:
                self.ev("frame_failed", cam=cam, error=str(e2))
                return None
        if self.log is not None:
            self.log.frame(f"{cam}_{tag}", jpeg)
        return jpeg

    def find(self, obj: str, cam: str = "wrist", angle: bool = False, fast_ok: bool = True) -> Detection | None:
        """The object's box in a fresh `cam` frame, or None. Tracking an object already seen
        in this camera, the fast model answers first; its box must have the object's colour
        (E2B boxes something even when nothing is there), else the main model is asked. The
        wrist box is snapped to the object's colour blob. angle: also its long-side angle."""
        jpeg = self.frame(cam)
        if jpeg is None:
            return None
        prompt = f"the {obj} on the floor"
        tracked = self.seen.get(f"{cam}:{obj}")
        a, by = None, ""
        if tracked is not None and self.fast is not None and fast_ok:
            a = self.fast.box(jpeg, prompt)
            by = "fast"
            if a.value is None or not has_colour(jpeg, a.value, obj):
                self.ev("fast_miss", obj=obj, cam=cam, box=a.value)
                a = None
        if a is None:
            if self.main is None:
                return None
            a, by = self.main.box(jpeg, prompt), "main"
        if a.value is None or not has_colour(jpeg, a.value, obj):
            self.ev("find", obj=obj, cam=cam, found=False, box=a.value, by=by)
            return None
        if cam == "front" and a.value[3] < self.geom.front["range_horizon_v"]:
            # Nothing on the floor ends above the horizon: the held object's tips show there
            # when carried at the hover (Oct 28 check).
            self.ev("find", obj=obj, cam=cam, found=False, box=a.value, by=by, why="above the horizon")
            return None
        box = snap_box(jpeg, a.value, what=obj)[0] if cam == "wrist" else list(a.value)
        det = Detection(box=[round(float(v), 4) for v in box], jpeg=jpeg, by=by)
        if angle:
            det.angle_deg, det.angle_how = measured_angle_of(jpeg, box, obj)
        self.seen[f"{cam}:{obj}"] = det
        self.ev("find", obj=obj, cam=cam, found=True, box=det.box, by=by, angle=det.angle_deg)
        return det

    # --- servo ---


    def servo(self, obj: str, goal, part: str | None = None) -> Result:
        """Move `part` (default the goal's) until `goal` is met for `obj` in the goal's camera:
        find, measure the goal's error, stop when it's within tolerance, else one capped move."""
        part = part or goal.part
        angle = goal.__class__.__name__ == "AngleGoal"
        err = None
        for k in range(int(prm.SERVO_MAX_LOOKS.value)):
            det = self.find(obj, goal.cam, angle=angle)
            if det is None:
                return Result(False, f"lost the {obj} in the {goal.cam} camera", {"looks": k})
            if angle and det.angle_deg is None:
                return Result(False, f"the {obj}'s angle couldn't be measured", {"looks": k + 1, "box": det.box})
            err = goal.error(det, self)
            done = goal.done(err, self)
            # Commanded and measured TCP at the look: at 3 cm the first move overshot ~3 mm and the
            # reversal after it barely moved (Oct 28), hysteresis the encoders may or may not see.
            self.ev("servo", obj=obj, goal=goal.__class__.__name__, k=k, err=[round(float(e), 4) for e in err], done=done,
                    tcp_cmd=[round(float(v), 4) for v in self.cmd_tcp()], tcp_meas=[round(float(v), 4) for v in self.tcp()])
            if done:
                return Result(True, data={"looks": k + 1, "err": err.tolist()})
            step = goal.step(err, self)
            if part == "roll" or ("roll" in step and part == "arm"):
                r = self.move("arm", delta=(0.0, 0.0, 0.0), roll=step["roll"])
            elif part == "base":
                r = self.move("base", delta=step.get("delta"), roll=step.get("roll"))
            else:
                d = step["delta"]
                r = self.move("arm", delta=(d[0], d[1], 0.0))
            if not r.ok:
                return Result(False, f"servo move refused: {r.reason}", {"looks": k + 1})
        return Result(False, f"{goal.__class__.__name__} not met after {int(prm.SERVO_MAX_LOOKS.value)} looks",
                      {"err": None if err is None else err.tolist()})

    # --- grip ---

    def grip(self, how: str, width: float | None = None) -> Result:
        """open: to `width` (default the demo's OPEN_WIDTH). close: until the jaws stall;
        data["held"]: they stopped short of closed-empty and the Pi's hold watch saw no drop."""
        if how == "open":
            res = self.robot.gripper(prm.OPEN_WIDTH.value if width is None else width).get("result", {})
            return Result(True, data=res)
        if how != "close":
            return Result(False, f"grip: 'open' or 'close', not {how!r}")
        res = self.robot.gripper(0, speed=prm.CLOSE_SPEED.value).get("result", {})
        held = self.held()
        self.ev("grip", how="close", held=held, **res)
        if not held:
            return Result(False, f"closed to {res.get('pos')} without holding anything", {**res, "held": False})
        return Result(True, data={**res, "held": True})

    def held(self) -> bool:
        """The Pi's hold watch has the jaws closed on something (state "holding"; it clears when
        they open and says "dropped" when they close further), and they stopped short of
        closed-empty. The motor reading alone isn't enough: open jaws read high too (Oct 27
        place-in said "still holding" after opening, with the brick in the cup)."""
        st = self.robot.status()
        hold = st.get("gripper_hold") or {}
        return hold.get("state") == "holding" and st["joints"]["arm_gripper"] >= prm.GRIPPER_EMPTY_MAX.value
