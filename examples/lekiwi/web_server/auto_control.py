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

"""
"auto" control mode: executes motion primitives that an off-board planner (the Mac-side
Gemma brain, examples/lekiwi/gemma_brain/) submits over HTTP, one at a time.

The split: the planner does the heavy compute (IK, path planning, perception) and sends
joint-space trajectories -- normalized joint targets with a duration per segment -- and
relative base moves. This module only validates and executes them from RobotBridge's 30Hz
control loop (step()): trajectory playback, a gripper close with stall detection, and a
relative base move closed on wheel odometry. A Pi 4 can't afford planner-sized numerics
in the same Python process as the control loop, while the loops that must react within a
tick (contact, stall, the base reaching its goal, a lost planner) stay here, next to the
motors, independent of planner/network latency (seconds, for a VLM).

Safety, enforced here rather than trusted to the planner:
- A command runs only while the planner keeps sending heartbeats (HEARTBEAT_TIMEOUT_S).
- Every trajectory segment respects MAX_JOINT_SPEED and the joint ranges, and the TCP and
  the jaw tips stay above z_min all along it (forward kinematics only, sampled at least every
  CLEARANCE_SAMPLE_UNITS of joint motion; ~1 ms per sample on a Pi 4).
- Arm moves abort on a stalled joint or a large tracking error (the arm hit something),
  and then hold the *measured* pose instead of pushing on toward the commanded one.
- Base speed and distance per move are capped, with a timeout.
- Any gamepad input, e-stop or arm reset leaves auto mode (see RobotBridge); leaving it
  aborts the running command.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable

import numpy as np

import so101_kinematics as kin

logger = logging.getLogger(__name__)

IK_ARM_JOINTS = [f"arm_{name}" for name in kin.JOINT_NAMES]
GRIPPER = "arm_gripper"
ARM_JOINTS = [*IK_ARM_JOINTS, GRIPPER]

# 1.5 s (was 0.6): over the robot's Wi-Fi hotspot the Mac's heartbeats arrive with 1.5-4 s
# gaps a few times a minute (Oct 24: 11 gaps > 0.6 s in 84 s), while the cable never exceeded
# 32 ms. Worst case after a real loss: the arm keeps moving ~1.5 s (a few mm at these
# speeds), the base up to ~0.2 m at its 0.15 m/s cap.
HEARTBEAT_TIMEOUT_S = 1.5

# Joint units are LeKiwi's normalized -100..100 (gripper 0..100).
JOINT_RANGE = dict.fromkeys(IK_ARM_JOINTS, (-100.0, 100.0))
JOINT_RANGE[GRIPPER] = (0.0, 100.0)
MAX_JOINT_SPEED = 60.0  # units/s, on any joint in any segment
DEFAULT_JOINT_SPEED = 35.0
MAX_WAYPOINTS = 600
CLEARANCE_SAMPLE_UNITS = 5.0
# The jaw tips extend this far past the TCP along its approach axis (gemma_brain/arm.py's
# JAW_TIP_OFFSET_M, from the floor probe).
JAW_TIP_OFFSET_M = 0.015
# Floor guard for the TCP z in the arm base frame when a command doesn't send its own
# z_min. Conservative until the floor height is calibrated.
DEFAULT_Z_MIN_M = -0.05
MAX_GRIPPER_SPEED = 80.0
DEFAULT_GRIPPER_SPEED = 50.0

MAX_BASE_XY_SPEED = 0.15  # m/s
MAX_BASE_THETA_SPEED = 45.0  # deg/s
DEFAULT_BASE_XY_SPEED = 0.08
DEFAULT_BASE_THETA_SPEED = 25.0
MIN_BASE_XY_SPEED = 0.025  # below this the omniwheels barely move
MIN_BASE_THETA_SPEED = 8.0
BASE_POS_TOL_M = 0.01
BASE_THETA_TOL_DEG = 2.0
BASE_GAIN = 1.5  # 1/s: commanded speed = gain * remaining, then capped

# Abort thresholds while an arm move runs (gripper excluded: squeezing a block is a stall).
TRACKING_ERROR_LIMIT = 12.0  # units, |target - measured| on any IK joint
TRACKING_ERROR_HOLD_S = 0.3
STALL_SEVERITY_LIMIT = 0.25
STALL_HOLD_S = 0.25
STALL_LAG_MIN = 4.0  # units: below this lag, a loaded joint is just holding, not blocked
SETTLE_TIMEOUT_S = 1.0

# Contact-stop trajectories (the floor touch): watch the TCP's measured height (FK of the
# measured joints) against the commanded one every tick. In free air the measured TCP trails
# the command by a steady sag/lag (its "free gap", measured over CONTACT_BASELINE_S after
# CONTACT_WARMUP_S); contact is either the floor holding it up (gap grows past the free gap
# by contact_gap_mm on CONTACT_CONFIRM_TICKS ticks) or the TCP no longer following (over the
# last CONTACT_FOLLOW_S it moved under CONTACT_FOLLOW_FRACTION of the commanded distance,
# which also catches starting already in contact). The commanded distance counts only below
# where the TCP measured at the window's start: an arm hanging below its command (the
# vertical gripper droops ~4 mm) doesn't move until the command passes it, and counting from
# the command fired on every touch, 22-26 mm up (Oct 27, run 20261027T085930).
# On contact: hold the measured pose.
CONTACT_WARMUP_S = 0.15
CONTACT_BASELINE_S = 0.2
CONTACT_CONFIRM_TICKS = 2
CONTACT_FOLLOW_S = 0.3
CONTACT_FOLLOW_FRACTION = 0.4
CONTACT_FOLLOW_MIN_M = 0.002

# Gripper: closed is 0. When it stalls on an object, hold this far past the measured
# position, so it keeps a firm grip without driving the servo at full torque.
GRIPPER_SQUEEZE = 4.0
GRIPPER_STALL_TICKS = 4

# Hold: after a close stalls on an object, the hold is watched every tick until the next
# gripper command (status()["gripper_hold"]).
# Dropped: the jaws closed this far past where they stalled, for HOLD_DROP_TICKS ticks. With
# the object gone they travel most of the way to the squeeze target (GRIPPER_SQUEEZE). Load
# can't tell: the firmware itself cuts the gripper to Protective_Torque (20%) after ~2 s over
# Overload_Torque (25%), and today's holds load it ~41% (registers read Oct 26).
HOLD_DROP_UNITS = 2.5
HOLD_DROP_TICKS = 3
# Protection: over-current and overload are the gripper firmware's own (Protection_Current
# 250 ~1.6 A, Overload_Torque 25% -> Protective_Torque 20%, each after ~2 s); a load rule here
# would fire on every hold (~41%). Its only thermal response is cutting torque at
# Max_Temperature_Limit 70 C, which drops the object, so temperature is watched here: at
# GRIPPER_TEMP_WARN_C the squeeze eases to GRIPPER_SQUEEZE_EASED, at GRIPPER_TEMP_HARD_C it
# stops (holds the stalled position; never opens: the object would fall).
GRIPPER_SQUEEZE_EASED = 2.0
GRIPPER_TEMP_WARN_C = 55.0
GRIPPER_TEMP_HARD_C = 62.0
HOLD_MAX_EVENTS = 5
HOLD_PUBLIC = ("state", "pos", "squeeze", "temp", "events")


class CommandRefused(ValueError):
    """The command was rejected before anything moved; the message says why."""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class AutoController:
    """Holds the one active auto command and steps it from RobotBridge's control loop.

    Thread model: submit()/abort()/heartbeat()/status() are called from HTTP request threads;
    step() only from the control loop.
    """

    def __init__(
        self,
        norm_to_rad: Callable[[str, float], float],
        clock: Callable[[], float] = time.monotonic,
    ):
        # Must be the same clock the control loop passes to step() as `now`.
        self._clock = clock
        self._norm_to_rad = norm_to_rad
        self._lock = threading.Lock()
        self._last_heartbeat = -math.inf
        self._next_id = 1
        self._cmd: dict | None = None  # the running command
        self._last: dict | None = None  # the most recent finished command
        self._targets: dict[str, float] = {}  # the bridge's joint targets as of the last tick
        self._odom = np.zeros(3)  # x, y (m), theta (rad), integrated from wheel velocities
        self._odom_time: float | None = None
        self._hold: dict | None = None  # the gripper's hold on an object (see HOLD_*)

    # --- planner-facing (HTTP threads) ---

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = self._clock()

    def submit(self, cmd: dict) -> int:
        """Validates cmd and queues it to run. Raises CommandRefused (nothing moves) if it
        isn't safe. Returns the command id."""
        kind = cmd.get("cmd")
        with self._lock:
            if self._clock() - self._last_heartbeat > HEARTBEAT_TIMEOUT_S:
                raise CommandRefused("no recent heartbeat from the planner")
            self._refuse_if_busy_locked()
            start = dict(self._targets)
        if not start:
            raise CommandRefused("no joint state yet")

        if kind == "trajectory":
            run = self._check_trajectory(start, cmd)
        elif kind == "joints":
            run = self._plan_joints(start, cmd)
        elif kind == "gripper":
            run = self._plan_gripper(start, cmd)
        elif kind == "base":
            run = self._plan_base(cmd)
        else:
            raise CommandRefused(f"unknown cmd {kind!r} (trajectory, joints, gripper, base)")

        with self._lock:
            self._refuse_if_busy_locked()
            cmd_id = self._next_id
            self._next_id += 1
            if kind == "gripper":
                self._hold = None  # a new open or close ends the old hold
            self._cmd = {
                "id": cmd_id,
                "kind": kind,
                "state": "running",
                "reason": "",
                "result": {},
                "submitted": time.time(),
                "finished": None,
                "started": None,
                **run,
            }
        logger.info("auto: command %d (%s) accepted", cmd_id, kind)
        return cmd_id

    def abort(self, reason: str) -> None:
        with self._lock:
            self._finish_locked("aborted", reason)

    def status(self, measured: dict[str, float]) -> dict:
        """The current (or last finished) command, the arm's commanded and measured joints,
        its TCP pose from the measured joints, and the base odometry."""
        with self._lock:
            cur = self._cmd if self._cmd is not None else self._last
            targets = dict(self._targets)
            odom = self._odom.copy()
            hold = None if self._hold is None else {k: self._hold[k] for k in HOLD_PUBLIC}
        public = ("id", "kind", "state", "reason", "result", "submitted", "finished")
        out = {
            "command": None if cur is None else {k: cur[k] for k in public},
            "joints": measured,
            "targets": targets,
            "odom": {"x": float(odom[0]), "y": float(odom[1]), "theta_deg": math.degrees(odom[2])},
            "gripper_hold": hold,
            "time": time.time(),
        }
        if all(j in measured for j in IK_ARM_JOINTS):
            pose = kin.forward_kinematics(np.array([self._norm_to_rad(j, measured[j]) for j in IK_ARM_JOINTS]))
            out["ee"] = {"xyz": [float(v) for v in pose[:3, 3]], "rot": pose[:3, :3].round(4).tolist()}
        return out

    # --- control loop ---

    def step(
        self,
        now: float,
        targets: dict[str, float],
        joint_state: dict[str, dict],
        base_vel: dict[str, float],
    ) -> dict[str, float]:
        """Advances the running command by one control tick. Mutates targets (the bridge's
        joint targets) in place and returns the base velocity command (x.vel, y.vel in m/s,
        theta.vel in deg/s)."""
        vel = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self._integrate_odom(now, base_vel)
        with self._lock:
            cmd = self._cmd
            if cmd is not None and now - self._last_heartbeat > HEARTBEAT_TIMEOUT_S:
                if cmd["kind"] == "base" and "odom_start" in cmd:
                    cmd["result"] = {"moved": self._base_moved(cmd)}  # so the planner can finish it
                self._finish_locked("aborted", "heartbeat lost")
                if cmd["kind"] != "base":
                    self._hold_measured(targets, joint_state)
                cmd = None
            if cmd is not None and cmd["started"] is None:
                cmd["started"] = now
                cmd["odom_start"] = self._odom.copy()

        if cmd is not None:
            t = now - cmd["started"]
            if cmd["kind"] == "base":
                vel = self._step_base(cmd, t)
            elif cmd["kind"] == "gripper":
                self._step_gripper(cmd, t, targets, joint_state)
            else:
                self._step_trajectory(cmd, t, targets, joint_state, now)
        if GRIPPER in joint_state and (cmd is None or cmd["kind"] != "gripper"):
            self._watch_hold(now, targets, joint_state[GRIPPER])
        with self._lock:
            self._targets = dict(targets)
        return vel

    # --- validation ---

    def _refuse_if_busy_locked(self) -> None:
        if self._cmd is not None:
            raise CommandRefused(f"busy: command {self._cmd['id']} ({self._cmd['kind']}) is still running")

    def _check_trajectory(self, start: dict[str, float], cmd: dict) -> dict:
        """waypoints: [{joint: position}, ...] (each may name any subset of ARM_JOINTS; joints
        it leaves out keep their previous value), durations: seconds per segment, the first
        segment starting from the arm's current targets."""
        wps, durs = cmd.get("waypoints"), cmd.get("durations")
        if not isinstance(wps, list) or not isinstance(durs, list) or not wps or len(wps) != len(durs):
            raise CommandRefused("trajectory: 'waypoints' and 'durations' must be equal-length non-empty lists")
        if len(wps) > MAX_WAYPOINTS:
            raise CommandRefused(f"trajectory: at most {MAX_WAYPOINTS} waypoints")
        z_min = float(cmd["z_min"]) if cmd.get("z_min") is not None else DEFAULT_Z_MIN_M
        prev = {j: start[j] for j in ARM_JOINTS}
        full = [prev]
        for i, (wp, d) in enumerate(zip(wps, durs, strict=True)):
            try:
                d = float(d)
                wp = {j: float(v) for j, v in wp.items()}
            except (AttributeError, TypeError, ValueError):
                raise CommandRefused(f"trajectory: waypoint {i} must be {{joint: number}}, duration a number") from None
            if not d > 0:
                raise CommandRefused(f"trajectory: duration {i} must be > 0")
            cur = dict(prev)
            for j, v in wp.items():
                if j not in JOINT_RANGE:
                    raise CommandRefused(f"trajectory: unknown joint {j!r}")
                lo, hi = JOINT_RANGE[j]
                if not lo <= v <= hi:
                    raise CommandRefused(f"trajectory: waypoint {i}: {j} = {v:.1f} is outside {lo:.0f}..{hi:.0f}")
                cur[j] = v
            speed = max(abs(cur[j] - prev[j]) for j in ARM_JOINTS) / d
            if speed > MAX_JOINT_SPEED * 1.001:
                hint = " -- does the first waypoint start near the current pose?" if i == 0 else ""
                raise CommandRefused(
                    f"trajectory: segment {i} needs {speed:.0f} units/s (max {MAX_JOINT_SPEED:.0f}){hint}"
                )
            span = max(abs(cur[j] - prev[j]) for j in IK_ARM_JOINTS)
            n = max(1, math.ceil(span / CLEARANCE_SAMPLE_UNITS))
            for k in range(1, n + 1):
                a = k / n
                z = self._lowest_z({j: prev[j] + (cur[j] - prev[j]) * a for j in IK_ARM_JOINTS})
                if z < z_min:
                    where = f"waypoint {i}" if k == n else f"{100 * a:.0f}% of the way to waypoint {i}"
                    raise CommandRefused(f"trajectory: the gripper reaches z {z:.3f} m at {where}, below z_min {z_min:.3f} m")
            full.append(cur)
            prev = cur
        durations = [float(d) for d in durs]
        run = {"waypoints": full, "durations": durations, "total": sum(durations)}
        if cmd.get("contact_stop"):
            run["contact_stop"] = True
            run["contact_gap_m"] = _clamp(float(cmd.get("contact_gap_mm") or 1.5), 0.8, 5.0) / 1000.0
            run["watch"] = []
        return run

    def _lowest_z(self, joints: dict[str, float]) -> float:
        """Lower of the TCP and the jaw tips."""
        q = np.array([self._norm_to_rad(j, joints[j]) for j in IK_ARM_JOINTS])
        t = kin.forward_kinematics(q)
        return float(min(t[2, 3], t[2, 3] + JAW_TIP_OFFSET_M * t[2, 2]))

    def _plan_joints(self, start: dict[str, float], cmd: dict) -> dict:
        """One straight joint-space move, smoothstepped, at `speed` units/s (average) on the
        joint that moves farthest. Validated like a one-segment trajectory."""
        goal = cmd.get("targets")
        if not isinstance(goal, dict) or not goal:
            raise CommandRefused("joints: 'targets' must be a non-empty {joint: position} dict")
        speed = _clamp(float(cmd.get("speed") or DEFAULT_JOINT_SPEED), 1.0, MAX_JOINT_SPEED / 1.5)
        try:
            span = max([abs(float(v) - start[j]) for j, v in goal.items() if j in start] or [0.0])
        except (TypeError, ValueError):
            raise CommandRefused("joints: positions must be numbers") from None
        # Smoothstep's peak speed is 1.5x its average; the check below uses the average, so
        # the cap on `speed` above keeps the peak within MAX_JOINT_SPEED.
        dur = max(span / speed, 0.2)
        run = self._check_trajectory(start, {"waypoints": [goal], "durations": [dur], "z_min": cmd.get("z_min")})
        run["smooth"] = True
        return run

    def _plan_gripper(self, start: dict[str, float], cmd: dict) -> dict:
        try:
            goal = _clamp(float(cmd["pos"]), *JOINT_RANGE[GRIPPER])
        except (KeyError, TypeError, ValueError):
            raise CommandRefused("gripper: 'pos' must be a number, 0 (closed) .. 100 (open)") from None
        speed = _clamp(float(cmd.get("speed") or DEFAULT_GRIPPER_SPEED), 5.0, MAX_GRIPPER_SPEED)
        return {"gripper_start": start[GRIPPER], "gripper_goal": goal, "gripper_speed": speed, "stall_ticks": 0}

    def _plan_base(self, cmd: dict) -> dict:
        try:
            dx, dy = float(cmd.get("dx") or 0.0), float(cmd.get("dy") or 0.0)
            dth = float(cmd.get("dtheta_deg") or 0.0)
            vxy = float(cmd.get("speed") or DEFAULT_BASE_XY_SPEED)
            vth = float(cmd.get("theta_speed") or DEFAULT_BASE_THETA_SPEED)
        except (TypeError, ValueError):
            raise CommandRefused("base: dx, dy (m), dtheta_deg and speeds must be numbers") from None
        if abs(dx) > 1.0 or abs(dy) > 1.0 or abs(dth) > 180.0:
            raise CommandRefused("base: one move is limited to 1 m per axis and 180 deg")
        vxy = _clamp(vxy, MIN_BASE_XY_SPEED, MAX_BASE_XY_SPEED)
        vth = _clamp(vth, MIN_BASE_THETA_SPEED, MAX_BASE_THETA_SPEED)
        expected = max(math.hypot(dx, dy) / vxy, abs(dth) / vth)
        return {
            "base_goal": np.array([dx, dy, math.radians(dth)]),
            "base_speed": (vxy, vth),
            "timeout": 3.0 * expected + 2.0,
        }

    # --- execution ---

    def _tcp_z(self, joints: dict[str, float]) -> float:
        q = np.array([self._norm_to_rad(j, joints[j]) for j in IK_ARM_JOINTS])
        return float(kin.forward_kinematics(q)[2, 3])

    def _watch_contact(self, cmd: dict, t: float, targets: dict, joint_state: dict) -> str:
        """For a contact_stop trajectory: a non-empty description once the TCP is in contact
        (see the CONTACT_* constants)."""
        if not all(j in joint_state for j in IK_ARM_JOINTS):
            return ""
        zc = self._tcp_z(targets)
        zm = self._tcp_z({j: joint_state[j]["pos"] for j in IK_ARM_JOINTS})
        hist = cmd["watch"]
        hist.append((t, zc, zm))
        if t < CONTACT_WARMUP_S + CONTACT_BASELINE_S:
            return ""
        if "free_gap" not in cmd:
            gaps = [m - c for tt, c, m in hist if tt >= CONTACT_WARMUP_S]
            cmd["free_gap"] = float(np.median(gaps)) if gaps else 0.0
        gap = zm - zc
        cmd["contact_last"] = (zc, zm, gap)
        if gap - cmd["free_gap"] > cmd["contact_gap_m"]:
            cmd["held_ticks"] = cmd.get("held_ticks", 0) + 1
            if cmd["held_ticks"] >= CONTACT_CONFIRM_TICKS:
                return f"held up: {1000 * (gap - cmd['free_gap']):.1f} mm above its free-air sag"
        else:
            cmd["held_ticks"] = 0
        old = next((h for h in reversed(hist) if t - h[0] >= CONTACT_FOLLOW_S), None)
        if old is not None:
            # descent over the window; the command's only from below where the TCP was
            commanded, measured = min(old[1], old[2]) - zc, old[2] - zm
            if commanded > CONTACT_FOLLOW_MIN_M and measured < CONTACT_FOLLOW_FRACTION * commanded:
                return f"not following: moved {1000 * measured:.1f} of {1000 * commanded:.1f} mm commanded"
        return ""

    def _finish_contact(self, cmd: dict, how: str, targets: dict, joint_state: dict, t: float) -> None:
        zc, zm, gap = cmd.get("contact_last", (None, None, None))
        self._hold_measured(targets, joint_state)
        cmd["result"] = {
            "contact": True,
            "how": how,
            "z_meas": None if zm is None else round(zm, 4),
            "z_cmd": None if zc is None else round(zc, 4),
            "free_gap_mm": round(1000 * cmd.get("free_gap", 0.0), 1),
            "t": round(t, 2),
        }
        with self._lock:
            self._finish_locked("done", f"contact ({how})")

    def _step_trajectory(self, cmd: dict, t: float, targets: dict, joint_state: dict, now: float) -> None:
        wps, durs = cmd["waypoints"], cmd["durations"]
        if cmd.get("contact_stop"):
            how = self._watch_contact(cmd, t, targets, joint_state)
            if how:
                self._finish_contact(cmd, how, targets, joint_state, t)
                return
        if t >= cmd["total"]:
            targets.update(wps[-1])
            err = self._max_tracking_error(targets, joint_state)
            if err < 3.0 or t - cmd["total"] > SETTLE_TIMEOUT_S:
                # "done" either way, but say how close it got: a stretched-out arm sags short
                # of its target and only settles by the timeout.
                cmd["result"] = {
                    **({"contact": False} if cmd.get("contact_stop") else {}),
                    "settled": err < 3.0,
                    "max_error": round(err, 1),
                    "errors": {j: round(targets[j] - joint_state[j]["pos"], 1) for j in IK_ARM_JOINTS if j in joint_state},
                }
                with self._lock:
                    self._finish_locked("done", "" if err < 3.0 else f"settle timeout, {err:.1f} units off")
                return
        else:
            i, rem = 0, t
            while rem >= durs[i]:
                rem -= durs[i]
                i += 1
            a = rem / durs[i]
            if cmd.get("smooth"):
                a = a * a * (3 - 2 * a)
            w0, w1 = wps[i], wps[i + 1]
            for j in ARM_JOINTS:
                targets[j] = w0[j] + (w1[j] - w0[j]) * a
        reason = self._contact_check(cmd, targets, joint_state, now)
        if reason:
            if cmd.get("contact_stop"):
                # Looking for contact: a stall or tracking error is the contact we wanted.
                self._finish_contact(cmd, reason, targets, joint_state, t)
                return
            self._hold_measured(targets, joint_state)
            with self._lock:
                self._finish_locked("failed", reason)

    def _contact_check(self, cmd: dict, targets: dict, joint_state: dict, now: float) -> str:
        """Non-empty reason when the arm looks blocked: a stalled joint or a tracking error
        that persists (a single-tick spike is normal at the start of a move)."""
        # A stall counts only on a joint that's also lagging its target: a joint holding the
        # arm out against gravity is loaded and stationary too, and that's fine.
        stalled = [
            j
            for j in IK_ARM_JOINTS
            if j in joint_state
            and joint_state[j].get("severity", 0.0) > STALL_SEVERITY_LIMIT
            and abs(targets[j] - joint_state[j]["pos"]) > STALL_LAG_MIN
        ]
        if stalled:
            since = cmd.setdefault("stall_since", now)
            if now - since > STALL_HOLD_S:
                return f"stall on {', '.join(stalled)} (blocked or in contact)"
        else:
            cmd.pop("stall_since", None)
        err = self._max_tracking_error(targets, joint_state)
        if err > TRACKING_ERROR_LIMIT:
            since = cmd.setdefault("track_since", now)
            if now - since > TRACKING_ERROR_HOLD_S:
                return f"tracking error {err:.1f} units (blocked or in contact)"
        else:
            cmd.pop("track_since", None)
        return ""

    @staticmethod
    def _max_tracking_error(targets: dict, joint_state: dict) -> float:
        errs = [abs(targets[j] - joint_state[j]["pos"]) for j in IK_ARM_JOINTS if j in joint_state]
        return max(errs) if errs else 0.0

    def _step_gripper(self, cmd: dict, t: float, targets: dict, joint_state: dict) -> None:
        start, goal, speed = cmd["gripper_start"], cmd["gripper_goal"], cmd["gripper_speed"]
        span = abs(goal - start)
        a = 1.0 if span < 1e-6 else min(1.0, t * speed / span)
        targets[GRIPPER] = start + (goal - start) * a
        st = joint_state.get(GRIPPER, {})
        meas, load = st.get("pos", targets[GRIPPER]), st.get("load", 0.0)
        # Stalled on an object: closing, the target has run ahead of the measured position,
        # and the motor is loaded and not moving (the bridge's stall severity).
        if goal < start and targets[GRIPPER] < meas - 3.0 and st.get("severity", 0.0) > 0.0:
            cmd["stall_ticks"] += 1
        else:
            cmd["stall_ticks"] = 0
        if cmd["stall_ticks"] >= GRIPPER_STALL_TICKS:
            targets[GRIPPER] = max(JOINT_RANGE[GRIPPER][0], meas - GRIPPER_SQUEEZE)
            with self._lock:
                cmd["result"] = {"pos": round(meas, 1), "load": load, "stalled": True, "temp": st.get("temp")}
                self._hold = {"state": "holding", "pos": round(meas, 1), "squeeze": GRIPPER_SQUEEZE,
                              "temp": st.get("temp"), "events": [], "drop_ticks": 0}
                self._finish_locked("done", "stalled on an object")
            return
        if a >= 1.0 and (abs(meas - goal) < 2.0 or t > span / speed + SETTLE_TIMEOUT_S):
            with self._lock:
                cmd["result"] = {"pos": round(meas, 1), "load": load, "stalled": False}
                self._finish_locked("done", "")

    def _integrate_odom(self, now: float, base_vel: dict[str, float]) -> None:
        if self._odom_time is None:
            self._odom_time = now
            return
        dt = min(now - self._odom_time, 0.2)
        self._odom_time = now
        vx, vy = base_vel.get("x.vel", 0.0), base_vel.get("y.vel", 0.0)
        wz = math.radians(base_vel.get("theta.vel", 0.0))
        th = self._odom[2]
        self._odom += np.array(
            [(vx * math.cos(th) - vy * math.sin(th)) * dt, (vx * math.sin(th) + vy * math.cos(th)) * dt, wz * dt]
        )

    def _base_moved(self, cmd: dict) -> list[float]:
        """[dx (m), dy (m), dtheta (deg)] covered so far, in the frame the move started in."""
        start = cmd["odom_start"]
        c, s = math.cos(start[2]), math.sin(start[2])
        d = self._odom - start
        return [round(float(c * d[0] + s * d[1]), 3), round(float(-s * d[0] + c * d[1]), 3), round(math.degrees(d[2]), 1)]

    def _step_base(self, cmd: dict, t: float) -> dict[str, float]:
        stop = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        start = cmd["odom_start"]
        # Progress so far, in the frame the robot had when the move started.
        c, s = math.cos(start[2]), math.sin(start[2])
        d = self._odom - start
        done = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
        rem = cmd["base_goal"] - done
        vxy, vth = cmd["base_speed"]
        rem_xy = math.hypot(rem[0], rem[1])
        rem_th = math.degrees(rem[2])
        result = {"moved": [round(float(done[0]), 3), round(float(done[1]), 3), round(math.degrees(done[2]), 1)]}
        if rem_xy < BASE_POS_TOL_M and abs(rem_th) < BASE_THETA_TOL_DEG:
            with self._lock:
                cmd["result"] = result
                self._finish_locked("done", "")
            return stop
        if t > cmd["timeout"]:
            with self._lock:
                cmd["result"] = result
                self._finish_locked("failed", f"timeout after {t:.1f} s (wheels slipping or blocked?)")
            return stop
        # Remaining translation, rotated into the robot's *current* body frame.
        cur = d[2]
        bx = math.cos(cur) * rem[0] + math.sin(cur) * rem[1]
        by = -math.sin(cur) * rem[0] + math.cos(cur) * rem[1]
        v = _clamp(BASE_GAIN * rem_xy, MIN_BASE_XY_SPEED, vxy) if rem_xy >= BASE_POS_TOL_M else 0.0
        w = 0.0
        if abs(rem_th) >= BASE_THETA_TOL_DEG:
            w = math.copysign(_clamp(BASE_GAIN * abs(rem_th), MIN_BASE_THETA_SPEED, vth), rem_th)
        scale = v / rem_xy if rem_xy > 1e-9 else 0.0
        return {"x.vel": bx * scale, "y.vel": by * scale, "theta.vel": w}

    # --- the hold ---

    def _watch_hold(self, now: float, targets: dict, st: dict) -> None:
        """One tick of the hold on an object (HOLD_*): spot a drop from the jaw position, and
        ease the squeeze on high temperature."""
        with self._lock:
            h = self._hold
            if h is None:
                return
            meas = st.get("pos", h["pos"])
            temp = st.get("temp")
            h["temp"] = temp
            if h["state"] == "dropped":
                return
            if meas < h["pos"] - HOLD_DROP_UNITS:
                h["drop_ticks"] += 1
                if h["drop_ticks"] >= HOLD_DROP_TICKS:
                    h["state"] = "dropped"
                    targets[GRIPPER] = max(JOINT_RANGE[GRIPPER][0], meas)  # stop pushing on nothing
                    self._hold_event(h, f"dropped: the jaws closed to {meas:.1f} from {h['pos']:.1f}")
                return
            h["drop_ticks"] = 0
            squeeze, why = h["squeeze"], ""
            if temp is not None and temp >= GRIPPER_TEMP_HARD_C:
                squeeze, why = 0.0, f"temperature {temp:.0f} C"
            elif temp is not None and temp >= GRIPPER_TEMP_WARN_C:
                squeeze, why = min(squeeze, GRIPPER_SQUEEZE_EASED), f"temperature {temp:.0f} C"
            if squeeze < h["squeeze"]:
                h["squeeze"] = squeeze
                targets[GRIPPER] = max(JOINT_RANGE[GRIPPER][0], h["pos"] - squeeze)
                self._hold_event(h, f"squeeze eased to {squeeze:.0f}: {why}")

    @staticmethod
    def _hold_event(h: dict, what: str) -> None:
        logger.warning("auto: gripper hold: %s", what)
        h["events"] = (h["events"] + [{"t": time.time(), "what": what}])[-HOLD_MAX_EVENTS:]

    # --- helpers ---

    @staticmethod
    def _hold_measured(targets: dict, joint_state: dict) -> None:
        for j in IK_ARM_JOINTS:
            if j in joint_state:
                targets[j] = joint_state[j]["pos"]

    def _finish_locked(self, state: str, reason: str) -> None:
        cmd = self._cmd
        if cmd is None:
            return
        cmd["state"] = state
        cmd["reason"] = reason
        cmd["finished"] = time.time()
        log = logger.info if state == "done" else logger.warning
        log("auto: command %d (%s) %s%s", cmd["id"], cmd["kind"], state, f": {reason}" if reason else "")
        self._last = cmd
        self._cmd = None
