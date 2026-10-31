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

"""Every number the skills use, with where it comes from. The owner's rule (Oct 27): few
hand-set values; each one is measured, taken from a demo, a spec, or set by a stated rule.

Source tags: CALIB = measured by a script into calib.json (see geometry.py); DEMO = the owner's
demo or teach; SPEC = part datasheet / object; RULE = generic control value set by a stated
rule, the same for any object. Nothing here is tuned per pick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
# Committed (data/demos/): the raw recording under demos/ is not, and without this file a
# fresh checkout silently fell back to the defaults below (a 9 cm lift, no place view).
DEMO_EXTRACT = HERE / "data" / "demos" / "pick_cup" / "extract-ep0.json"
CALIB_FILE = HERE / "data" / "calib.json"


@dataclass(frozen=True)
class Param:
    value: float
    source: str  # CALIB | DEMO | SPEC | RULE
    why: str


def _demo(key: str, fallback: float) -> float:
    try:
        return float(json.loads(DEMO_EXTRACT.read_text())[key])
    except (OSError, KeyError, ValueError):
        return fallback


def _floor_accuracy(fallback: float) -> float:
    """How well the floor under the reach is known: the last calibrate_all.py run's check band (m)
    when its touches and the cameras agreed within it, else the fallback."""
    try:
        ck = json.loads(CALIB_FILE.read_text())["floor_tcp_z_m"]["check"]
        return ck["band_mm"] / 1000.0 if ck.get("made") and ck.get("pass") else fallback
    except (OSError, KeyError, ValueError, TypeError):
        return fallback


def demo_place_view() -> tuple[list[float], float] | None:
    """The container's box in the wrist image at the demo's pause above it, and that height
    (jaw tips above the floor): where the owner held the brick before lowering it in."""
    try:
        d = json.loads(DEMO_EXTRACT.read_text())
        return list(d["pre_release"]["box"]), float(d["above_target_z_above_floor_m"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


# --- DEMO (the owner's Oct 27 pick_cup demo; re-extract after the joint-zero fix for heights) ---
OPEN_WIDTH = Param(_demo("open_width_mm", 44.5), "DEMO", "jaw opening before a grasp, mm (demo: 44.5, opened to 32.1 deg / 24.6 normalized units; geometry.gripper_mm)")
LIFT_M = Param(_demo("lift_m", 0.09), "DEMO", "lift after the grasp (demo: 9.5 cm)")
RELEASE_ABOVE_FLOOR_M = Param(_demo("release_z_above_floor_m", 0.06), "DEMO",
                              "TCP height above the floor at release inside a cup (demo: 5.9 cm)")
PLACE_HOVER_M = Param(_demo("above_target_z_above_floor_m", 0.10), "DEMO",
                      "height above a container before lowering into it (demo: 10.6 cm; its view there is the place target)")
# The grasp target in the wrist image (where the brick's box sits just before closing for a
# middle grip) comes from the owner's midpoint teach: geometry.grasp_target(); not here yet.

# --- SPEC ---
BRICK_HEIGHT_M = Param(0.0114, "SPEC", "2x4 brick body height")
BASE_STOP_SHORT_M = Param(0.01, "SPEC", "the Pi ends a base drive within this of its goal (auto_control.BASE_POS_TOL_M): drives under it didn't move, longer ones stopped ~1 cm short (Oct 27 trials)")
BASE_TURN_MAX_DEG = Param(180.0, "SPEC", "the Pi refuses a base turn over this in one command (auto_control: 1 m per axis, 180 deg)")
BASE_TURN_STOP_SHORT_DEG = Param(2.0, "SPEC", "the Pi ends a base turn within this of its goal (auto_control.BASE_THETA_TOL_DEG): turns stopped ~2 deg short")
CONTROL_PERIOD_S = Param(1 / 30, "SPEC", "the Pi reads the joints once per control tick (web_server/robot_bridge.py FPS 30): status reads closer together can return the same sample")
GRIPPER_EMPTY_MAX = Param(5.0, "SPEC", "mm: a close that stops wider holds something: closed on nothing the jaws meet (0 mm, 1.2-1.6 deg), the brick's narrowest side is 15.8 mm (calibrate_gripper.py, Oct 29; was 4.6 deg, 3.5 normalized units)")

# --- RULE ---
GRASP_LIFT_M = Param(0.004, "RULE", "jaw tips above the floor when closing: inside the brick's lower half")
SERVO_GAIN = Param(0.8, "RULE", "fraction of the measured error corrected per look: 0.5 until Oct 28; with a measured image Jacobian (the fitted one until Oct 30, the camera model's since, within 3 px of each other on logged moves) less damping is needed, and the simulated servo converges at 0.8 with the Jacobian 30% off either way")
STEP_IN_VIEW_FRACTION = Param(0.25, "RULE", "one move may shift the target by at most this share of the image, so it stays in view")
REACH_POINT_X_M = Param(0.25, "RULE", "where the base parks the object (arm x): vertical reach at 10 cm is 0.14-0.28 m; owner: 0.22 sat too close to the base")
HOVER_HEIGHT_M = Param(0.10, "RULE", "line-up and first servo height: the object fills a small part of the view, with room to line up (the fitted image Jacobian was measured at 9.4 cm; the camera model holds at any height)")
FINAL_LOOK_HEIGHT_M = Param(0.03, "RULE", "last servo height: jaws clear of the brick's top (11.4 mm) by more than the floor fit's error")
PLACE_APPROACH_TOL_M = Param(0.05, "RULE", "the approach to a container is done this close (its near edge is a rough mark; the wrist step after it centres it): at the brick's 1.5 cm a cup approach ran out of looks (Oct 28)")
PLACE_TOL_M = Param(0.002, "RULE", "how centred over a container before lowering: as tight as the grasp servo (SERVO_TOL_M); at 1 cm, measured at the floor, the brick landed in the cup near the rim (the rim is nearer the camera: ~1.5x the error; Oct 27)")
NEXT_GAP_FRACTION = Param(0.5, "RULE", "gap left beside the reference object, as a share of the held brick's width in the image")
LOOK_OVERLAP = Param(0.5, "RULE", "look-around: consecutive front views overlap by this share of the view (turn step = the rest of the view's width, from the front camera's calibration)")
CLOSE_CHOICE_M = Param(0.5, "RULE", "front-camera looks this close to the object go to the main model (26B), whatever is tracked: on Oct 30 it chose the named block among several in all 15 asks at 0.4-0.5 m (red, pink, blue, light blue, green, light green), but 4 of 7 at ~1 m (15-30 px blocks: a wrong block, a miss, a false find)")
LOOK_BACK_M = Param(0.3, "RULE", "look-around: back up this far once if a full turn didn't find it (it may be too close to the front camera)")
LAND_SEARCH_M = Param(
    _floor_accuracy(0.008),
    "CALIB",
    "how far past its goal a contact-stop move may go: the floor's verified accuracy, the band within which "
    "calibrate_all.py's touches and cameras agreed (3 mm; Oct 30: 1.2 mm worst); 8.4 mm, an old floor fit's "
    "worst error, if no check passed",
)
SERVO_TOL_M = Param(0.002, "RULE", "servo position tolerance at the object: under the last descent's landing scatter (2.5 mm), so the servo isn't what limits a grasp")
REACH_TOL_M = Param(0.015, "RULE", "line-up: close enough for the arm to take the rest, with room: vertical reach at the hover goes 3 cm past the reach point, half of it (at 3 cm the arm then needed 4.4 cm after a large roll and was refused, Oct 27; v1's ARM_REACH_M was 1.5 cm)")
ANGLE_TOL_DEG = Param(12.0, "RULE", "roll tolerance: squeezing jaws square up a brick that far off (v1, Oct 23)")
ROLL_FROM_REST_DEG = Param(90.0, "RULE", "the wrist-roll joint stays within this of its rest angle (owner, Oct 27)")
SERVO_MAX_LOOKS = Param(8, "RULE", "looks per servo before giving up")
MIN_BASE_TURN_DEG = Param(3.0, "RULE", "smallest base turn that moves: the Pi finishes turns under 2 deg without moving")
LOW_MOVE_GUARD_M = Param(0.002, "RULE", "below a low move's lowest point the Pi's z_min sits this much lower: room for servo jitter, not for the jaws")
CLOSE_SPEED = Param(39.0, "RULE", "gripper closing speed (deg/s; 30 units/s until Oct 29): slow enough for the Pi's stall detection to stop on the brick")
LOOK_SETTLE_S = Param(0.3, "RULE", "longest wait after a move before a frame (servo settling); after an arm move it ends once the measured TCP stops (skills._settle)")
PICK_TRIES = Param(3, "RULE", "a pick step retries this many times in code before the planner replans (owner's choice, Oct 24)")
MAX_REPLANS = Param(2, "RULE", "replans per task before giving up")
FOLD_CLEAR_M = Param(0.165, "RULE", "fold: jaw tips above the floor beyond which folding can't catch a place target, so a fold that can't lift straight up still goes (v1's cup rim plane 0.105 + 0.06; Oct 26 the fold dragged a cup with the jaws ~13 cm up)")
FREE_SPEED_MPS = Param(0.04, "RULE", "straight-line speed in free air")
NEAR_SPEED_MPS = Param(0.01, "RULE", "speed for the last centimetres to the floor")

ALL = {name: p for name, p in globals().items() if isinstance(p, Param)}
