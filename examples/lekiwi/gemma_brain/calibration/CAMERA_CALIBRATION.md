# Camera calibration with ChArUco boards

One session with three printed ChArUco boards measures both cameras and three joint zeros,
replacing hand-tuned camera constants. Phase D of the measured-parameters work (branch
`lekiwi-measured-params`, 2026-10-29).

What it measures:

- **each lens**: focal length, optical centre, distortion (wrist and front camera);
- **the wrist camera's mount**: its pose on the gripper (TCP frame);
- **the front camera's pose** in the arm's frame (it sits higher and tilted down compared with
  the stock design, so no model has it);
- **joint zero errors** of shoulder lift, elbow flex and wrist flex. Shoulder pan and wrist
  roll can't be told apart from where the boards lie and how the camera is mounted; they stay 0.

Tools, in this folder: `boards.py` (detection), `capture_cameras.py` (lens and scene captures),
`solve_cameras.py` (the solve). The boards: `targets/` (PDFs; `make_boards.py` regenerates them).
Tests: `tests/test_camera_calibration.py`.

## What you need

Three boards, each one Letter or A4 page from `targets/` (print the PDF, not the preview PNG).
Each board has its own marker IDs, so a camera can see several at once. Dictionary DICT_5X5_100.

| Board | Squares | Square / marker | Marker IDs | Pattern size | Where it goes |
|---|---|---|---|---|---|
| wrist | 10 x 7 | 25 mm / 18 mm | 0-34 | 25 x 17.5 cm | flat on the floor |
| front | 7 x 5 | 35 mm / 26 mm | 35-51 | 24.5 x 17.5 cm | upright, in front |
| side | 7 x 5 | 35 mm / 26 mm | 52-68 | 24.5 x 17.5 cm | upright, to one side |

Preparing them:

1. Print at 100% / actual size, no "fit to page".
2. Measure the 100 mm scale bar under each board: 99.5-100.5 mm is fine. Off by more: note the
   measured length (the solve can be scaled to it; no need to reprint).
3. Glue or tape each print flat on stiff card or foam board. A curled or bowed sheet bends the
   corner grid, and that error goes straight into the result.
4. Matte paper, or lamps off to the side: glare hides corners.

Also: tape for the floor board, something heavy (a box, books) to prop each upright board.

## Board placement

Distances from the front edge of the robot's body, straight ahead of the arm (the arm's base
is about 10 cm behind that edge: `BODY_FRONT_X_M`, the planner's body keep-out, not a
measurement).

```
                 TOP VIEW (robot facing up the page)

                 ┌──────── FRONT board ────────┐   upright, facing the robot,
                 └─────────────────────────────┘   32 cm from the edge, centred
                                                       ╲
                                                        ╲  SIDE board: upright,
                 ┌─────────────────────────────┐         ╲ 25-40 cm out, 20-30 cm
                 │                             │          ╲ to one side, turned
                 │      WRIST board (flat)     │   ~17 cm  ╲ 30-45 deg to face the
                 │       25 cm across          │   deep     robot
                 │                             │
                 └─────────────────────────────┘   near edge ~6 cm from the robot
         ═══════════════════ robot front edge ═══════════════════
                         [ arm ]   [front camera]
                          LeKiwi
```

1. **Wrist board, flat on the floor**, printed side up, its 25 cm side left to right, centred
   on the arm. Near edge ~6 cm from the robot's front edge, far edge ~24 cm out: its centre is
   15 cm out (25 cm from the arm's base), where a brick is usually picked. Tape it down.
2. **Front board, upright**, its printed face straight back at the robot, ~32 cm from the
   front edge (42 cm from the arm's base), centred, ~8 cm beyond the floor board. Bottom edge on
   the floor: the pattern runs from near the floor to ~19 cm up. Propped so it can't tip.
   Both cameras must see it: the front camera with the arm folded, the wrist camera from the
   forward poses. It ties the front camera to the arm.
3. **Side board, upright**, 25-40 cm out and 20-30 cm to the left or right, turned 30-45 deg
   toward the robot, not hiding the front board from the front camera. It only adds views for
   the front camera's lens.

## How accurate the placement must be

By eye: a few centimetres don't matter. The solve measures each board's real pose as an
unknown; the positions only make sure each camera sees the right board from the planned poses.

| Board | Tolerance | Why |
|---|---|---|
| wrist (flat) | +/-3-5 cm, any small rotation | each view needs only part of it (6 corners or more); far off, the arm's poses move to it (`--board-x`), not the board |
| front (upright) | +/-5 cm in distance and sideways, facing the robot within ~15 deg | the one that matters most: in simulation the wrist camera saw it from only 7 of the 23 poses; keep it near centre |
| side (upright) | anywhere the front camera sees it whole | lens views only |

What does matter:

- **print scale**: the squares really 25 and 35 mm; every distance in the result comes from them;
- **flat and rigid** boards;
- **nothing moves during the scene capture**: no board shifts, the robot isn't bumped;
- **no glare** on the patterns.

Before any motion, a live frame from each camera checks the setup; a board cut off or hidden
gets moved then.

## Capture and solve

From `examples/lekiwi/gemma_brain`, robot connected. Each capture saves its frames and a JSON
file under `runs/<time>-<kind>/`. Only step 4 moves the robot, after the owner approves the plan.

1. **Wrist lens** (no motion; the robot isn't taken over):

       .venv/bin/python calibration/capture_cameras.py lens --cam wrist

   Hold any board in front of the wrist camera and move it slowly: near and far, every corner
   of the image, tilted every way. A view is kept when the board has been still for two frames
   and is somewhere new (position or size). Stops at 25 views (`--views`); Ctrl-C ends early.
2. **Front lens** (no motion): the same with `--cam front`.
3. **Place the boards**, check a live frame from each camera, then print the plan (moves
   nothing):

       .venv/bin/python calibration/capture_cameras.py scene --plan-only

   The 23 poses with their joint targets from the live calibration, or why one is refused.
4. **Scene capture** (arm motion, after approval; the base is locked):

       .venv/bin/python calibration/capture_cameras.py scene

   The arm folds and the front camera takes 3 frames; then the arm visits each planned pose (a
   joint move, 0.5 s to settle, a wrist frame and the measured joint angles) and folds at the
   end, even after an error.
5. **Solve** (offline):

       .venv/bin/python calibration/solve_cameras.py --lens-wrist runs/<t>-lens-wrist \
           --lens-front runs/<t>-lens-front --scene runs/<t>-scene

   Prints the results; `--write` stores them in `data/calib.json` under `"cameras"`, only after
   they're reviewed.

## The scene's arm poses

23 poses in joint space around three anchors. Each anchor is a TCP pose solved once; each pose
turns one joint (two for a height change), so every joint's zero is seen on its own and the
views rotate about several axes. Every move from the previous pose is checked for floor and
body clearance and for the joint limits less 1.5 deg.

| Anchor | TCP pose | Joint changes around it (deg) | Poses |
|---|---|---|---|
| down, 11 cm | over the floor board's centre, 11 cm above the floor, pointing straight down | none; wrist flex +/-12; wrist roll +/-45; pan +/-10; lift -6 with wrist flex +6 | 8 |
| down, 14 cm | the same, 14 cm up | the same | 8 |
| forward | 26 cm out, 18 cm above the floor, 30 deg down | none; wrist flex +/-6; pan +/-8; wrist roll +/-30 | 7 |

The anchors come from an offline plan check with the Oct 29 limits: higher vertical poses
(17-21 cm) are outside the arm's vertical reach, and "ready" is too high and too close to the
upright board to see it. `--plan-only` confirms the plan against the live calibration.

In simulation every pose sees the floor board and 7 also see the upright board, with a guessed
wrist camera mount (along the jaws, 6 cm behind the TCP). If the real camera misses the upright
board, the forward poses change.

## Results and checks

On simulated views of the same 23 poses (0.3 px corner noise; OpenCV 4.13 and 5.0):

| Quantity | Recovered within |
|---|---|
| joint zero offsets (lift, elbow, wrist flex) | 0.03 deg |
| wrist camera mount on the gripper | 0.5 mm / 0.09 deg |
| front camera pose in the arm's frame | 1.6 mm / 0.11 deg |
| wrist lens focal length | 1 px of 480 |

How it solves:

1. each lens: OpenCV's `calibrateCamera` over every board view, hand-held and scene;
2. the wrist camera's mount: a Park-Martin hand-eye solution, then a least-squares fit of every
   corner's reprojection that refines the mount, the boards' poses and the three joint offsets
   together (a NumPy Levenberg-Marquardt: no SciPy here, and OpenCV 5.0 has no
   `calibrateHandEye`);
3. the front camera: from its view of the upright board, whose pose step 2 found.

Before `--write`, look at:

- **reprojection rms**: under about 0.6 px for each lens and the scene; more points to a bent
  board, a board that moved, or a wrong print scale;
- **joint offsets**: a few degrees at most after the Oct 29 recalibration; more needs a second
  look;
- **front camera spread**: its frames agree within a few mm;
- **wrist mount**: physically sensible (the camera a few cm behind and beside the TCP).

After `--write`, the geometry switches to the measured cameras, wrist first then front, each
checked on logged frames against the current fitted calibrations. Re-measuring what depends on
them (floor, grasp target) is phase E.
