# Camera calibration with ChArUco boards

One session with three printed ChArUco boards measures both cameras, with nobody holding anything:
the boards stay put and the robot moves (its arm, then its base). Phase D of the
measured-parameters work (branch `lekiwi-measured-params`); first run 2026-10-29.

What it measures (`data/calib.json` "cameras"):

- **each lens**: focal length, optical centre, distortion (wrist and front camera);
- **the wrist camera's mount**: its pose on the gripper (TCP frame);
- **the front camera's pose** in the arm's frame (it sits higher and tilted down compared with
  the stock design, so no model has it).

It can also fit the zero offsets of shoulder lift, elbow and wrist flex (`--fit-offsets`), but
doesn't by default: see [Results](#results-2026-10-29).

Tools, in this folder: `boards.py` (detection), `capture_cameras.py` (the capture),
`solve_cameras.py` (the solve). The boards: `targets/` (PDFs; `make_boards.py` regenerates them).
Tests: `tests/test_camera_calibration.py`.

## What you need

Three boards, each one Letter or A4 page from `targets/` (print the PDF, not the preview PNG).
Each board has its own marker IDs, so a camera can see several at once. Dictionary DICT_5X5_100.

| Board | Squares | Square / marker | Marker IDs | Pattern size | Where it goes |
|---|---|---|---|---|---|
| wrist | 10 x 7 | 25 mm / 18 mm | 0-34 | 25 x 17.5 cm | flat on the floor |
| front | 7 x 5 | 35 mm / 26 mm | 35-51 | 24.5 x 17.5 cm | upright, in front |
| side | 7 x 5 | 35 mm / 26 mm | 52-68 | 24.5 x 17.5 cm | upright, beside the front board |

Preparing them:

1. Print at 100% / actual size, no "fit to page".
2. Measure the 100 mm scale bar under each board: 99.5-100.5 mm is fine. Off by more: note the
   measured length (the solve can be scaled to it; no need to reprint).
3. Glue each print flat on stiff card (spread the glue evenly; a few mm of thickness or a black
   backing don't matter). A curled or bowed sheet bends the corner grid, and that error goes
   straight into the result.
4. Matte paper, or lamps off to the side: glare hides corners.

Any rotation of a board is fine (every corner is identified by its markers); only face down
isn't.

## Board placement

Distances from the front edge of the robot's body, straight ahead of the arm (the arm's base is
about 10 cm behind that edge). By eye: the solve measures every board's real pose.

```
                 TOP VIEW (robot facing up the page)

          ┌──── FRONT board ────┐ ┌──── SIDE board ────┐   both upright, facing the robot,
          └─────────────────────┘ └────────────────────┘   ~50-55 cm out, side by side;
                                                            the front camera sees both whole

                 ┌─────────────────────────────┐
                 │                             │
                 │      WRIST board (flat)     │   ~17 cm deep
                 │       25 cm across          │
                 │                             │
                 └─────────────────────────────┘   near edge ~6 cm from the robot
         ═══════════════════ robot front edge ═══════════════════
                         [ arm ]   [front camera]
                          LeKiwi
```

1. **Wrist board, flat on the floor**, printed side up, its 25 cm side left to right, centred on
   the arm, near edge ~6 cm from the front edge (its centre 15 cm out, where a brick is usually
   picked). Tape it down.
2. **Front board, upright**, facing the robot, ~50-55 cm from the front edge, roughly centred,
   bottom edge on the floor, propped so it can't tip. At ~32 cm its top was above the front
   camera's view (the camera sits low and tilts down).
3. **Side board, upright**, next to the front board, turned toward the robot. The front camera
   sees only about +/-30 deg, so 20-30 cm off to the side is outside its view.
4. **Keep ~30 cm clear behind the robot** for the base sweep.

Then check a frame from each camera (no motion): the front camera should see all three boards,
the front board whole; the wrist camera (arm folded) the floor board. Part of a board is fine
(a view counts from 6 corners); the front board whole matters most.

What matters more than position: the print scale, flat and rigid boards, no glare, and
**nothing moving during the capture** (no board shifts, nobody bumps the robot).

## Capture and solve

From `examples/lekiwi/gemma_brain`, robot connected and free to drive.

1. **The plan** (moves nothing):

       .venv/bin/python calibration/capture_cameras.py scene --plan-only

   Every arm pose's joint targets from the live calibration (or why one is refused), and the
   base sweep's moves.
2. **The capture** (arm and base motion, after approval; about 6 minutes):

       .venv/bin/python calibration/capture_cameras.py scene

   - the front camera takes 3 frames, the arm folded;
   - the arm visits 38 poses (35 deg/s joint moves, each checked for floor and body clearance;
     the base still): at each, a wrist frame and the measured joint angles;
   - the arm folds; the base sweep: at 0, 10 and 20 cm back, the base turns -20, -10, 0, +10,
     +20 deg and the front camera takes a frame at each of the 15 stops. Never forward (the
     floor board is 6 cm ahead); the robot ends 20 cm back, at heading 0.

   Frames and `scene.json` under `runs/<time>-scene/`.
3. **The solve** (offline):

       .venv/bin/python calibration/solve_cameras.py --scene runs/<t>-scene

   Prints the results; `--write` stores them in `data/calib.json` "cameras", after they're
   reviewed.

The hand-held lens capture (`capture_cameras.py lens --cam wrist|front`, then `--lens-wrist` /
`--lens-front` for the solve) is only a fallback for a camera the robot can't point at the
boards. Hold the board 30 cm or more away; closer, too few corners fit.

## The arm poses

38 poses in joint space around six anchors. Each anchor is a TCP pose solved once; each pose
turns one joint (two for a height change), so the views rotate about several axes. Every move
from the previous pose is checked for floor and body clearance and the joint limits less 1.5 deg.

| Anchor | TCP pose | Arm shape (lift / elbow / wrist flex, deg) | Changes around it (deg) | Poses |
|---|---|---|---|---|
| down 11 cm | over the floor board's centre, 11 cm up, straight down | 20 / 6 / 64 | wrist flex +/-12, roll +/-45, pan +/-10, lift -6 with wrist flex +6 | 8 |
| down 14 cm | the same, 14 cm up | 16 / 0 / 75 | the same | 8 |
| forward | 26 cm out, 18 cm up, 30 deg down, at the upright boards | -48 / 76 / 2 | wrist flex +/-6, pan +/-8, roll +/-30 | 7 |
| low | 3 cm past the board centre, 6 cm up, 50 deg down | 27 / 49 / -26 | wrist flex +/-8, pan +/-10 | 5 |
| far | 11 cm past the centre, 9 cm up, 70 deg down | 58 / -46 / 58 | the same | 5 |
| near | 9 cm short of the centre, 16 cm up, 70 deg down | -66 / 63 / 73 | the same | 5 |

The low, far and near anchors spread the arm's shapes, so a joint-offset fit can tell lift from
elbow (their predicted correlation -0.75 with the first three anchors, -0.34 with all six). The
lowest arm point is 35 mm above the floor.

## Results (2026-10-29)

Scene `runs/20261029T224146-scene`, written to `data/calib.json` (commit 149ea7b3).

| Quantity | Result |
|---|---|
| wrist lens | f 553 px; rms 0.41 px over 47 board views |
| front lens | f 550 px; rms 0.53 px over 43 views (the sweep: the optical centre now agrees within 8 px between lens models, 48 px apart without it) |
| wrist camera mount | 92 mm behind the TCP, 61 mm off the jaw axis, looking along the jaws; the same within 3 mm in two scenes |
| front camera pose | 82 mm ahead of the arm's base, 48 mm below it (~6 cm above the floor), 7.6 deg down; its 3 views agree within 0.3 mm |
| arm model fit | rms 3.8 px over 38 poses (2-6 px by arm shape: about 1-3 mm at the board) |

**The front camera is placed through the floor board.** The upright boards stand ~65 cm out in
the arm's frame and are located only from the forward poses; through them the front camera came
out up to 20 mm and 3 deg apart.

**Joint zero offsets are not applied.** The arm's zeros were set by eye at the Oct 29
recalibration. Fitted, the offsets came out small (lift +0.8, elbow -0.9, wrist flex -1.0 deg)
but moved up to 2.6 deg when one arm shape was left out: the rigid arm model misses 1-3 mm of
shape-dependent bending or play (worst in the low and near shapes), which the offsets can't
remove, and fitting them lowered the error only from 3.8 to 3.1 px. So the by-eye zeros are as
good as this method can tell. `--fit-offsets` reports them and their spread. Pan and wrist roll
offsets can't be measured here at all; for camera-guided moves their errors cancel.

Before `--write` on a new run, look at: lens rms under ~0.6 px; the wrist mount physically
sensible; the front camera's views agreeing within a few mm.

Next: the geometry switches to these cameras (wrist first, then front), each checked on logged
frames against the current fitted calibrations; then phase E re-measures what depends on them.
