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

              ┌──── FRONT board ────┐     upright, facing the robot, ~50-55 cm out;
              └─────────────────────┘     the front camera sees it whole

                 ┌─────────────────────────────┐
                 │      WRIST board (flat)     │   ~17 cm deep, 25 cm across
                 └─────────────────────────────┘   near edge ~6 cm from the robot
         ═══════════════════ robot front edge ═══════════════════
                         [ arm ]   [front camera]           ║ SIDE board: upright, facing
                          LeKiwi            ~50-60 cm  →    ║ the robot's side, level with
                                                            ║ its centre (either side)
```

1. **Wrist board, flat on the floor**, printed side up, its 25 cm side left to right, centred on
   the arm, near edge ~6 cm from the front edge (its centre 15 cm out, where a brick is usually
   picked). Tape it down.
2. **Front board, upright**, facing the robot, ~50-55 cm from the front edge, roughly centred,
   bottom edge on the floor, propped so it can't tip. At ~32 cm its top was above the front
   camera's view (the camera sits low and tilts down).
3. **Side board, upright**, beside the robot, facing its side, level with its centre, ~50-60 cm
   from its side edge, on either side (the capture finds it). The floor between must be clear:
   the robot drives to 30 cm from it.
4. **Keep ~30 cm clear behind the robot** for the base sweep.

Then check a frame from each camera (no motion): the front camera sees the front board whole and
the floor board; the wrist camera (arm folded) the floor board. Part of a board is fine (a view
counts from 6 corners).

What matters more than position: the print scale, flat and rigid boards, no glare, and
**nothing moving during the capture** (no board shifts, nobody bumps the robot).

## Capture and solve

From `examples/lekiwi/gemma_brain`, robot connected and free to drive.

1. **The plan** (moves nothing):

       .venv/bin/python calibration/capture_cameras.py scene --plan-only

   Every arm pose's joint targets from the live calibration (or why one is refused), the base
   sweep's moves and the close-up.
2. **The capture** (arm and base motion, after approval; about 8 minutes):

       .venv/bin/python calibration/capture_cameras.py scene [--reverse]

   - the front camera takes 3 frames, the arm folded;
   - the arm visits 38 poses (35 deg/s joint moves, each checked for floor and body clearance;
     the base still): at each, a wrist frame and the measured joint angles;
   - the arm folds; the base sweep: at 0, 10 and 20 cm back, the base turns -20, -10, 0, +10,
     +20 deg and the front camera takes a frame at each of the 15 stops. Never forward (the
     floor board is 6 cm ahead);
   - back to the start, closed loop on the front camera's view of the boards against the run's
     first frame (turn, then drive, until within the base's own tolerances: 1 cm, 2 deg);
   - the side board close-up: a 90 deg turn right (then left if it isn't there) at 12 deg/s,
     driving in to 30 cm from it (closed loop on its measured distance, at most 15 cm a drive),
     frames at -15, 0, +15 deg, back out, turn back, back to the start.

   `--reverse` visits the arm poses in the opposite order. Frames and `scene.json` under
   `runs/<time>-scene/`. Needs the measured front camera in `calib.json` for the return and the
   close-up; on a first calibration the robot stays 20 cm back and skips the close-up.
3. **The solve** (offline), one or more scenes together:

       .venv/bin/python calibration/solve_cameras.py --scene runs/<t1>-scene runs/<t2>-scene ...

   Prints the results; `--write` stores them in `data/calib.json` "cameras", after they're
   reviewed. Runs in both pose orders (one with `--reverse`) average out the arm's play.

Also: `capture_cameras.py home --scene runs/<t>-scene` drives the robot back to where that scene
started; `capture_cameras.py closeup` runs only the close-up (frames under `runs/<t>-closeup/`,
for the solve's `--closeup`). The hand-held lens capture (`capture_cameras.py lens --cam
wrist|front`) is only a fallback for a camera the robot can't point at the boards.

## Correcting the arm's joint zeros

For an arm whose zeros were set less carefully (this one's are within the method's resolution,
see Results). The capture is the same; the solve fits the shoulder lift, elbow and wrist flex
zero offsets too and names those larger than their spread over the arm shapes:

    .venv/bin/python calibration/solve_cameras.py --scene runs/<t1>-scene runs/<t2>-scene --fit-offsets

    "to apply (larger than their spread): {'arm_elbow_flex': -9.0}"

Then, on the Pi, write them into the robot's own calibration, so teleop, the Pi and the planner
all read the corrected zeros:

    sudo systemctl stop lekiwi-web
    .venv/bin/python examples/lekiwi/web_server/apply_joint_offsets.py \
        --port-name /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 --offsets arm_elbow_flex=-9.0
    # the same with --write, once the printed change looks right: the arm goes limp, fold it first
    sudo systemctl start lekiwi-web

It moves each motor's Homing_Offset and position limits by the same ticks, in the motors and the
calibration file together (LeKiwi refuses to start quietly when they differ), after backing the
file up; `--restore <backup> --write` undoes it. Then capture and solve again: the offsets
should come out within their spread. Pan and wrist roll can't be corrected this way (the boards
can't tell their zeros).

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

Four scenes (runs 0, A and C in the usual pose order, B reversed) and one side board close-up,
solved together and written to `data/calib.json` (commit 2679a902).

| Quantity | Result |
|---|---|
| wrist lens | f 547 px, radial k1 k2; rms 0.41 px over 194 board views |
| front lens | f 549 px, radial k1 k2; rms 0.53 px over 171 views |
| wrist camera mount | 90 mm behind the TCP, 61 mm off the jaw axis, looking along the jaws |
| front camera pose | 4.3 cm ahead of the arm's pan axis, 0.7 cm to its left, 4.8 cm below the arm's mounting surface (~5-6 cm above the floor), 8.1 deg down; the four scenes within 2.7 mm of their mean, tilts 7.4-8.5 deg. The owner's ruler check agrees (the camera sits right below the arm's base) |
| arm model fit | rms 4.0 px over 152 poses (2-6 px by arm shape: about 1-3 mm at the board) |

**Repeatability.** Run to run, the arm returns to the same place very consistently: in the same
pose order the camera-seen TCP repeats within 0.55 mm and 0.2 deg (median), the encoder readings
within 0.09 deg. Approached from the other side (run B), the encoders read 1.6 deg apart for the
same commanded pose (gearbox backlash and the servos' stopping band: seen, so the solve accounts
for it), and the camera sees a further 1.6-1.8 mm and ~0.5 deg (up to 2.9 mm) that the encoders
don't: play past the output shafts (horns, brackets, links). That play is most of the 1-3 mm the
rigid arm model can't fit, and why the calibration combines runs in both orders.

**The lens model is radial k1, k2 only.** With the tangential terms and k3 the fit was 0.01-0.02
px better, but the optical centre wandered between runs (front: 15 px across, 8 px down; without
them 1.5 and 4.8 px). Residuals don't grow toward the image edges with two terms (front: 0.48 px
in the outer ring; wrist 0.50 against 0.39 px with the full model, a ring the jaws mostly block).

**The front camera is placed through the floor board.** The upright boards stand ~65 cm out and
are located only from the forward poses; through them the front camera came out up to 20 mm and
3 deg apart.

**Joint zero offsets are not applied.** The arm's zeros were set by eye at the Oct 29
recalibration. Fitted, the offsets came out small (lift +0.9, elbow -1.0, wrist flex -1.1 deg),
repeatable run to run within ~0.5 deg, but moved up to 2.7 deg when one arm shape was left out:
the play above, which the offsets can't remove. Fitting them lowered the error only from 4.0 to
3.3 px. So the by-eye zeros are as good as this method can tell. For an arm whose zeros are set
less carefully the fit does catch them: adding 5-10 deg errors to the recorded readings of a run,
the solve found each again within 0.01 deg (see Correcting the arm's joint zeros).

**The base.** The sweep's turns came out 3-8% short, more so turning left than right: wheel
slip or friction more than a wrong base radius (5.0 cm wheels and 12.5 cm base radius are
LeRobot's defaults, not measured); its drives were within 1-2%. The calibration doesn't depend
on them: it returns to its start closed loop on the cameras.

Before `--write` on a new run, look at: lens rms under ~0.6 px; the wrist mount physically
sensible; the scenes' front camera estimates agreeing within a few mm.

Positions in `calib.json` are in the arm model's frame (so101_kinematics, the URDF's
`base_link`): its origin is on the arm's mounting surface, 3.9 cm behind the pan axis, so the
front camera's T_base_cam reads x 82 mm there.

Since Oct 30 the brain's geometry (brain/geometry.py) works from these cameras instead of the fitted
image-space values, after checking both on logged runs (see the README's calibration table).
Next: a robot check of the switched geometry; then phase E re-measures what depends on the
cameras (the floor, the grasp target).
