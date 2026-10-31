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

**Since Oct 30 this is one part of `calibrate_all.py`**, which calibrates the arm's joints,
these cameras and the floor in one unattended run (the brain's README, "Calibrations"). The board
placement below is the same; in that run the floor board also takes some of the floor touches (its
thickness, `--board-thickness`, a ruler's), and it has to be readable by the front camera from
where the robot starts: nearer the robot rather than further (with its near edge 23-27 cm in
front of the arm the front camera read 6-26 of its corners; the run needs 6, 15 or more is
comfortable). Then, per run: the base sweep and the side board close-up only in the first scene
(the front camera's views don't depend on the arm's zeros, later solves reuse them), and the
front camera placed through the upright board if no start frame saw the floor board (less exact,
noted in the output).

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
file up; `--restore <backup> --write` undoes it. The offset register holds +/-2047 (11-bit
sign-magnitude), so a shifted offset is wrapped by a turn: this arm's elbow sits at 2014.
LeRobot's normalized units don't change (reading and range move together), so teleop and
recorded demos are unaffected; the degrees the planner and the Pi use are corrected.

Then redo what was measured through the arm's kinematics: the floor touches and the camera
calibration (its offsets should now come out within their spread). The named poses (rest,
ready, in degrees) move physically by the correction to their intended true angles. Pan and
wrist roll can't be corrected this way (the boards can't tell their zeros).

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

## The brain's geometry from these cameras (Oct 30)

The brain's geometry (`brain/geometry.py`, `brain/cameras.py`) works from these cameras instead of
fitted image-space values: the wrist camera's image motion, descent and one-look locate are
projections at any height and roll; the front approach turns the degrees and drives the metres
the camera model gives, about the base's measured turning centre (4.6 cm behind the arm frame's
origin, centred: `base_turn_centre`); horizon, view width and object heights likewise, on the
floor plane the cameras see (`floor_z`, the floor board's, -0.1058). Kept: the floor touches,
the taught grasp spot, and a brick's stopping row at the reach point (`front_camera` near_bottom).

Checked on logged runs before switching (predicting a still object's box after each logged move):

| | Camera model | Fitted values |
|---|---|---|
| front, turns (253) | 16 px median | 27 px (the turn rate 41% off: the camera 12.8 cm ahead of the turning centre) |
| front, drives near the reach point (134) | 3.3 px | 4.2 px |
| front, drives from far (99) | 23 px median, 70 at the 90th percentile | 27 px, 53 |
| wrist (37) | 14 px | 11 px (within half a millimetre at the brick; the model also through roll turns) |

The model's straight ahead matched the logs: every line-up after a fitted approach first moved
the base ~2 cm left (107 approaches); the model put the fitted column 3.1 cm left. And the
cup's height came out 5.8-5.9 cm, against the fitted formula's 4.3 cm (biased low by the same
parallax).

On the robot (Oct 30): 3/3 driven picks (74, 59, 58 s; the line-up's first sideways move -0.6,
+0.8, +1.0 cm, mean +0.4 against +2.1 before; two approaches needed no turn) and the full cup
task in 63 s, the brick centred in the cup, the cup measured 6.2 cm. Still weak: distance from
far away (one approach from 52 cm stopped 5 cm past its target; a pixel of the box's bottom is
~1 cm there); the line-up recovers it.

## Phase E: what depended on the cameras (Oct 30)

- **The grasp target: kept.** Through the camera model the taught brick sits where it physically
  should: its fixed-jaw edge 0.5 mm from the TCP across the closing direction, centred within
  2 mm, its top face level with the TCP within 1 mm. The spot is image-space and the camera
  rides the gripper, so neither the arm's recalibration nor the camera model changes it.
- **The floor under the board.** The cameras measured the floor board's top (z -0.1058); the
  robot works on the bare floor. The owner's ruler gave the board 1-1.5 mm, so the cameras'
  floor is now -0.1071 (`solve_cameras.py --floor-board-thickness 0.00125`, calib.json
  `floor_z`, `floor_board_thickness_m`). A touch on the board couldn't resolve this: the floor
  probe's 3 mm steps and run-to-run sag are as large as the board, and 1 mm steps were below
  the smallest move the Pi carries out (the arm didn't move).
- **The jaw-tip offset: 9.7 mm, measured.** The jaw tips are modelled as one point on the TCP's
  pointing axis (the URDF's gripper_frame_link); with the gripper straight down the offset is
  the TCP's height above the tips. The floor probe's TCP at contact (-0.0973 m at x 0.24) less
  the cameras' floor (-0.1071) gives 9.7 mm; the taught grasp agrees (the held brick's top
  face level with the TCP, the tips 1.5 mm up: ~10 mm). It was 15 mm, estimated against a floor
  measured before the joint-zero fix. Defined once, in `web_server/auto_control.py` (the Pi's
  floor guard), imported by `common/arm.py`.
- **Two floor-guard flaws the old offset hid.** With the tips where they are, the final look's
  descent (TCP 3 cm up) ends with the tips 2.03 cm up, 0.3 mm over the free-move clearance: a
  free move under the free-move guard. That guard was taken from the floor plane's height at
  x 0, off by the plane's tilt at the reach point; it now comes from the floor at the reach
  point, and a free move's guard never sits above its own lowest jaw tips (as joint moves
  already had).
- **Robot check** after the deploy: 3/3 driven picks (72, 61, 65 s), no move refused or stopped
  by the Pi, every grip square on the brick's 15.8 mm side (15.6-15.7 mm).

**The joint-zero correction on the real motors (Oct 30).** With the service stopped and the arm
folded at rest, a -5 deg elbow correction (`apply_joint_offsets.py --offsets arm_elbow_flex=-5
--write`), then `--restore` of the backup:

| | before | after the write | after the restore |
|---|---|---|---|
| `homing_offset` | 2014 | -2025 (2014 + 57 ticks, past the register's 2047: wrapped by a turn) | 2014 |
| `range_min` / `range_max` | 914 / 3129 | 857 / 3072 (-57) | 914 / 3129 |
| elbow reading | 82.71 deg (held) | 89.47 deg | 94.48 deg |

- The motor took the wrapped offset; offset and position limits went to the motors and the
  file together, read back equal each time (`is_calibrated`), and the service started both
  times without LeKiwi's calibration prompt.
- The shift itself, measured on the restore, with the arm limp and unmoved: 89.47 -> 94.48 deg,
  +5.01 deg for 5. (The write's reading includes the folded arm sagging when its torque went
  off, to ~94.5 deg as at any restart: 94.5 - 5 = 89.5.)
- After the restore the calibration file is byte-for-byte the original. Both backups stay on the
  Pi (`kiwi_sn_0.json.before-zero-fix-*`, named by the Pi's own clock).

Not tested end to end: a capture finding a real zero error, the correction, and a capture
confirming it (this arm's zeros are within the method's resolution, so nothing is applied).
