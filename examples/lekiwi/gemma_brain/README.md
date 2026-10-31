# gemma_brain: Gemma 4 drives LeKiwi

Say a task in plain words; Gemma 4 26B (llama.cpp on this Mac) plans it once from the two
cameras; code runs each step on a small set of measured skills that close every loop on a
sensor; the LeKiwi web service on the Pi plays the motions back with its own safety checks.

```
task ──> plan_brain.py ──> brain/task_loop.py ──> brain/chains.py ──> brain/actions.py ──> brain/primitives.py
          26B: one plan     check, run, retry,     pick, place in,     approach, land,       find, servo,
          from 2 frames     fold, replan, check    place next to       carry, release...     move, grip
                                                                                                │
                              common/arm.py (IK) / common/robot.py (HTTP) ──> Pi: web_server/auto_control.py
                                                                              (30 Hz playback, watchdog, stops)
```

The skills have three layers (design doc "v2 skills: three layers"): primitives
(`brain/primitives.py`: find, servo, move, grip), goals (`brain/goals.py`: what "done" looks
like in an image), and actions and chains (`brain/actions.py`, `brain/chains.py`), which hold no
numbers of their own. Every number is in `brain/params.py` with its source (CALIB, DEMO, SPEC or
RULE) or measured into `data/calib.json` (`brain/geometry.py`). v1 (one tool call per skill) was
retired on 2026-10-28; it is at the git tag `lekiwi-v1-final`.

## Quick start: a spoken task

With the robot on and connected, and the two model servers running (steps 1-3 below):

```bash
curl -s http://127.0.0.1:8081/health; curl -s http://127.0.0.1:8082/health   # each: {"status":"ok"}
curl -s -m 3 http://10.55.0.102:8000/auto/status | head -c 60; echo          # the robot: {"mode":...

cd ~/Work/lerobot/examples/lekiwi/gemma_brain
.venv/bin/python plan_brain.py "put the red block in the paper cup"
.venv/bin/python plan_brain.py "put the red block next to the green block"
.venv/bin/python plan_brain.py --holding "red block" "put the red block in the paper cup"   # already holding it
.venv/bin/python plan_brain.py --plan-only "put the red block in the paper cup"           # plan and check only: nothing moves
```

It prints the plan, each step and its result, and ends with `SUCCESS` or `NOT DONE` and the log
folder (`runs/<time>-plan/`). Stop with Ctrl-C, any gamepad input, or the gamepad's RB e-stop;
the B button then folds the arm. A task can pick, place in, and place next to; other relations
("on the plate") are refused by the plan check before anything moves.

## Folders

| | |
|---|---|
| `plan_brain.py` | say a task (entry point) |
| `run_skills.py` | one chain on the robot, or driven pick trials, without the planner |
| `brain/` | the skills: params, geometry, primitives, goals, actions, chains, the task loop |
| `common/` | shared by everything: the arm model and planner, the robot client, the models, vision, the run log |
| `calibration/` | scripts that measure this robot into `data/calib.json`; the demo extractor |
| `tools/` | robot checks: the link (`check_link.py`), the front view with the arm carried (`check_carry_view.py`) |
| `data/` | `calib.json`, the floor touches, the demo extract (`data/demos/`) |
| `tests/` | offline tests by layer; `uv run pytest` |
| `runs/`, `demos/` | run logs and frames, raw teleop recordings (not in git) |

## A task, start to finish

**1. Power on LeKiwi** and wait ~2 minutes. The Pi starts the `lekiwi-web` service by itself at
boot (systemd, restarts on failure); its USB cameras can take up to ~90 s after that to deliver
frames.

**2. Connect the Mac to the robot**, one of:

- **Ethernet cable** between the Pi and the Mac's USB LAN adapter (`en7`): fixed addresses (Mac
  `10.55.0.1`, Pi `10.55.0.102`), nothing to set up. The fastest and steadiest link (~17 ms).
- **The robot's own Wi-Fi hotspot `LeKiwi-RoboNet`**: join it from the Mac's Wi-Fi; the Pi is at
  `10.42.0.1`. The Mac has no internet while on it (the models run locally). Its password:
  `ssh lekiwi 'sudo nmcli -s -g 802-11-wireless-security.psk con show LeKiwi-RoboNet'`.

The scripts find the robot on whichever link is up (`10.55.0.102`, `10.42.0.1`, then
`raspberrypi.local`). A shared office/home Wi-Fi is not recommended: replies took 0.2-1.5 s
(Oct 24), and the Pi stops any motion that goes too long without the Mac's heartbeat. Check the
link: `curl -s -m 3 http://10.55.0.102:8000/auto/status | head -c 120; echo` should print
`{"mode":"gamepad",...`; if not, `ssh lekiwi 'systemctl status lekiwi-web'`.

**3. Start the two model servers** (llama.cpp, `brew install llama.cpp`; GGUFs in
`~/models/gguf`), each in its own terminal; together they need ~22 GB, so close big apps:

```bash
llama-server -m ~/models/gguf/gemma-4-26B_gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --mmproj ~/models/gguf/gemma-4-26B_mmproj-F16.gguf --alias gemma4-26b --port 8081 \
  -ngl 99 --jinja -np 1 -c 4096 --swa-full --cache-ram 0 --image-min-tokens 70 --image-max-tokens 70
llama-server -m ~/models/gguf/gemma-4-E2B_gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/models/gguf/gemma-4-E2B_mmproj-F16.gguf --alias gemma4-e2b --port 8082 \
  -ngl 99 --jinja -np 1 -c 4096 --swa-full --cache-ram 0 --image-min-tokens 70 --image-max-tokens 70
```

26B plans, boxes objects it hasn't seen yet in a camera, and answers every front look within
0.5 m (`CLOSE_CHOICE_M`): it picks the named object ("the light green block") by comparing what's
in view. E2B answers the other repeat looks; its box must match the object's colour as sampled
from 26B's box (`common/vision.py` Look: no colour tables), else 26B is asked. Check both are up:
`curl -s http://127.0.0.1:8081/health; curl -s http://127.0.0.1:8082/health` (each prints
`{"status":"ok"}`).

**4. Set the scene.** The robot on the floor with room to turn and drive; the brick on the floor;
for a place, the cup or the other brick nearby. Keep the gamepad at hand: any stick input or the
RB e-stop takes control away at once.

**5. Run it:**

```bash
cd ~/Work/lerobot/examples/lekiwi/gemma_brain
.venv/bin/python plan_brain.py "put the red block in the paper cup"
.venv/bin/python plan_brain.py "put the red block next to the green block"
.venv/bin/python plan_brain.py --plan-only "..."      # plan and check it; nothing moves
```

It prints the plan, each step (`>>> pick "red block"`) and its result (`<<< OK: done`), and ends
with `SUCCESS` or `NOT DONE` and the log folder (`runs/<time>-plan/`: every event, frame and
number). Options: `--holding "green block"` (already in the gripper), `--no-base` (wheels
locked), `--no-fast` (no E2B), `--robot http://<pi>:8000`.

**6. Stop early:** Ctrl-C, any gamepad input, or the RB e-stop. After a stop the arm holds where
it is; the gamepad's B button folds it back to rest.

## What a task can say

The planner writes `pick "x"` and `place "x" in|next_to "y"`; the plan is checked before
anything moves.

| Step | Chain |
|---|---|
| `pick "x"` | look around until the front camera finds it, drive it to the reach point, hover, line the base up, roll the wrist to the brick's long side, centre at 10 cm then 3 cm, land on the floor, close, lift (3 tries) |
| `place "x" in "y"` | carry at the hover height, approach the container, measure its height side-on, centre over it from one wrist look, lower to the demo's release height, open |
| `place "x" next_to "y"` | carry, approach, line up beside it at 10 cm and 3 cm, land, open |

A failed step folds the arm and the planner replans from what's done and why it failed.

## Robot runs without the planner

```bash
.venv/bin/python run_skills.py pick                 # the pick chain on the red block (approach first)
.venv/bin/python run_skills.py place-in "red block" --to "paper cup"     # holding the brick
.venv/bin/python run_skills.py trials --n 5         # 5 driven picks: pick, put back, back up 40 cm, repeat
.venv/bin/python run_skills.py land                 # hover at the reach point, land, back up
```

## Calibrations (data/calib.json, this robot)

### All of it in one run: calibration/calibrate_all.py

The arm's joint zeros and ranges, the cameras and the floor, with no one touching the robot
(Oct 30, 8.4 min). The arm needs no calibration to start from: a brand-new arm works too.

    .venv/bin/python calibration/calibrate_all.py --board-thickness 0.00125 --check-only   # no motion
    .venv/bin/python calibration/calibrate_all.py --board-thickness 0.00125

Before it:

- **The boards** placed as in [the camera calibration](calibration/CAMERA_CALIBRATION.md#board-placement).
  The floor board stays on the floor for the whole run. The front camera has to read it (the
  checks print how many corners: 15 or more is comfortable; under ~10 its view of it comes and goes):
  if not, move it nearer the robot.
- **The arm folded at rest** on the base, nothing within its reach; ~30 cm clear behind the robot,
  the floor clear to the side board.
- **The Mac awake and, best, plugged in.** The script holds a no-sleep assertion itself
  (`caffeinate -w`), but on battery a closed lid or low charge still sleeps the Mac: every
  request stops at once and the Pi aborts the arm on the lost heartbeat.
- **No web page open on the Pi** during the run: its two video streams load the Pi.

It runs: checks (no motion); the coarse calibration (each joint's two ends probed at a low
torque limit, `self_calibrate.py`); scenes and zero corrections until none is larger than its
spread over two scenes (`capture_cameras.py`, `solve_cameras.py --fit-offsets`); the cameras
written; three floor touches. The run's check: at each touch on the floor board, the jaw tips'
height against the board's top the cameras saw there, **PASS within 3 mm** (Oct 30: within 1.2 mm).
The summary and the calibration from before the run go to `runs/<time>-calibrate_all/`; put the
old one back with `calibration/self_calibrate.py --state runs/<time>-calibrate_all/self_calibration.json restore`.

**On the Pi**, the web service starts on a provisional calibration when the motors don't match
the saved file (or there is none) instead of LeKiwi's keyboard prompt, which a service can't
answer (it crashed and restarted in a loop). It does so with `--allow-uncalibrated` (in the
lekiwi-web unit file since Oct 30) and on its own whenever it has no terminal. `/auto/status`
reports `arm_calibration`: `calibrated` or `provisional`. The planner (`common/robot.py`)
refuses to run on a provisional arm: run `calibrate_all.py` first.

### The parts

| What | Script | When |
|---|---|---|
| cameras (lenses, wrist mount, front pose, base turning centre, floor they see; joint zeros if needed) | `calibration/capture_cameras.py` + `solve_cameras.py` with the ChArUco boards: [setup](calibration/CAMERA_CALIBRATION.md); within `calibrate_all.py` | a camera moved, or an arm recalibration |
| floor height (where the jaws touch) | `calibration/calibrate_floor.py`; within `calibrate_all.py` | new floor or arm calibration |
| arm joint zeros and ranges | `calibration/self_calibrate.py` (bootstrap, probe, write, correct, restore), within `calibrate_all.py` | new arm, or zeros in doubt |
| jaw-tip offset (`web_server/auto_control.py` JAW_TIP_OFFSET_M, 6.3 mm) | the SO-101 URDF's jaw meshes: the fixed jaw's tip past `gripper_frame_link` (stock parts; it was 9.7 mm, fitted, Oct 30) | gripper parts changed |
| front stop row (`front_camera` near_bottom: a brick at the reach point) | from `calibration/calibrate_front.py` (Oct 27; the script is otherwise superseded) | reach point changed |
| grasp target | `calibration/teach_grasp.py` (the owner's midpoint teach) | gripper or wrist camera changed |
| gripper opening (mm against angle) | `calibration/calibrate_gripper.py` | gripper recalibrated |
| demo values (open width, lift, release, place view) | `calibration/demo_extract.py` on a teleop demo (writes `data/demos/`) | new demo |

Superseded by the camera calibration on Oct 30 and no longer read: the wrist image Jacobian
(`wrist_calib.py`), the descent model (`calibrate_descent.py`), the jaw rows (`calibrate_jaws.py`) and
the front camera's turn rate, range curve and horizon (`calibrate_front.py`). Their values stay in
`calib.json` for the record.

The arm's joint angles are (ticks - 2047) x 2pi/4096 from the homing tick (fixed Oct 27).

## Tests

```bash
uv run pytest                    # all offline tests (tests/), from this folder
uv run pytest tests/test_goals.py
```

| File | Covers |
|---|---|
| `test_geometry.py` | geometry, and the rules: every param names its source, no stray numbers above the primitives |
| `test_primitives.py` | find, move (in the web_server arm simulation), measured moves, settle, grip, base |
| `test_goals.py` | the goals, and servo converging with the true Jacobian off |
| `test_actions_chains.py` | land, lift, carry, look around, over a container, place checks, chain order |
| `test_task_loop.py` | the planner's plans, retries, replans, folds, against fake chains |
| `test_arm.py`, `test_vision.py`, `test_demo_extract.py` | IK and the Pi's checks, jaw geometry, demo keyframes |

Scripts that move the robot are never named `*_test.py` (pytest would run them);
`runs/` and `demos/` are not collected.

## Known limits (2026-10-30)

- Driven picks 5/5 in the last four batches (63-69 s each); the cup and next-to tasks work end
  to end (cup: 67.5 s). Small samples, not success rates.
- Place relations other than "in" and "next to" (on, left_of, ...) are refused by the plan
  check before anything moves.
- The fast model's box at 3 cm sometimes changes height between looks and the servo chases it.
- Choosing among several blocks by colour: the 26B chose the named one in 15 of 15 asks at
  0.4-0.5 m, 4 of 7 at ~1 m (hence the close-up looks). Touching blocks come back as one box (6
  of 6 asks), and close shades side by side (light and dark green) can't be told apart by the
  sampled colour after the choice. Keep blocks apart.
- The robot's web page streams both cameras at 15 frames/s (~1.4 MB/s, ~11 Mbit/s) while it's
  open; the brain itself moves ~40 kB/s from the Pi and ~11 kB/s to it during a task (status
  polls ~28 kB/s, frames 47 kB a look). Close the page during brain runs over Wi-Fi.
- The arm has 2-3 mm of play in its reach; servo steps correct it from the encoders.
- The place-in check is one front-camera look for the object on the floor: a miss hidden behind
  the container passes as placed.
- The Pi's power supply sags under load; give it a solid 5.1 V / 3 A supply. On Oct 28 the Pi
  ran at its temperature limit and the motor bus dropped (no status packets); restarting
  `lekiwi-web` brought it back.
