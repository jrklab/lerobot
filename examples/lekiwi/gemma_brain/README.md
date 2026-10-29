# gemma_brain: Gemma 4 drives LeKiwi

Say a task in plain words; Gemma 4 26B (llama.cpp on this Mac) plans it once from the two
cameras; code runs each step on a small set of measured skills that close every loop on a
sensor; the LeKiwi web service on the Pi plays the motions back with its own safety checks.

```
task ──> plan_brain.py ──> v2/executor.py (task loop) ──> v2/chains.py ──> v2/actions.py ──> v2/skills.py
          26B: one plan      check, run, retry,           pick, place in,    approach, land,     find, servo,
          from 2 frames      fold, replan                 place next to      carry, release...   move, grip
                                                                                                    │
                                    arm.py (IK) / robot.py (HTTP) ──> Pi: web_server/auto_control.py
                                                                      (30 Hz playback, watchdog, stops)
```

The skills have three layers (design doc "v2 skills: three layers"): primitives
(`v2/skills.py`: find, servo, move, grip), goals (`v2/goals.py`: what "done" looks like in an
image), and actions and chains (`v2/actions.py`, `v2/chains.py`), which hold no numbers of their
own. Every number is in `v2/params.py` with its source (CALIB, DEMO, SPEC or RULE) or measured
into `calib.json` (`v2/geometry.py`). v1 (one tool call per skill, `brain.py`, `skills.py`) was
retired on 2026-10-28; it is at the git tag `lekiwi-v1-final`.

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

26B plans and boxes objects it hasn't seen yet; E2B answers the repeat looks at an object
already found (its box must have the object's colour, else 26B is asked).

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
.venv/bin/python run_v2.py pick                 # the pick chain on the red block (approach first)
.venv/bin/python run_v2.py place-in "red block" --to "paper cup"     # holding the brick
.venv/bin/python run_v2.py trials --n 5         # 5 driven picks: pick, put back, back up 40 cm, repeat
.venv/bin/python run_v2.py land                 # hover at the reach point, land, back up
```

## Calibrations (calib.json, this robot)

| What | Script | When |
|---|---|---|
| floor height | `calibrate_floor.py` | new floor or arm calibration |
| wrist image Jacobian | `wrist_calib.py` | wrist camera moved |
| descent (camera above the jaw tips) | `calibrate_descent.py` (from pick logs) | wrist camera moved |
| jaw geometry | `calibrate_jaws.py` | gripper or wrist camera changed |
| front camera (turn, range, reach row) | `calibrate_front.py`: one brick 0.6-1 m ahead, clear floor | front camera moved |
| grasp target | `teach_grasp.py` (the owner's midpoint teach) | gripper or wrist camera changed |
| demo values (open width, lift, release, place view) | `demo_extract.py` on a teleop demo | new demo |

The arm's joint angles are (ticks - 2047) x 2pi/4096 from the homing tick (fixed Oct 27).

## Tests

```bash
uv run pytest test_*.py          # offline: geometry, servo sims, chains, the task loop, vision
```

Scripts that move the robot are never named `*_test.py` (pytest would run them);
`runs/` and `demos/` are not collected.

## Known limits (2026-10-28)

- Driven picks 5/5 in the last four batches (63-69 s each); the cup and next-to tasks work end
  to end (cup: 67.5 s). Small samples, not success rates.
- Place relations other than "in" and "next to" are run as "next to".
- The fast model's box at 3 cm sometimes changes height between looks and the servo chases it.
- The arm has 2-3 mm of play in its reach; servo steps correct it from the encoders.
- The Pi's power supply sags under load; give it a solid 5.1 V / 3 A supply.
