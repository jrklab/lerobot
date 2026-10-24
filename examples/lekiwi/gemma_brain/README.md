# gemma_brain: Gemma 4 drives LeKiwi

Gemma 4 (via Ollama on this Mac) turns a goal in plain words into calls to robot skills; the
skills do the perception and motion, closing every loop on a sensor; the LeKiwi web service
on the Pi plays the motions back with its own safety checks.

```
goal ──> brain.py (Gemma E4B, tool calls) ──> skills.py ──> arm.py (IK) / robot.py (HTTP)
                                                │                          │
                                       Gemma boxes + OpenCV       Pi: web_server/auto_control.py
                                       (front + wrist cameras)    (30 Hz playback, watchdog, stops)
```

## Pick and place, start to finish

**1. Power on LeKiwi** and wait ~2 minutes. The Pi starts the `lekiwi-web` service by itself at
boot (systemd, enabled, restarts on failure); its USB cameras can take up to ~90 s after that to
deliver frames.

**2. Connect the Ethernet cable** between the Pi and the Mac's USB LAN adapter (`en7`). Both ends
have fixed addresses (Mac `10.55.0.1`, Pi `10.55.0.102`), nothing to set up. Check:

```bash
curl -s -m 3 http://10.55.0.102:8000/auto/status | head -c 120; echo
```

It should print `{"mode":"gamepad",...`. If not: `ping 10.55.0.102`, then `ssh lekiwi 'systemctl status lekiwi-web'`.

**3. Start Ollama** on the Mac, in its own terminal window (leave it running):

```bash
ollama serve
```

(Or run `brew services start ollama` once, and it starts at every login from then on.)

**4. Check the model is there and load it** (the first load takes a few seconds; the brain
would also load it on its first call):

```bash
ollama list | grep gemma4:e4b          # if missing: ollama pull gemma4:e4b
ollama run --think=false gemma4:e4b "Reply with OK"   # loads it into memory, prints OK
```

**5. Set the scene.** The robot on the floor with ~1 m of room to turn and drive; the red brick on
the floor somewhere it can drive to; the short cup on the floor (here: to its right). Keep the
gamepad at hand: any stick input or the RB e-stop takes control away from the brain at once.

**6. Run it:**

```bash
cd ~/Work/lerobot/examples/lekiwi/gemma_brain
.venv/bin/python brain.py "Pick up the red block and drop it into the short cup on your right"
```

It prints each skill Gemma calls (`>>> pick(object='red block')`) and its result (`<<< OK: ...`),
and ends with `SUCCESS` or `NOT DONE` and the log folder. The web page's mode bar shows **Auto**
while it drives.

**7. Stop early:** Ctrl-C in that terminal, any gamepad input, or the gamepad's RB e-stop. The
robot also stops by itself within 0.6 s if the program dies. After a stop the arm holds where it
is; the gamepad's B button folds it back to rest.

Options: `--model gemma4:26b` (any Ollama model with vision and tools), `--no-think` (faster
planning, worse recovery from a failed step), `--robot http://<pi>:8000`. Every turn, frame and
number is logged under `runs/<timestamp>-brain/`.

First-time setup of this folder's venv (already done on this Mac):
`python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless`.

## Run

```bash
cd examples/lekiwi/gemma_brain
.venv/bin/python brain.py "Pick up the red block and drop it into the short cup on your right"
```

Options: `--model gemma4:26b` (any Ollama vision+tools model), `--no-think` (faster planning,
worse recovery), `--robot http://<pi>:8000`. Every turn, frame and number is logged under
`runs/<timestamp>-brain/`.

**Stopping:** Ctrl-C here, any gamepad input, or the gamepad's e-stop (RB). The robot also stops
by itself within 0.6 s if this program stops talking to it.

First-time setup of the Mac venv: `python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless`.

## Skills Gemma can call

| Skill | What it does |
|---|---|
| `scan(objects)` | find every listed object: from the **overhead camera** if one is attached and sees the whole robot, else by turning a full circle in 30° stops; candidates checked by a multiple-choice crop question to E4B |
| `approach(object)` | face the best scan candidate, drive until it's in arm reach (front camera), confirm it with the wrist camera; on a wrong candidate, try the next |
| `pick(object)` | approach, line the base up (wrist camera), roll the wrist to the brick's angle, centre at 10/6/3 cm, touch the floor, close, lift and check it's held |
| `place_in(container)` | gripper 3 cm above the rim, line up over the opening (arm, then base), open gently, check the object landed inside |
| `reset_arm()` | lift clear of the floor, then fold to rest |
| `finish(success, summary)` | stop |

## Overhead camera (optional)

A USB camera on the Mac looking roughly straight down at the floor makes `scan` faster
and more reliable than turning the robot around. It needs no calibration: each scan finds the
robot, backs it up 10 cm on its odometry, and finds it again; that move gives the robot's
heading and the scale (m per pixel) in the image. Moving or re-aiming the camera between scans
is fine. Requirements: the camera roughly straight down, the robot fully in its view with
~10 cm free behind it, and camera access granted to the terminal app (System Settings ->
Privacy & Security -> Camera). Without the camera, or if it can't see the whole robot, `scan`
falls back to the 360° turn by itself.

## Other scripts

`fetch_test.py` (one scripted drive-and-pick), `grasp_test.py` (pick from in front),
`calibrate_floor.py`, `wrist_calib.py` (the robot's calibrations in `calib.json`), `overhead.py` (overhead camera),
`test_arm.py` and `../web_server/test_auto_control.py` (offline tests).

## Known limits (2026-10-23)

- Few trials so far: pick 3/3 on the red brick after the roll/clearance fixes; the first
  place-in-cup missed (the brick rode off on the opening jaw; fixed, not yet re-run).
- Gemma's yes/no answers about these objects are unreliable; presence and success are
  checked geometrically instead (box position, the gripper's stall, image motion).
- Calibrations are for this robot, this floor, a 2x4 brick and a ~10 cm cup.
