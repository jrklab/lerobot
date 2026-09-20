# LeKiwi Controls — Web App, Gamepad, Leader Arm & Quest VR

How to drive LeKiwi via the web app (phone or computer browser), a Bluetooth/USB gamepad, a
leader arm + keyboard, or a Meta Quest 2. All drive the same robot through the same server
process — see `lekiwi_web_server.md` for the overall project spec and `networking_setup.md`
for how the Pi's network is set up.

## Control mode selector

The web app has a **Control** selector (top of the page, under the status bar) with three
options: **Gamepad**, **Web joystick/buttons**, and **Leader arm + keyboard**. Only one is
ever active at a time — the server ignores commands from any source that isn't the currently
selected mode, so e.g. moving the on-page joystick while "Gamepad" is selected does nothing.
The inactive mode's on-page controls (joystick, jog buttons, speed selector) gray out to make
this obvious. The mode is shared across every connected browser tab/device — switching it on
one screen switches it everywhere.

**Reset arm to neutral** and the gamepad's emergency stop remain available regardless of
mode. Switching *into* Leader arm + keyboard mode ramps the arm smoothly from its current
pose to the leader arm's live pose over ~1.5s instead of snapping there instantly, in case
the leader isn't already in a matching pose when you switch (see `robot_bridge.py`'s
`LEADER_CATCHUP_S`).

## Starting the server

On the Pi (or host PC with `--mock`, no hardware needed):

```bash
uv run python examples/lekiwi/web_server/server.py --port 8000
```

Useful flags:
- `--mock` — use a simulated robot (no hardware), for UI development/testing.
- `--no-gamepad` — disable gamepad auto-detection (on by default).
- `--robot-id` / `--port-name` — override the LeKiwi calibration id / serial port
  (defaults: `kiwi_sn_0`, `/dev/ttyUSB0`).
- `--repo-id` — Hugging Face dataset repo episodes are recorded into (default:
  `jrkhf/lekiwi_recordings`; one dataset per deployment).
- `--dataset-root` — local directory for the recorded dataset (default:
  `$HF_LEROBOT_HOME/<repo-id>`).
- `--no-recording` — disable the Record tab entirely (enabled by default).

## Web app

Open in a browser:
- From the host PC (dev/management path): `http://10.55.0.102:8000`
- From a phone/tablet/laptop connected to the `LeKiwi-RoboNet` Wi-Fi hotspot (password
  `lekiwi_test`): `http://10.42.0.1:8000`

### Base tab
Two joysticks (drag with a finger or mouse):
- **Translate** (left) — forward/back and strafe left/right.
- **Rotate** (right) — spin in place.

### Arm tab
Six explicit per-joint hold-to-jog controls, grouped base-to-tip:

| Control | Joint | Buttons |
|---|---|---|
| Base rotation | `shoulder_pan` | ← / → |
| Shoulder | `shoulder_lift` | ↓ / ↑ |
| Elbow | `elbow_flex` | ↓ / ↑ |
| Wrist tilt | `wrist_flex` | ↓ / ↑ |
| Wrist twist | `wrist_roll` | ⟲ / ⟳ |
| Gripper | `gripper` | close / open |

Press and hold a button to jog that joint continuously; release to stop wherever it is
(no snap-to-open/closed). **Reset arm to neutral** (above the joint grid) sends the arm to
its folded rest pose in one action.

### Speed selector
Slow / Medium / Fast, applies to whichever tab (Base or Arm) is active.

### Record tab
Records episodes in the standard LeRobot dataset format (`LeRobotDataset`) — the same
format `lerobot-record` produces, usable for training/visualization with the rest of the
lerobot ecosystem. Recording works **regardless of which Control mode is driving the
robot** — gamepad, web, or leader arm — it just captures whatever's actually happening.

- Type a task description, then **● Record**. Click **■ Stop** to save the episode, or
  **Discard** to throw it away without saving.
- Each saved episode appears in the list with a **▶ Play** button — playback takes
  exclusive control of the robot for the duration of the replay (temporarily overriding
  whichever Control mode is selected, restored automatically once it finishes or if you
  click **■ Stop** on it), then hands control back.
- **↑ Upload all to Hub** pushes every recorded episode to the configured `--repo-id`.
  This only works while the Pi has an internet route — normally only when Ethernet is
  connected to the host PC (the Pi's own hotspot mode, used for cordless ground testing,
  has no upstream route) — the **internet: connected/disconnected** badge next to the
  episode list shows this. Episodes are safe on local disk regardless; upload is a
  deliberate, retryable action, not automatic.
- One-time setup on the Pi before uploading works: `uv run hf auth login` (needs a
  Hugging Face token with write access, and Ethernet connected to verify it).

## Gamepad (Bluetooth or USB)

### One-time Bluetooth pairing (on the Pi)
```bash
bluetoothctl
> power on
> scan on
> pair <MAC>
> trust <MAC>
> connect <MAC>
```
`trust` is what makes this a one-time step — once trusted, the Pi reconnects the controller
automatically whenever it's powered on and in range, no need to re-run `bluetoothctl`. The
server itself also auto-detects the controller the moment it's connected (checks every 2s
for a controller if one isn't already attached) — nothing to configure or restart.

**Gotchas hit during initial setup (already fixed on this Pi, documented in case a fresh
Pi image needs the same fix):**
- `connect` failed with `org.bluez.Error.Failed br-connection-create-socket`, and the
  journal (`sudo journalctl -u bluetooth`) showed
  `profiles/input/device.c:control_connect_cb() ... Permission denied (13)`. Fixed with a
  systemd drop-in adding `CAP_NET_RAW` (needed for the HID control/interrupt L2CAP
  sockets), which was missing from `bluetooth.service`'s `CapabilityBoundingSet`:
  ```bash
  sudo mkdir -p /etc/systemd/system/bluetooth.service.d
  echo '[Service]
  CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_BIND_SERVICE CAP_NET_RAW
  AmbientCapabilities=CAP_NET_ADMIN CAP_NET_BIND_SERVICE CAP_NET_RAW' | \
    sudo tee /etc/systemd/system/bluetooth.service.d/override.conf
  sudo systemctl daemon-reload && sudo systemctl restart bluetooth
  ```
- That alone didn't fix it, though — the actual fix was `bluetoothctl remove <MAC>` and
  re-pairing from scratch (a stale/incomplete bonding record from an earlier attempt was
  the real cause). If pairing succeeds but `connect` keeps failing the same way, try
  removing and re-pairing before assuming it's a deeper system issue.

### Mode toggle
The gamepad has two top-level modes for what the two analog sticks control, plus a sub-mode
within Arm mode. Only one is ever active at a time — switching releases whatever was
previously being driven.

- **Base mode** (default on startup): left stick = translate, right stick X = rotate.
- **Arm mode**: has two sub-modes, see below.

**A button** — toggle between Base mode and Arm mode.
**"-" / "+" buttons** (`BTN_TL`/`BTN_TR`) — toggle Arm mode's sub-mode: **Joint** (default)
or **Cartesian/IK**. Harmless to press from Base mode too (it just changes which sub-mode
Arm mode will be in next time you switch to it).

### Motion mapping — Base mode
| Control | Motion |
|---|---|
| Left stick, push up/down | Drive forward / backward (`x.vel`) |
| Left stick, push left/right | Strafe left / right (`y.vel`) |
| Right stick, push left/right | Rotate in place (`theta.vel`) |

Directions follow directly from the code (`x = -stick_y`, `y = -stick_x`, `theta = -right_stick_x`),
verified on hardware.

### Motion mapping — Arm mode, Joint sub-mode
| Control | Joint | Direction |
|---|---|---|
| Left stick, left/right | `shoulder_pan` | stick right → pan positive |
| Left stick, up/down | `shoulder_lift` | stick up → arm raises |
| Right stick, up/down | `elbow_flex` | stick up → elbow folds in |
| Right stick, left/right | `wrist_flex` | stick right → wrist tilts down |
| Y | `wrist_roll` + | |
| X | `wrist_roll` − | |
| LT | `gripper` open (proportional — press further = faster) | |
| RT | `gripper` close (proportional) | |

Same caveat as base mode — these directions come from the code's sign conventions
(`gamepad_input.py`'s `_dispatch_arm()`), inherited from the original
`gamepad_teleoperate.py` mapping, but not yet re-confirmed by hand on this specific unit
since the code changed. Verify on hardware and flip any sign that feels backwards.

### Motion mapping — Arm mode, Cartesian/IK sub-mode
Drives the end-effector's position and orientation directly (via `so101_kinematics.py`'s
inverse kinematics) instead of jogging individual joints. **Entering this sub-mode first
ramps the arm to a fixed, more open mid-range pose over ~1.5s** (all 5 arm joints at
normalized 0) rather than jogging from wherever it was — starting cartesian jogging from a
deeply-folded pose needs disproportionately large joint swings for small hand motions, which
looks like a bug but isn't (see `robot_bridge.py`'s `CARTESIAN_READY_POSE`). Jog input is
ignored while this ramp is in progress.

| Control | Motion | Frame |
|---|---|---|
| Left stick, up/down | Translate x | Arm base frame |
| Left stick, left/right | Translate y | Arm base frame |
| Right stick, up/down | Translate z | Arm base frame |
| D-pad, left/right | Roll | End-effector's own axes |
| D-pad, up/down | Pitch | End-effector's own axes |
| Right stick, left/right | Yaw | End-effector's own axes |
| LT | `gripper` open (proportional) | |
| RT | `gripper` close (proportional) | |

Rotation is about the end-effector's *own* axes (e.g. "roll" always spins the gripper about
its own pointing direction, whatever direction that currently is), not the arm's fixed base
frame. Y/X (`wrist_roll` direct jog in Joint sub-mode) are inert here — `wrist_roll` is part
of the IK chain in this sub-mode, so driving it directly at the same time as the IK solve
would fight over the same joint every tick.

The arm has only 5 joints, so it cannot hit an arbitrary position *and* orientation target
simultaneously (a 6-value target from a 5-value system is rank-deficient by one degree of
freedom) — position keeps priority if the two conflict, so heavy simultaneous
translate+rotate input can make rotation lag a little. This is an inherent hardware limit,
not a bug.

Jog speeds (`robot_bridge.py`'s `CARTESIAN_JOG_SPEEDS` / `CARTESIAN_ROT_JOG_SPEEDS`):

| Speed | Translation | Rotation |
|---|---|---|
| Slow | 0.05 m/s | 1.5 rad/s |
| Medium | 0.12 m/s | 3.0 rad/s |
| Fast | 0.22 m/s | 5.0 rad/s |

### Always-active controls (any mode/sub-mode)
| Input | Action |
|---|---|
| LB | Cycle speed: slow → medium → fast |
| RB (hold) | **Emergency stop** — zeroes all motion immediately. Does not move the arm to any position, just halts it in place. |
| B | Reset arm to neutral pose |

Select/Start would have been the more conventional buttons for speed-cycle/e-stop, but on
this specific controller unit they're dead — confirmed via `probe_gamepad.py` to produce no
evdev event at all — so LB/RB are used instead.

### Verified button/axis mapping
Confirmed one control at a time via `examples/lekiwi/probe_gamepad.py` against the actual
paired controller — this differs from the SHANWAN reference mapping in the original
`gamepad_teleoperate.py`'s docstring (different right-stick axes and face-button codes):

| Physical control | evdev code |
|---|---|
| Left stick | `ABS_X` / `ABS_Y` |
| Right stick | `ABS_RX` / `ABS_RY` |
| D-pad | `ABS_HAT0X` / `ABS_HAT0Y` (roll/pitch in Cartesian/IK sub-mode; unused in Joint sub-mode) |
| Y | `BTN_C` |
| X | `BTN_NORTH` |
| A | `BTN_B` |
| B | `BTN_A` |
| LB | `BTN_WEST` |
| RB | `BTN_Z` |
| LT | `ABS_Z` (proportional) |
| RT | `ABS_RZ` (proportional) |
| "-" / "+" (arm sub-mode toggle) | `BTN_TL` / `BTN_TR` |
| Home | `KEY_MENU` (not currently used) |
| Select / Start | dead buttons on this unit — no event at all |

## Leader arm + keyboard

For precise teleoperation: an SO101 leader arm mirrors directly onto LeKiwi's arm, and the
keyboard drives the base. Unlike the gamepad (paired directly to the Pi), **the leader arm
and keyboard connect to your own workstation** (host PC), not the Pi — a separate script
there talks to the already-running web server over the network:

```bash
uv run python examples/lekiwi/web_server/leader_keyboard_client.py \
    --server ws://raspberrypi.local:8000/ws/control \
    --leader-port /dev/ttyACM0 --leader-id leader_arm_1
```

This needs a real desktop session on the machine you run it from (keyboard capture uses
`pynput`, which requires `DISPLAY` on Linux) — run it on your workstation, not the headless
Pi. Running the script alone doesn't drive anything by itself: you still need to select
**Leader arm + keyboard** in the web app's Control selector for its commands to take effect.

### Base key mapping
Same keys as `examples/lekiwi/teleoperate.py`'s reference mapping (`LeKiwiConfig.teleop_keys`):

| Key | Action |
|---|---|
| W / S | Drive forward / backward |
| A / D | Strafe left / right |
| Z / X | Rotate left / right |
| R / F | Cycle speed up / down (slow → medium → fast) |
| B | Toggle torque feedback on the leader arm ON/OFF (starts disabled) |

The script also prints a line whenever the server confirms Leader arm + keyboard mode is (or
stops being) active, so it's clear whether anything you do here currently has an effect —
after connecting, it otherwise runs quietly (no per-tick logging).

### Torque feedback
Same behavior as `examples/lekiwi/teleoperate.py`: when enabled, the leader arm resists more
as the corresponding follower joint's load rises (using the follower's live load/speed,
fetched from the server's periodic state broadcast rather than a local reading). Same
recommended per-motor scales/thresholds as the reference script.

### Safety notes
- If the gamepad disconnects (Bluetooth drop, battery, etc.) mid-use, the server detects
  the silence and halts all motion automatically rather than continuing on the last
  received stick position.
- Only the selected Control mode's commands take effect; the other two are fully inert
  (not just visually grayed out) while a different mode is active.

## Quest 2 VR

A 4th control mode, driving the arm's end-effector (position + orientation) from a Quest 2
controller's tracked pose, with the headset displaying the `front`/`wrist` camera feeds (the
same MJPEG streams the web app's Arm/Base tabs already use) on two flat panels — both shown
at once, side by side, unlike the flat web app which only streams whichever tab is active —
in a WebXR session.

Has two top-level modes, toggled by the **right thumbstick click**:
- **Motor Control mode** (default on entry) — direct joint/base jogging, mirroring the
  gamepad's controls exactly (`gamepad_input.py`'s `_dispatch_base()`/`_dispatch_arm()`), just
  split across two hands instead of one gamepad's two sticks. Has its own base/arm sub-mode,
  toggled by the **left thumbstick click** (mirrors the gamepad's joint/cartesian toggle).
  Needs no dedicated server-side logic at all -- it just sends the same `"jog"`/`"base"` WS
  messages the gamepad and flat web app already use, with `source: "vr"`.
- **IK mode** — absolute end-effector pose + clutch (see below). The original VR control
  scheme, still available but no longer the default since direct jogging turned out to give
  more reliable control while the IK tuning is still being refined.

### Opening it on the headset
WebXR requires a "secure context" (HTTPS, or `localhost`) — the plain `http://...:8000`
address the flat web app uses does **not** qualify, so `/vr` is served on a separate HTTPS
port instead: **`https://<pi-ip>:8443/vr`**. Port 8000 (plain HTTP) is unaffected and still
serves the regular flat app for phone/laptop browsers.

Two ways to reach it, depending on which network the Quest is on (see
`networking_setup.md`'s Step 5 for how the second path was set up):

- **Quest on the `LeKiwi-RoboNet` hotspot** (always available): `https://10.42.0.1:8443/vr`
  or `https://raspberrypi.local:8443/vr`.
- **Quest on the home Wi-Fi** (only when the Pi's USB Wi-Fi dongle is plugged in and
  connected to it): `https://192.168.1.180:8443/vr` (this IP comes from the home router's
  DHCP and may change — check `ip -4 addr show wlan1` on the Pi if it stops responding). Lets
  the headset stay on the home network instead of switching onto the isolated hotspot, and as
  a side effect gives the Pi a real internet route (Hub uploads work whenever this link is
  up). If the dongle is removed, this path simply stops working — the hotspot path above is
  unaffected either way, and the Pi's own startup isn't affected by the dongle's presence or
  absence at all (see `networking_setup.md`).

The cert is self-signed (generated once on the Pi, `examples/lekiwi/web_server/certs/`,
gitignored, with both IPs above in its `subjectAltName` list), so the Quest Browser will show
a certificate warning the first time on each network — tap **Advanced** → **proceed
(unsafe)** (wording varies by browser version). This is a one-time step per device per
network path; the browser remembers the exception afterward. If `navigator.xr` still isn't
available after that, check the page's own status line (split into a WebXR line and a
WebSocket line specifically so a "not supported" error can't get silently overwritten by an
unrelated "Connected" message) for the actual reason.

### Why absolute + clutch, not jogging
The gamepad drives the arm by *rate* (hold a stick, it keeps moving). A hand-tracked
controller instead drives by *absolute pose* — the end-effector tracks your hand's motion
directly, like a leader arm. Since your hand's reachable volume doesn't match the arm's
workspace, this needs a clutch, same idea as lifting a mouse to reposition it:

- **Grip held down** = enabled. On the press, latch two references: the controller's current
  pose, and the arm's current end-effector pose.
- While held, target pose = latched EE pose + (current controller pose − latched controller
  pose) — the *delta* your hand has moved, not its absolute position.
- **Release grip** = arm freezes in place. Squeezing again re-latches wherever your hand
  physically is now.

### Button mapping
Right controller always drives the arm; left always drives the base/e-stop. Not
configurable. Buttons that are always active, regardless of mode/sub-mode (mirrors the
gamepad's "always-active controls"):

| Control | Action |
|---|---|
| Left grip (hold) | Emergency stop. Not a trigger, because Motor Control's arm sub-mode needs both triggers for the gripper -- grip is unused everywhere else, so it works identically and always |
| A (right controller) | Cycle speed: slow → medium → fast |
| B (right controller) | Reset arm to neutral pose |
| Right thumbstick click | Toggle IK mode ↔ Motor Control mode |
| Left thumbstick click | Toggle Motor Control's base/arm sub-mode (no effect in IK mode) |

**Motor Control mode:**

| Control | Base sub-mode | Arm sub-mode |
|---|---|---|
| Left stick | Translate (forward/back/left/right) | `shoulder_pan` / `shoulder_lift` |
| Right stick | Rotate (X axis) | `wrist_flex` / `elbow_flex` |
| Left Y / X | -- | `wrist_roll` +/− |
| Left trigger / Right trigger | -- | Gripper open / close (rate, like the gamepad -- not absolute like IK mode's gripper) |

**IK mode:**

| Control | Action |
|---|---|
| Grip (right controller) | Hold = enable arm tracking (clutch); release = freeze in place |
| Trigger (right controller) | Analog position → gripper position: released = open, fully squeezed = closed |
| Left stick | Drive the base: forward/back/left/right |
| X (left controller) | Rotate base left |
| Y (left controller) | Rotate base right |

Speed affects two different things depending on which control: for the base and Motor
Control's joint jogging it's a rate (m/s or normalized-units/s, same `BASE_SPEEDS`/
`JOG_SPEEDS` as every other control mode); for IK mode's arm it's a **motion scale**
(`robot_bridge.py`'s `VR_MOTION_SCALE`) applied to hand motion, since that control is
absolute-pose tracking, not rate-based jogging — "fast" (1.0) is full 1:1 hand tracking,
"slow"/"medium" deliberately under-track hand motion (0.3x/0.6x) for finer control. There's
no in-headset display of the current mode/speed yet (the status lines showing them live on
the 2D overlay, only visible before entering VR / after taking the headset off) — toggle and
cycle by feel and watch the robot's response.

Axis mapping from the controller's WebXR-frame pose to the arm's base frame
(`robot_bridge.py`'s `WEBXR_TO_ARM_FRAME`, IK mode only), rotate directions (X/Y → left/right
in IK mode; Y/X → wrist_roll +/− in Motor Control), and the camera panels' placement/size
(`vr.js`) are first guesses, not yet confirmed by hand on the headset — expect to retune by
feel, same as the gamepad's sign conventions were.

### Stall haptics
The right controller (the one driving the arm in both modes) vibrates when an arm joint is
detected as stalled (per `robot_bridge.py`'s existing `_stall_severity()` -- same load/speed
thresholds the flat web app's per-joint load readout already uses), at an intensity matching
the worst-stalled joint's severity. Needs no server-side changes -- the periodic `"state"`
broadcast every client already receives (every 200ms) already carries this; the VR client
just wasn't listening to incoming messages before.

## Troubleshooting

### Camera feed freezes mid-session (joint load/position readout freezes too)
A USB UVC camera can wedge -- its background read thread blocks inside the OS/driver, or
keeps delivering a frame whose timestamp never advances -- without `send_action()` ever
noticing, so arm/base motor control keeps working throughout. `LeKiwi.get_observation()`
reads motor state *then* cameras in the same call, so a wedged camera also freezes the
joint-state/load readout in the web UI, not just the video.

This now self-heals: if `get_observation()` keeps failing for more than
`robot_bridge.py`'s `CAMERA_WATCHDOG_TIMEOUT_S` (2s), `RobotBridge` disconnects and
reconnects every camera on a background thread (so arm/base control isn't stalled during the
reconnect, which can itself take several seconds). `CAMERA_RECONNECT_COOLDOWN_S` (8s) caps
how often it'll retry, so a genuinely-dead camera doesn't get hammered. Watch
`sudo journalctl -u lekiwi-web.service` for `"attempting reconnect"` / `"reconnected"` lines
to confirm it's working. A full Pi reboot should no longer be necessary for this specific
failure mode -- if it still is, that's worth reporting, since it'd mean this recovery path
isn't actually fixing whatever state the camera/driver is stuck in.

### "LeKiwi is hanging" -- web app reachable but the robot doesn't move or update
Check `sudo journalctl -u lekiwi-web.service --since '10 min ago'` for a wall of repeating
`ConnectionError: ... [TxRxResult] Port is in use!` -- if every `send_action`/`get_observation`
call is failing identically, forever, the service itself won't crash (it catches these per
tick and keeps looping), so systemd shows it as healthy ("active/running") even though the
robot is completely unresponsive.

Root cause (confirmed via `dmesg`, once): the USB-to-serial adapter for the motor bus
physically disconnected and reconnected (loose cable, brief power blip, etc.) --
`ch341-uart ttyUSB0: ... urb stopped` followed by `USB disconnect` then the same device
re-enumerating as `ttyUSB1`. The running process was still holding a handle to the now-gone
`/dev/ttyUSB0` node. A plain restart wouldn't have been enough on its own, since the config
still pointed at the old, now-nonexistent path.

Fix applied: the systemd service's `--port-name` now points at the stable udev alias
(`/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`, found via `ls /dev/serial/by-id/`)
instead of a raw `/dev/ttyUSBx` path, so it keeps working across future re-enumeration
events regardless of which number the kernel assigns. If this happens again anyway (e.g. a
different USB adapter with its own by-id name), `sudo systemctl restart lekiwi-web.service`
after first confirming (`ls /dev/serial/by-id/`) the service's configured port still matches
where the device actually landed.
