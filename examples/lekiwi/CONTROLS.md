# LeKiwi Controls — Web App & Gamepad

How to drive LeKiwi right now, via the web app (phone or computer browser) and/or a
Bluetooth/USB gamepad. Both control the same robot through the same server process — see
`lekiwi_web_server.md` for the overall project spec and `networking_setup.md` for how the
Pi's network is set up.

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
The gamepad has two modes for what the two analog sticks control. Only one is ever active
at a time — switching modes releases whatever the previous mode was doing.

- **Base mode** (default on startup): left stick = translate, right stick X = rotate.
- **Arm mode**: left stick = shoulder pan/lift, right stick = elbow/wrist tilt.

**A button** — toggle between Base mode and Arm mode.

### Motion mapping — Base mode
| Control | Motion |
|---|---|
| Left stick, push up/down | Drive forward / backward (`x.vel`) |
| Left stick, push left/right | Strafe left / right (`y.vel`) |
| Right stick, push left/right | Rotate in place (`theta.vel`) |

Directions follow directly from the code (`x = -stick_y`, `y = stick_x`, `theta = -right_stick_x`)
but haven't been physically re-verified on hardware since the gamepad code changed — if
strafe or rotation comes out backwards from what feels intuitive, swap the sign in
`gamepad_input.py`'s `_dispatch()` (the `_dispatch_base(x=..., y=..., theta=..., ...)` call).

### Motion mapping — Arm mode
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

### Always-active controls (either mode)
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
| D-pad | `ABS_HAT0X` / `ABS_HAT0Y` (not currently used) |
| Y | `BTN_C` |
| X | `BTN_NORTH` |
| A | `BTN_B` |
| B | `BTN_A` |
| LB | `BTN_WEST` |
| RB | `BTN_Z` |
| LT | `ABS_Z` (proportional) |
| RT | `ABS_RZ` (proportional) |
| "-" / "+" (separate small buttons, unused) | `BTN_TL` / `BTN_TR` |
| Home | `KEY_MENU` (not currently used) |
| Select / Start | dead buttons on this unit — no event at all |

### Safety notes
- If the gamepad disconnects (Bluetooth drop, battery, etc.) mid-use, the server detects
  the silence and halts all motion automatically rather than continuing on the last
  received stick position.
- The web app and gamepad can be used interchangeably at any time — both drive the same
  robot through the same watchdog-protected control loop; whichever one is actively being
  used takes effect, and going idle on one doesn't block the other.
