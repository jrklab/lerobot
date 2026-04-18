#!/usr/bin/env python

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

"""
probe_gamepad.py — Helper script to discover the axis and button event codes
for your specific USB or Bluetooth gamepad.

Usage:
    python examples/lekiwi/probe_gamepad.py

Move every stick, press every button, and the script will print the raw
event code, symbolic name, and value. Use this output to verify (or customise)
the axis/button mappings in gamepad_teleoperate.py.

Press Ctrl+C to quit.
"""

import select

import evdev
from evdev import ecodes


def find_gamepads() -> list[evdev.InputDevice]:
    """Return all /dev/input/event* devices that look like gamepads."""
    found = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except Exception:
            continue
        caps = dev.capabilities()
        abs_codes = {c for c, _ in caps.get(ecodes.EV_ABS, [])}
        key_codes = {c for c in caps.get(ecodes.EV_KEY, [])}
        has_sticks  = ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes
        has_buttons = ecodes.BTN_SOUTH in key_codes or ecodes.BTN_GAMEPAD in key_codes
        if has_sticks and has_buttons:
            found.append(dev)
        else:
            dev.close()
    return found


def main():
    gamepads = find_gamepads()
    if not gamepads:
        print("No gamepad found!")
        print("  USB: make sure the controller is plugged in.")
        print("  BT : pair it first with bluetoothctl, then re-run.")
        return

    print(f"Found {len(gamepads)} gamepad(s):")
    for i, gp in enumerate(gamepads):
        print(f"  [{i}] {gp.name}  ({gp.path})")
    print("\nMove sticks, push triggers, press buttons … (Ctrl+C to quit)\n")
    print(f"{'type':<12} {'code_name':<24} {'raw_value'}")
    print("-" * 50)

    # Use select so we can read from all gamepads at once
    devices = {gp.fd: gp for gp in gamepads}

    while True:
        r, _, _ = select.select(devices, [], [])
        for fd in r:
            dev = devices[fd]
            try:
                for event in dev.read():
                    if event.type == ecodes.EV_SYN:
                        continue
                    if event.type == ecodes.EV_ABS:
                        names = ecodes.ABS.get(event.code, str(event.code))
                        name = names[0] if isinstance(names, list) else names
                        print(f"{'Absolute':<12} {name:<24} {event.value}")
                    elif event.type == ecodes.EV_KEY:
                        names = ecodes.BTN.get(event.code) or ecodes.KEY.get(event.code, str(event.code))
                        name = names[0] if isinstance(names, (list, tuple)) else names
                        print(f"{'Key':<12} {name:<24} {event.value}")
            except Exception as e:
                print(f"[read error] {e}")


if __name__ == "__main__":
    main()
