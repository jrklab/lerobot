# Note for Human NOT Agent

## How to run uv
```bash
uv sync --locked --extra feetech --extra lekiwi --extra viz --extra pynput-dep
```

add needed dependency with `--extra`

## Run teleoperation with SO101 arm
```bash
uv run python examples/lekiwi/teleoperate.py
```

## Run teleoperation with gamepad
```bash
uv run python examples/lekiwi/gamepad_teleoperate.py
```

## Run calibration on Leader arm
```bash
uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyUSB0 --teleop.id=leader_arm_1
```
## Run calibration on Lekiwi
```bash
uv run lerobot-calibrate --robot.type=lekiwi --robot.port=/dev/ttyUSB0 --robot.id=kiwi_sn_0
```

## How to connect Gamepad to Laptop via Bluetooth
1. Make sure the controller is turned off.
2. Press and hold X + Home simultaneously for 2–3 seconds.
3. The LED indicator will start flashing rapidly (green in my case).
4. Open your laptop's Bluetooth settings and search for devices.
5. Bluetooth Name: Look for "Gamepad"
6. Click to pair, and the LED will remain solid once connected. or when you SSH to the laptop, use "bluetoothctl" in the terminal:
    -Type "scan on" to find the MAC address. ("A0:5A:5D:AC:B9:5F")
    -Type pair [MAC Address]
    -Type trust [MAC Address] to ensure it reconnects automatically.
    -Type connect [MAC Address] to connect