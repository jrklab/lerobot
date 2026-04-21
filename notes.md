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