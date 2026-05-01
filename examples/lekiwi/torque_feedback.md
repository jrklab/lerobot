# Torque Feedback for Teleoperation

Torque feedback lets the operator **feel** resistance on the leader arm that mirrors what the follower arm is experiencing. When the follower arm collides with an object or reaches its joint limits, the leader arm stiffens proportionally, giving the human operator a haptic cue.

---

## 1. Motor Registers

### Reading load from the follower arm

| Register | Address (STS/SMS) | Size | Direction | Description |
|---|---|---|---|---|
| `Present_Load` | `0x3C` (60) | 2 bytes | Read-only | Current motor load. Uses **sign-magnitude encoding**: bit 10 is the sign (direction), bits 0-9 are the magnitude (0–1000, where 1000 = 100% load). |

The raw value is decoded with sign-magnitude logic inside `FeetechMotorsBus`. The teleoperation code then takes `abs()` of the decoded value so the load magnitude is always non-negative (0–1000).

```python
# src/lerobot/robots/so_follower/so_follower.py
load_dict = self.bus.sync_read("Present_Load")
obs_dict.update({f"{motor}.load": abs(val) for motor, val in load_dict.items()})
```

### Writing torque limit to the leader arm

| Register | Address (STS/SMS) | Size | Direction | Description |
|---|---|---|---|---|
| `Torque_Enable` | `0x28` (40) | 1 byte | Read/Write | `1` = torque on (motor resists movement), `0` = torque off (motor is passive/compliant). |
| `Torque_Limit` | `0x30` (48) | 2 bytes | Read/Write | Maximum torque the motor will apply (0–1000, where 1000 = 100% of rated torque). Effective only when `Torque_Enable = 1`. |

Both registers live in **SRAM** (volatile; reset on power cycle).

```python
# src/lerobot/teleoperators/so_leader/so_leader.py  send_feedback()
self.bus.sync_write("Torque_Enable", torque_enable_dict)   # 1 if feedback > 0, else 0
self.bus.sync_write("Torque_Limit",  torque_limit_dict)    # only for motors with feedback > 0
```

Motors with zero computed feedback have their torque **disabled**, keeping the arm fully back-drivable and preventing stiff joints when the follower is unloaded.

---

## 2. Load → Torque Limit Mapping Formula

The mapping is implemented in `map_load_to_torque_limit()` ([src/lerobot/teleoperators/torque_feedback.py](../src/lerobot/teleoperators/torque_feedback.py)) and consists of three sequential layers.

### Layer 1 — Threshold gate

$$
\text{load\_effective} = \begin{cases} 0 & \text{if } L \le \theta_\text{motor} \cdot 1000 \\ L & \text{otherwise} \end{cases}
$$

Where $L \in [0, 1000]$ is the absolute `Present_Load` value and $\theta_\text{motor} \in [0, 1]$ is the per-motor threshold ratio. Values at or below the threshold are discarded; the motor remains compliant at those load levels.

### Layer 2 — Scaled normalisation

$$
L_\text{norm} = \frac{L - \theta_\text{motor} \cdot 1000}{1000 - \theta_\text{motor} \cdot 1000}
\quad \in [0, 1]
$$

$$
L_\text{scaled} = L_\text{norm} \cdot g
$$

Where $g \in [0, 1]$ is `global_scale_factor`. This maps the load range $[\theta \cdot 1000,\ 1000]$ linearly onto $[0,\ g]$.

### Layer 3 — Per-motor gain

$$
T_\text{limit} = \text{clamp}\!\left(L_\text{scaled} \cdot s_\text{motor} \cdot 1000,\ 0,\ 1000\right)
$$

Where $s_\text{motor} \in [0, 1]$ is `per_motor_scales[motor]`.

### Combined closed-form

$$
\boxed{
T_\text{limit} = \text{clamp}\!\left(
  \frac{L - \theta \cdot 1000}{1000(1 - \theta)} \cdot g \cdot s \cdot 1000,\quad 0,\quad 1000
\right)
\quad \text{if } L > \theta \cdot 1000, \text{ else } 0
}
$$

**Example** (gripper, $\theta=0.2$, $g=1.0$, $s=0.3$, $L=600$):

$$
T = \frac{600 - 200}{800} \times 1.0 \times 0.3 \times 1000 = 0.5 \times 300 = 150
$$

---

## 3. Files Changed and Change Details

### New file

| File | Description |
|---|---|
| [`src/lerobot/teleoperators/torque_feedback.py`](../src/lerobot/teleoperators/torque_feedback.py) | Core module. Defines `TorqueFeedbackConfig` dataclass and `map_load_to_torque_limit()` function. Contains all threshold/scaling logic and input validation. |

### Modified files

#### `src/lerobot/robots/so_follower/so_follower.py`

- **`get_observation()`** — added a second `sync_read("Present_Load")` call after reading position. Absolute-value decoded load values are merged into `obs_dict` under keys `{motor}.load`.
- **`observation_features`** property — extended to include `{motor}.load: float` entries via new `_motors_load_ft` property.

#### `src/lerobot/teleoperators/so_leader/so_leader.py`

- **`send_feedback()`** — new method. Strips `.torque` suffixes, builds a `Torque_Enable` dict (1 if value > 0, else 0), syncs it to the bus, then syncs non-zero values to `Torque_Limit`.
- **`feedback_features`** property — returns `{motor}.torque: float` for all motors.

#### `examples/lekiwi/teleoperate.py`

- Imports `TorqueFeedbackConfig` and `map_load_to_torque_limit`.
- Instantiates `torque_feedback_config` with per-motor scales and thresholds for all six SO101 joints.
- Adds keyboard toggle (`'b'` key, edge-detected) to enable/disable feedback at runtime.
- In the main loop: extracts `arm_*.load` keys from the observation, strips prefixes/suffixes to get bare motor names, calls `map_load_to_torque_limit()`, then calls `leader_arm.send_feedback()`.
- Sends an all-zero feedback dict when feedback is toggled off to ensure the leader arm returns to a compliant state immediately.

---

## 4. Hyperparameters to Tune

### `TorqueFeedbackConfig` fields

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `enabled` | `bool` | `True` | Master on/off switch. Set to `False` at startup and toggle with `'b'` in the example script. |
| `global_scale_factor` (`g`) | `float` [0, 1] | `1.0` | Global multiplier applied before per-motor scales. Reduce to soften feedback across **all** motors simultaneously. Good first knob for overall intensity. |
| `per_motor_scales` (`s`) | `dict[str, float]` each in [0, 1] | `0.0` (no feedback) | Individual gain per joint. Joints not listed default to **0** (no feedback). Tune to balance the felt resistance between large joints (shoulder) and delicate ones (wrist, gripper). |
| `per_motor_thresholds` (`θ`) | `dict[str, float]` each in [0, 1] | `1.0` (disabled) | Dead-zone threshold as a fraction of full load. Motors not listed default to `1.0`, meaning feedback is **off** unless explicitly configured. Higher threshold = less sensitivity, more noise rejection. |

### Recommended starting point (SO101, from `teleoperate.py`)

```python
TorqueFeedbackConfig(
    enabled=False,          # toggle on with 'b'
    global_scale_factor=1.0,
    per_motor_scales={
        "shoulder_pan":  0.5,
        "shoulder_lift": 0.5,
        "elbow_flex":    0.5,
        "wrist_flex":    0.5,
        "wrist_roll":    0.5,
        "gripper":       0.3,   # lower: gripper motor is weaker
    },
    per_motor_thresholds={
        "shoulder_pan":  0.3,   # 30% dead zone
        "shoulder_lift": 0.5,   # 50% dead zone (noisier joint)
        "elbow_flex":    0.5,
        "wrist_flex":    0.5,
        "wrist_roll":    0.3,
        "gripper":       0.2,   # 20% dead zone (sensitive)
    },
)
```

### Tuning guidelines

- **Feedback feels too strong / arm locks up**: decrease `global_scale_factor` or lower individual `per_motor_scales`.
- **Feedback feels too weak / unnoticeable**: increase `per_motor_scales` or lower `per_motor_thresholds` to activate at lower loads.
- **Arm buzzes or feels noisy at rest**: raise the threshold (`per_motor_thresholds`) for the offending joint so small static loads are filtered out.
- **Gripper stalls or burns out**: keep `per_motor_scales["gripper"]` ≤ 0.3 and confirm `Max_Torque_Limit` is set to 500 (50%) in `configure()`.
- **Latency feels high**: the feedback path runs synchronously in the main loop at `FPS = 30`. Reducing the number of sync writes (e.g. skipping motors with near-zero load) can help on slower USB connections.
