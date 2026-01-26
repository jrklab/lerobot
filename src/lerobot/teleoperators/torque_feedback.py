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
Torque feedback system for teleoperation.

This module provides utilities to map follower motor load/torque feedback
to leader motor resistance control, enabling haptic feedback for the operator.

The mapping process involves three layers:
1. Threshold detection: Ignore load below threshold to avoid noise
2. Scaled normalization: Scale load (0-1000) to torque limit (0-1000) range with sensitivity control
3. Per-motor adjustment: Apply motor-specific calibration factors
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TorqueFeedbackConfig:
    """Configuration for torque feedback system.

    Attributes:
        enabled: Whether to enable torque feedback.
        global_scale_factor: Global scaling factor (0.0-1.0) for feedback magnitude.
            0.5 means feedback will be 50% of follower load. Default: 1.0 (100%).
        per_motor_scales: Dict mapping motor names to individual scale factors.
            Default: 0.0 (no feedback unless explicitly specified).
        per_motor_thresholds: Dict mapping motor names to individual threshold ratios (0.0-1.0).
            Specifies minimum load ratio to activate feedback per motor. Default: 1.0 (100%, feedback disabled).
    """

    enabled: bool = True
    global_scale_factor: float = 1.0
    per_motor_scales: dict[str, float] = field(default_factory=dict)
    per_motor_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.global_scale_factor <= 1.0:
            raise ValueError(f"global_scale_factor must be in [0, 1], got {self.global_scale_factor}")
        for motor, scale in self.per_motor_scales.items():
            if not 0.0 <= scale <= 1.0:
                raise ValueError(f"per_motor_scales[{motor}] must be in [0, 1], got {scale}")
        for motor, threshold in self.per_motor_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"per_motor_thresholds[{motor}] must be in [0, 1], got {threshold}")




def map_load_to_torque_limit(
    load_dict: dict[str, float],
    config: TorqueFeedbackConfig,
    motor_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Map follower motor loads to leader torque limit commands.

    Implements three-layer mapping:
    1. Threshold: Load values below motor-specific threshold are zeroed
    2. Scale: Remaining load is scaled from [0, 1000] to [0, 1000] with global_scale_factor
    3. Per-motor: Apply individual motor scale factors

    Args:
        load_dict: Dict mapping motor names to Present_Load values (0-1000 range,
            where 1000 = 100% load).
        config: TorqueFeedbackConfig instance.
        motor_names: Optional list of expected motor names for validation.
            If provided, missing motors will be logged as warnings.

    Returns:
        Dict mapping motor names to torque limit values (0-1000 range) ready for
        writing to leader motors' Torque_Limit registers.

    Example:
        >>> config = TorqueFeedbackConfig(
        ...     global_scale_factor=1.0,
        ...     per_motor_scales={"shoulder_pan": 0.8, "gripper": 0.5},
        ...     per_motor_thresholds={"gripper": 0.2}  # Gripper threshold 20%, others 100% (disabled)
        ... )
        >>> loads = {"shoulder_pan": 150, "shoulder_lift": 50, "gripper": 250}
        >>> torque_limits = map_load_to_torque_limit(loads, config)
        >>> # shoulder_pan: threshold=1000, 150 < 1000, feedback = 0 (threshold too high)
        >>> # shoulder_lift: threshold=1000, 50 < 1000, feedback = 0 (no scale, threshold disabled)
        >>> # gripper: threshold=200, 250 > 200, applies scale 0.5
    """
    if not config.enabled:
        return {motor: 0 for motor in load_dict.keys()}

    torque_limits = {}

    for motor, load in load_dict.items():
        # Get motor-specific threshold, default to 1.0 (100%, effectively disabled)
        threshold_ratio = config.per_motor_thresholds.get(motor, 1.0)
        threshold = threshold_ratio * 1000.0

        # Layer 1: Threshold
        if load <= threshold:
            torque_limits[motor] = 0.0
            continue

        # Layer 2: Scale from [threshold, 1000] to [0, 1000]
        # Normalize to [0, 1] range, scale, then map to [0, 1000]
        normalized_load = (load - threshold) / (1000.0 - threshold)
        # normalized_load = load/1000.0 # Changed to use raw load for direct mapping
        scaled = normalized_load * config.global_scale_factor

        # Layer 3: Per-motor adjustment (default 0.0 = no feedback if not specified)
        motor_scale = config.per_motor_scales.get(motor, 0.0)
        torque_limit = scaled * motor_scale * 1000.0

        # Clamp to valid range [0, 1000]
        torque_limits[motor] = min(1000.0, max(0.0, torque_limit))

    # Log warnings for missing motors
    if motor_names:
        missing = set(motor_names) - set(load_dict.keys())
        if missing:
            logger.warning(f"Missing load values for motors: {missing}")

    return torque_limits
