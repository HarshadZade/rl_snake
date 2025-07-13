# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oscillation controller for snake robot locomotion testing."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .snake_env_cfg import SnakeEnvCfg


class OscillationController:
    """Oscillation controller for testing snake robot locomotion patterns.

    This class implements various oscillation patterns for snake robots, including:
    - Sidewinding: Biological locomotion pattern with alternating joint phases
    - Constant velocity: Simple constant velocity pattern for all joints

    The controller generates joint velocity targets based on configured parameters
    and can be extended to support additional locomotion patterns.
    """

    def __init__(self, cfg: SnakeEnvCfg.TestingCfg, num_joints: int, device: torch.device):
        """Initialize the oscillation controller.

        Args:
            cfg: Testing configuration containing oscillation parameters
            num_joints: Number of joints in the snake robot
            device: PyTorch device for tensor operations
        """
        self.cfg = cfg
        self.num_joints = num_joints
        self.device = device

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the oscillation configuration parameters."""
        if self.cfg.oscillation_type not in ["sidewinding", "constant"]:
            raise ValueError(f"Unknown oscillation type: {self.cfg.oscillation_type}")

        if self.num_joints <= 0:
            raise ValueError(f"Number of joints must be positive, got: {self.num_joints}")

    def generate_velocity_targets(self, current_time: float) -> torch.Tensor:
        """Generate velocity targets based on the configured oscillation pattern.

        Args:
            current_time: Current simulation time in seconds

        Returns:
            torch.Tensor: Velocity targets for all joints. Shape: [num_joints]
        """
        if self.cfg.oscillation_type == "sidewinding":
            return self._generate_sidewinding_pattern(current_time)
        elif self.cfg.oscillation_type == "constant":
            return self._generate_constant_pattern()
        else:
            raise ValueError(f"Unknown oscillation type: {self.cfg.oscillation_type}")

    def _generate_sidewinding_pattern(self, current_time: float) -> torch.Tensor:
        """Generate sidewinding locomotion pattern for snake robot.

        Implements the biological sidewinding gait using sinusoidal joint velocities
        with phase offsets between adjacent joints and different patterns for even/odd joints.

        The sidewinding pattern creates a wave-like motion that allows the snake to move
        sideways, which is effective for locomotion on loose surfaces like sand.

        Mathematical model:
        - Even joints (i % 2 == 0): velocity(i,t) = Ax * ωx * cos(ωx*t + i*δx)
        - Odd joints (i % 2 == 1):  velocity(i,t) = Ay * ωy * cos(ωy*t + i*δy + φ)

        Args:
            current_time: Current simulation time in seconds

        Returns:
            torch.Tensor: Velocity targets for sidewinding motion. Shape: [num_joints]
        """
        velocity_targets = torch.zeros((self.num_joints,), device=self.device)

        # Extract sidewinding parameters from config
        amplitude_x_rad = math.radians(self.cfg.amplitude_x_deg)
        amplitude_y_rad = math.radians(self.cfg.amplitude_y_deg)
        omega_x = self.cfg.omega_x
        omega_y = self.cfg.omega_y
        delta_x = self.cfg.delta_x
        delta_y = self.cfg.delta_y
        phi = self.cfg.phi

        # Generate sidewinding velocities using alternating patterns
        for i in range(self.num_joints):
            if i % 2 == 0:  # Even joints - lateral undulation
                # velocity(i,t) = Ax * ωx * cos(ωx*t + i*δx)
                phase = omega_x * current_time + i * delta_x
                velocity_targets[i] = amplitude_x_rad * omega_x * torch.cos(torch.tensor(phase))
            else:  # Odd joints - vertical undulation with phase offset
                # velocity(i,t) = Ay * ωy * cos(ωy*t + i*δy + φ)
                phase = omega_y * current_time + i * delta_y + phi
                velocity_targets[i] = amplitude_y_rad * omega_y * torch.cos(torch.tensor(phase))

        return velocity_targets

    def _generate_constant_pattern(self) -> torch.Tensor:
        """Generate constant velocity pattern for all joints.

        This pattern sets all joints to the same constant velocity, which is useful
        for testing basic robot functionality and control systems.

        Returns:
            torch.Tensor: Constant velocity targets for all joints. Shape: [num_joints]
        """
        velocity_targets = torch.zeros((self.num_joints,), device=self.device)
        velocity_targets.fill_(self.cfg.constant_velocity)
        return velocity_targets

    def get_pattern_info(self) -> dict:
        """Get information about the current oscillation pattern.

        Returns:
            dict: Dictionary containing pattern information including type and parameters
        """
        info = {
            "oscillation_type": self.cfg.oscillation_type,
            "num_joints": self.num_joints,
        }

        if self.cfg.oscillation_type == "sidewinding":
            info.update({
                "amplitude_x_deg": self.cfg.amplitude_x_deg,
                "amplitude_y_deg": self.cfg.amplitude_y_deg,
                "omega_x": self.cfg.omega_x,
                "omega_y": self.cfg.omega_y,
                "delta_x": self.cfg.delta_x,
                "delta_y": self.cfg.delta_y,
                "phi": self.cfg.phi,
            })
        elif self.cfg.oscillation_type == "constant":
            info.update({
                "constant_velocity": self.cfg.constant_velocity,
            })

        return info

    def update_config(self, new_cfg: SnakeEnvCfg.TestingCfg) -> None:
        """Update the controller configuration.

        Args:
            new_cfg: New testing configuration
        """
        self.cfg = new_cfg
        self._validate_config()
