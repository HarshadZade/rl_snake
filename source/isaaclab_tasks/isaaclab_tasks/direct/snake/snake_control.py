#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple Snake Robot Control for Lateral Undulation Motion in Isaac Sim

This script provides a standalone control system for testing basic lateral undulation
(planar wave) motion of a snake robot in Isaac Sim. It does not require the full
reinforcement learning environment setup.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/snake/snake_control.py [--fixed-base] [--headless]

Features:
    - Simple lateral undulation pattern implementation
    - Configurable wave parameters (amplitude, frequency, wave speed)
    - Support for both fixed and floating base robots
    - Real-time visualization in Isaac Sim
    - Easy parameter tuning for different motion patterns

Lateral Undulation Mathematics:
    The lateral undulation pattern is implemented as a traveling wave:
    θ(i,t) = A * sin(ωt - k*i*spacing)
    θ̇(i,t) = A * ω * cos(ωt - k*i*spacing)

    Where:
    - θ(i,t): Joint angle/velocity at joint i and time t
    - A: Wave amplitude
    - ω: Temporal frequency (rad/s)
    - k: Spatial wave number (rad/m)
    - i: Joint index (0 to num_joints-1)
    - spacing: Distance between joints
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Snake Robot Lateral Undulation Control")
parser.add_argument("--fixed-base", action="store_true", help="Use fixed base robot (default: floating base)")
parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds (default: 30)")
parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier (default: 1.0)")
parser.add_argument(
    "--control-mode", choices=["velocity", "position"], default="velocity", help="Control mode (default: velocity)"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""
import math
import time
import torch
from typing import Literal

# IsaacLab imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg


class LateralUndulationController:
    """
    Controller for generating lateral undulation motion patterns.

    Implements a traveling wave motion where each joint follows a sinusoidal
    pattern with a phase shift that creates a wave traveling along the snake body.
    """

    def __init__(
        self,
        amplitude_deg: float = 20.0,
        frequency_hz: float = 1.0,
        wave_speed_ms: float = 0.5,
        joint_spacing_m: float = 0.1,
        num_joints: int = 9,
    ):
        """
        Initialize the lateral undulation controller.

        Args:
            amplitude_deg: Maximum joint angle amplitude in degrees
            frequency_hz: Temporal frequency of the wave in Hz
            wave_speed_ms: Speed of wave propagation along the body in m/s
            joint_spacing_m: Distance between adjacent joints in meters
            num_joints: Number of joints in the snake robot
        """
        self.amplitude_rad = math.radians(amplitude_deg)
        self.omega = 2 * math.pi * frequency_hz  # Angular frequency (rad/s)
        self.wave_speed = wave_speed_ms
        self.joint_spacing = joint_spacing_m
        self.num_joints = num_joints

        # Calculate spatial wave number: k = ω/c where c is wave speed
        self.wave_number = self.omega / self.wave_speed if self.wave_speed > 0 else 0

        print("Lateral Undulation Controller initialized:")
        print(f"  Amplitude: {amplitude_deg:.1f}°")
        print(f"  Frequency: {frequency_hz:.2f} Hz")
        print(f"  Wave speed: {wave_speed_ms:.2f} m/s")
        print(f"  Wave number: {self.wave_number:.2f} rad/m")
        print(f"  Joint spacing: {joint_spacing_m:.3f} m")

    def generate_joint_velocities(self, current_time: float) -> torch.Tensor:
        """
        Generate joint velocities for lateral undulation at the current time.

        Args:
            current_time: Current simulation time in seconds

        Returns:
            torch.Tensor: Joint velocity targets [rad/s] for each joint
        """
        velocities = torch.zeros(self.num_joints)

        for i in range(self.num_joints):
            # Phase for this joint: combines temporal and spatial components
            phase = self.omega * current_time - self.wave_number * i * self.joint_spacing

            # Velocity is the derivative of the position: θ̇ = A*ω*cos(phase)
            velocity = self.amplitude_rad * self.omega * math.cos(phase)
            velocities[i] = velocity

        return velocities

    def generate_joint_positions(self, current_time: float) -> torch.Tensor:
        """
        Generate joint positions for lateral undulation at the current time.

        Args:
            current_time: Current simulation time in seconds

        Returns:
            torch.Tensor: Joint position targets [rad] for each joint
        """
        positions = torch.zeros(self.num_joints)

        for i in range(self.num_joints):
            # Phase for this joint
            phase = self.omega * current_time - self.wave_number * i * self.joint_spacing

            # Position: θ = A*sin(phase)
            position = self.amplitude_rad * math.sin(phase)
            positions[i] = position

        return positions

    def update_parameters(self, **kwargs):
        """Update controller parameters dynamically."""
        if "amplitude_deg" in kwargs:
            self.amplitude_rad = math.radians(kwargs["amplitude_deg"])
        if "frequency_hz" in kwargs:
            self.omega = 2 * math.pi * kwargs["frequency_hz"]
        if "wave_speed_ms" in kwargs:
            self.wave_speed = kwargs["wave_speed_ms"]
            self.wave_number = self.omega / self.wave_speed if self.wave_speed > 0 else 0
        if "joint_spacing_m" in kwargs:
            self.joint_spacing = kwargs["joint_spacing_m"]


class SnakeRobotController:
    """
    Main controller class for the snake robot in Isaac Sim.

    Handles robot setup, simulation management, and motion control.
    """

    def __init__(self, use_fixed_base: bool = False, control_mode: Literal["velocity", "position"] = "velocity"):
        """
        Initialize the snake robot controller.

        Args:
            use_fixed_base: Whether to use fixed base or floating base robot
            control_mode: Control mode - "velocity" or "position"
        """
        self.use_fixed_base = use_fixed_base
        self.control_mode = control_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the undulation controller
        self.undulation_controller = LateralUndulationController(
            amplitude_deg=25.0,  # Moderate amplitude for smooth motion
            frequency_hz=0.8,  # Slow enough to observe the wave
            wave_speed_ms=0.3,  # Reasonable wave propagation speed
            joint_spacing_m=0.1,  # Approximate spacing between joints
            num_joints=9,  # Based on the robot configuration
        )

        # Robot will be initialized in setup_robot()
        self.robot = None
        self.joint_vel_targets = None  # Will be initialized after robot setup
        self.sim_dt = 1.0 / 120.0  # 120 Hz simulation

    def setup_robot(self) -> None:
        """Setup the snake robot articulation."""
        # Choose USD file based on base type
        if self.use_fixed_base:
            usd_path = "./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_realistic_fixed_dim_v0.usda"
            print("Using fixed base snake robot")
        else:
            usd_path = (
                "./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_realistic_floating_dim_v0.usda"
            )
            print("Using floating base snake robot")

        # Configure the robot
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.1),  # Start slightly above ground
                rot=(0.0, 0.0, 0.0, 1.0),  # Identity quaternion
                joint_pos={"joint_.*": 0.0},  # All joints start at zero
            ),
            actuators={
                "snake_joints": ImplicitActuatorCfg(
                    joint_names_expr=["joint_[1-9]"],  # Joints 1 through 9
                    effort_limit_sim=100.0,  # Increased torque capability
                    velocity_limit_sim=5.0,  # Increased velocity limit
                    stiffness=0.0,  # Zero stiffness for pure velocity control
                    damping=20.0,  # Moderate damping for stability
                ),
            },
        )

        # Create the robot
        self.robot = Articulation(robot_cfg)
        print("Snake robot configuration created")

    def setup_scene(self) -> None:
        """Setup the simulation scene with ground plane and lighting."""
        # Add ground plane
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            )
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print("Scene setup complete")

    def apply_control(self, current_time: float) -> None:
        """
        Apply lateral undulation control to the robot.

        Args:
            current_time: Current simulation time in seconds
        """
        if self.robot is None or self.joint_vel_targets is None:
            return

        if self.control_mode == "velocity":
            # Generate velocity targets
            joint_velocities = self.undulation_controller.generate_joint_velocities(current_time)

            # Debug: Print velocity targets every few seconds
            if int(current_time * 10) % 50 == 0:  # Every 5 seconds
                print(f"  Generated velocities: {[f'{v:.3f}' for v in joint_velocities[:3]]}... (first 3)")

            # Update persistent velocity targets tensor (like in the environment)
            self.joint_vel_targets[0, :] = joint_velocities.to(self.device)

            # Apply to robot using the persistent tensor
            self.robot.set_joint_velocity_target(self.joint_vel_targets)

            # CRITICAL: Write the velocity targets to simulation
            self.robot.write_data_to_sim()

        elif self.control_mode == "position":
            # Generate position targets
            joint_positions = self.undulation_controller.generate_joint_positions(current_time)

            # Apply to robot
            position_targets = joint_positions.to(self.device).unsqueeze(0)  # Add batch dimension
            self.robot.set_joint_position_target(position_targets)

    def get_robot_state(self) -> dict:
        """Get current robot state for monitoring."""
        if self.robot is None:
            return {}

        return {
            "joint_positions": self.robot.data.joint_pos[0].cpu().numpy(),
            "joint_velocities": self.robot.data.joint_vel[0].cpu().numpy(),
            "root_position": self.robot.data.root_pos_w[0].cpu().numpy(),
            "root_orientation": self.robot.data.root_quat_w[0].cpu().numpy(),
        }

    def run_simulation(self, duration_seconds: float = 30.0, real_time_factor: float = 1.0) -> None:
        """
        Run the simulation with lateral undulation control.

        Args:
            duration_seconds: How long to run the simulation
            real_time_factor: Speed multiplier (1.0 = real-time, 0.5 = half speed, etc.)
        """
        print(f"\nStarting simulation for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early")

        start_time = time.time()
        sim_time = 0.0
        step_count = 0

        # Target step time for real-time control
        target_step_time = self.sim_dt / real_time_factor

        try:
            while sim_time < duration_seconds:
                step_start_time = time.time()

                # Apply lateral undulation control
                self.apply_control(sim_time)

                # Step the simulation
                sim_utils.SimulationContext.instance().step()

                # Update timing
                sim_time += self.sim_dt
                step_count += 1

                # Print status every 5 seconds
                if step_count % (5.0 / self.sim_dt) == 0:
                    state = self.get_robot_state()
                    if state:
                        joint_pos = state["joint_positions"]
                        root_pos = state["root_position"]
                        print(
                            f"Time: {sim_time:.1f}s | Root pos: [{root_pos[0]:.2f}, {root_pos[1]:.2f},"
                            f" {root_pos[2]:.2f}]"
                        )
                        print(f"  Joint angles: {[f'{p:.2f}' for p in joint_pos[:3]]}... (first 3 joints)")

                # Real-time control
                elapsed = time.time() - step_start_time
                if elapsed < target_step_time:
                    time.sleep(target_step_time - elapsed)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        actual_duration = time.time() - start_time
        print("\nSimulation complete!")
        print(f"Simulated time: {sim_time:.1f}s")
        print(f"Actual time: {actual_duration:.1f}s")
        print(f"Average FPS: {step_count / actual_duration:.1f}")


def main():
    """Main function to run the snake control demonstration."""

    print("=== Snake Robot Lateral Undulation Control ===")
    print(f"Mode: {'Fixed base' if args_cli.fixed_base else 'Floating base'}")
    print(f"Control: {args_cli.control_mode}")
    print(f"Duration: {args_cli.duration}s")
    print(f"Speed: {args_cli.speed}x")

    try:
        # Initialize simulation
        sim_cfg = SimulationCfg(dt=1 / 120, render_interval=1)
        sim = sim_utils.SimulationContext(sim_cfg)

        # Create and setup the controller
        controller = SnakeRobotController(use_fixed_base=args_cli.fixed_base, control_mode=args_cli.control_mode)

        # Setup scene and robot
        controller.setup_scene()
        controller.setup_robot()

        # Reset and play simulation to initialize everything
        sim.reset()
        sim.play()

        print("\nWaiting for simulation to settle...")
        for _ in range(10):  # Let physics settle
            sim.step()

        # Now we can safely access robot properties
        if controller.robot is not None:
            print(f"Snake robot initialized with {controller.robot.num_joints} joints")
            # Debug: Print joint names
            joint_names = controller.robot.joint_names
            print(f"Joint names: {joint_names}")

            # Initialize persistent velocity targets tensor (must be after robot is fully initialized)
            controller.joint_vel_targets = torch.zeros((1, controller.robot.num_joints), device=controller.device)

        # Run the main simulation
        controller.run_simulation(duration_seconds=args_cli.duration, real_time_factor=args_cli.speed)

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean shutdown
        simulation_app.close()


if __name__ == "__main__":
    main()
