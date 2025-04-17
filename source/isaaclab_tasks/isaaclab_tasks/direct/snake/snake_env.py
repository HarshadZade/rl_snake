# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains import TerrainImporter

@configclass
class SnakeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 100.0
    action_scale = 0.01  # rad #TODO: tune this
    action_space = 9    # 9 joints
    observation_space = 28 #TODO: fix this
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=40.0, replicate_physics=True)

    # -- Robot Configuration (Loading from USD)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot", # Standard prim path pattern
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/hzade/temp/snake_1.usd",
            activate_contact_sensors=False, # Set to True if you need contact sensors #TODO: check this
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0, # Tune if needed
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Define initial joint positions
            joint_pos={
                 "joint_1": 0.0,
                 "joint_2": 0.0,
                 "joint_3": 0.0,
                 "joint_4": 0.0,
                 "joint_5": 0.0,
                 "joint_6": 0.0,
                 "joint_7": 0.0,
                 "joint_8": 0.0,
                 "joint_9": 0.0,
            },
            pos=(0.0, 0.0, 1.0),  # Initial base position (adjust height based on robot)
            rot=(0.0, 0.0, 0.0, 1.0), # Initial base orientation
        ),
        actuators={
            # Define actuators for your joints #TODO: tune all these parameters
            "snake_joints": ImplicitActuatorCfg(
                # Use regex matching your joint names, or list them
                joint_names_expr=["joint_[1-9]"], # Example regex
                effort_limit=50000.0,   # <<< Tune based on your robot's specs
                velocity_limit=10.0,  # <<< Tune based on your robot's specs
                stiffness=100000.0,       # <<< Tune: Use >0 for position/velocity control
                damping=500.0,        # <<< Tune: Use >0 for position/velocity control (helps stability)
            ),
            # Add more actuator groups if joints have different properties
        },
    )

    # ground = GroundPlaneCfg(prim_path="/World/ground")
    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.9,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # reset
    joint_angle_range = [-1.57, 1.57] # rad

    # -- Task-Specific Parameters
    # Action scale now determines how much the target position changes per RL step
    action_scale = 0.01 #TODO: Tune this: Smaller means finer control over target change
    
    # reward scales #TODO: check if this makes sense, get the correct ones.
    rew_scale_forward_velocity = 100.0
    rew_scale_action_penalty = -0.005 #-0.005
    rew_scale_joint_vel_penalty = -0.001 #-0.001
    rew_scale_termination = -0.05 #-2.0
    rew_scale_alive = 0.1
    rew_scale_action_smoothness_penalty = -0.05 #-0.05
    rew_scale_lateral_velocity_penalty = -0.005 #-0.05
    rew_scale_joint_limit_penalty = -0.1 #-0.1

     # --- ADD TESTING CONFIGURATION ---
    @configclass
    class TestingCfg:
        """Configuration for testing modes."""
        # Set to True to override RL actions with manual oscillation
        enable_manual_oscillation: bool = False
        # Oscillation amplitude in degrees (will be converted to radians)
        oscillation_amplitude_deg: float = 30.0
        # Oscillation frequency in Hertz
        oscillation_frequency_hz: float = 1.0 # How many full cycles per second

    testing: TestingCfg = TestingCfg()
    # --- END TESTING CONFIGURATION ---

    @configclass
    class PositionTrackingCfg:
        """Configuration for position tracking analysis."""
        enable: bool = True
        env_id: int = 0     # Which environment to track
        joint_id: int = 0   # Which joint to track
        max_points: int = 1000  # Maximum number of data points to collect
        save_interval_s: float = 10.0  # How often to save plots (seconds)
    
    position_tracking: PositionTrackingCfg = PositionTrackingCfg()
    
    # --- ADD OBSERVATION HISTORY CONFIGURATION ---
    @configclass
    class ObservationHistoryCfg:
        """Configuration for observation history."""
        enable: bool = True
        history_length: int = 3  # How many past observations to include (including current)
    
    observation_history: ObservationHistoryCfg = ObservationHistoryCfg()
    # --- END OBSERVATION HISTORY CONFIGURATION ---

class SnakeEnv(DirectRLEnv):
    cfg: SnakeEnvCfg

    def __init__(self, cfg: SnakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.action_scale = self.cfg.action_scale
        self.env_step_counter = 0
        #TODO: Need to make sure all the required information is used and set here!!
        # Currently its just some random stuff!
        
        self.track_positions = self.cfg.position_tracking.enable # Use the flag from config
        if self.track_positions:
            self.tracking_env_id = self.cfg.position_tracking.env_id
            self.tracking_joint_id = self.cfg.position_tracking.joint_id
            self.max_tracking_points = self.cfg.position_tracking.max_points
            
            # Initialize position tracking data structure consistently
            self.position_tracking_data = {
                "timesteps": [],
                "commanded_positions": [],
                "actual_positions": [],
            }
            print(f"[Info] Position tracking enabled for Env {self.tracking_env_id}, Joint {self.tracking_joint_id}.")
            print(f"       Plots will be saved when you terminate the simulation (Ctrl+C).")
            
            # Set up signal handler for SIGINT (Ctrl+C)
            import signal
            def signal_handler(sig, frame):
                print("\nCaught interrupt signal. Saving position tracking plot before exiting...")
                self.save_position_tracking_plot()
                print("Plot saved. Exiting...")
                import sys
                sys.exit(0)
            
            # Register the signal handler for SIGINT
            signal.signal(signal.SIGINT, signal_handler)
        
        # --- Initialize observation history ---
        self.use_history = self.cfg.observation_history.enable
        if self.use_history:
            self.history_length = self.cfg.observation_history.history_length
            print(f"[Info] Observation history enabled with {self.history_length} frames.")
            
            # Calculate the size of a single observation
            single_obs_size = self._get_single_observation_size()
            
            # Initialize the observation history buffer with zeros
            # Shape: [num_envs, history_length, single_obs_size]
            self.obs_history = torch.zeros(
                (self.num_envs, self.history_length, single_obs_size), 
                device=self.device
            )
        # --- End observation history initialization ---
 
        self.joint_pos_limits = self.snake_robot.data.soft_joint_pos_limits
        self.joint_pos_lower_limits = self.joint_pos_limits[..., 0].to(self.device) # Ellipsis (...) means all preceding dims
        self.joint_pos_upper_limits = self.joint_pos_limits[..., 1].to(self.device)
        
        self.joint_pos_ranges = self.joint_pos_upper_limits - self.joint_pos_lower_limits + 1e-6
        self.joint_pos_mid = (self.joint_pos_lower_limits + self.joint_pos_upper_limits) / 2

        # Initialize action history
        # Start with default positions, cloned to avoid modifying the default tensor
        self.dof_targets = self.snake_robot.data.default_joint_pos.clone().to(device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Cache common data tensors (optional)
        self.joint_pos = self.snake_robot.data.joint_pos
        self.joint_vel = self.snake_robot.data.joint_vel
        self.root_state = self.snake_robot.data.root_state_w
    
    def _setup_scene(self):
        # Create snake robot articulation
        self.snake_robot = Articulation(self.cfg.robot)
        
        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Spawn the ground plane using the function and config
        # spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground)
        self.cfg.terrain.num_envs = self.cfg.scene.num_envs
        self.cfg.terrain.env_spacing = self.cfg.scene.env_spacing
        self._terrain = TerrainImporter(self.cfg.terrain)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # Add robot to the scene's list of articulations
        self.scene.articulations["snake_robot"] = self.snake_robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.env_step_counter += 1
        # Store action for smoothness calculations in reward
        self.prev_actions = self.dof_targets.clone()
        # Check if manual oscillation test mode is enabled
        if self.cfg.testing.enable_manual_oscillation:
            # --- ALTERNATING SINE WAVE LOGIC ---
            current_time = self.sim.current_time

            # Parameters from config
            amplitude_rad = math.radians(self.cfg.testing.oscillation_amplitude_deg)
            omega = 2.0 * math.pi * self.cfg.testing.oscillation_frequency_hz

            # Create tensor of joint indices [0, 1, 2, ..., 8]
            joint_indices = torch.arange(self.cfg.action_space, device=self.device) # Shape (9,)

            # Calculate phase offset for each joint: 0, pi, 2*pi, 3*pi, ...
            # This makes adjacent joints 180 degrees out of phase
            phase_offsets = joint_indices * math.pi # Shape (9,)

            # Calculate target angle for each joint at this time step
            # target = Amp * sin(omega * t + phase_offset)
            # Broadcasting: scalar * scalar + (9,) -> (9,)
            target_angles_rad = amplitude_rad * torch.sin(omega * current_time + phase_offsets) # Shape (9,)

            # Expand targets to all environments
            # Shape: (1, 9) -> (num_envs, 9)
            manual_targets = target_angles_rad.unsqueeze(0).expand(self.num_envs, -1)

            # Clamp the manually set targets to the joint limits (shape num_envs, 9)
            clamped_manual_targets = torch.clamp(
                manual_targets,
                self.joint_pos_lower_limits, # Shape (num_envs, 9)
                self.joint_pos_upper_limits  # Shape (num_envs, 9)
            )

            # Set the DOF targets directly
            self.dof_targets[:] = clamped_manual_targets

            # --- Optional: Print targets for debugging ---
            if self.env_step_counter % 20 == 0: # Print less often
                # Print targets for the first environment
                print(f"Step: {self.env_step_counter}, Time: {current_time:.4f}, "
                        f"Targets (rad): {(self.dof_targets[0].cpu().numpy().round(4))*180/math.pi}")
            # --- End Print ---

            # Set self.actions for potential use in reward calculations (optional)
            self.actions = torch.zeros_like(actions)
            # --- END ALTERNATING SINE WAVE LOGIC ---

        else:

            # Process actions from the policy for POSITION CONTROL
            self.actions = actions.clone().clamp_(-1.0, 1.0)

            # Calculate the desired change in target position
            # Scale action by action_scale and time step
            # The time step scaling makes the effect somewhat independent of simulation frequency
            delta_targets = self.action_scale * self.actions * self.cfg.sim.dt * self.cfg.decimation

            # Get current joint positions
            current_joint_pos = self.snake_robot.data.joint_pos

            # Calculate new targets by adding the delta to current positions
            new_targets = current_joint_pos + delta_targets
            
            self.dof_targets[:] = torch.clamp(new_targets, self.joint_pos_lower_limits, self.joint_pos_upper_limits)

        # NOTE: Removed position tracking data collection from here - now done in step() method
    
    def _apply_action(self) -> None:
        self.snake_robot.set_joint_position_target(self.dof_targets)

    def _get_single_observation_size(self):
        """Calculate the size of a single observation (without history)."""
        # Get joint positions and velocities
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        
        # Calculate a single observation
        single_obs = torch.cat(
            (
                # Normalized joint positions (shape: num_envs x 9)
                torch.zeros_like(joint_pos),
                # Scaled joint velocities (shape: num_envs x 9)
                torch.zeros_like(joint_vel),
                # Root position (shape: num_envs x 3)
                torch.zeros((self.num_envs, 3), device=self.device),
                # Root orientation (shape: num_envs x 4)
                torch.zeros((self.num_envs, 4), device=self.device),
                # Root linear velocity (shape: num_envs x 3)
                torch.zeros((self.num_envs, 3), device=self.device),
            ),
            dim=-1,
        )
        
        # Return the size of the last dimension (observation features)
        return single_obs.shape[-1]

    def _get_observations(self) -> dict:
        # Get joint positions and velocities
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel

        # # Calculate joint positions normalized to [-1, 1]
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits) / self.joint_pos_ranges - 1.0
        # Get root state information
        root_pos = self.snake_robot.data.root_pos_w
        root_quat = self.snake_robot.data.root_quat_w
        root_lin_vel = self.snake_robot.data.root_lin_vel_w
        
        # Combine observations (current frame only)
        current_obs = torch.cat(
            (
                joint_pos_normalized,   # Normalized joint positions
                joint_vel * 0.1,        # Scaled joint velocities
                root_pos,               # Root position
                root_quat,              # Root orientation
                root_lin_vel,           # Root linear velocity
            ),
            dim=-1,
        )
        
        if self.use_history:
            # Shift the history buffer (discard oldest, make room for newest)
            self.obs_history = self.obs_history.roll(-1, dims=1)
            
            # Insert the current observation as the newest entry
            self.obs_history[:, -1, :] = current_obs
            
            # Flatten the history for the policy
            # Shape goes from [num_envs, history_length, single_obs_size] 
            # to [num_envs, history_length * single_obs_size]
            policy_obs = self.obs_history.reshape(self.num_envs, -1)
            
            # Optional: add observation normalization if needed
            # policy_obs = torch.clamp(policy_obs, -10.0, 10.0)
            
            observations = {"policy": policy_obs}
            print("Shape of observations using history:", observations["policy"].shape)
            exit(0)
        else:
            # Just use the current observation if history is disabled
            observations = {"policy": current_obs}
            print("Shape of observations:", observations["policy"].shape)
            exit(0)
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Calculate forward velocity (x-direction for snake locomotion)
        root_vel = self.snake_robot.data.root_lin_vel_w
        forward_vel = root_vel[:, 0]  # X-component of velocity
        
        # Forward reward - primary objective is to move forward
        forward_reward = self.cfg.rew_scale_forward_velocity * forward_vel
        
        # Sideways movement penalty - discourage excessive lateral motion
        lateral_vel_penalty = self.cfg.rew_scale_lateral_velocity_penalty * torch.abs(root_vel[:, 1])
        
        # Action smoothness - penalize jerky changes in position targets
        action_diff = self.dof_targets - self.prev_actions
        action_smoothness_penalty = self.cfg.rew_scale_action_smoothness_penalty * torch.sum(action_diff**2, dim=-1)
        
        # Energy consumption - penalize high joint velocities
        joint_vel = self.snake_robot.data.joint_vel
        energy_penalty = self.cfg.rew_scale_joint_vel_penalty * torch.sum(joint_vel**2, dim=-1)
        
        # Joint limit penalty - discourage operating at the limits
        normalized_joint_pos = (self.snake_robot.data.joint_pos - self.joint_pos_mid) / (self.joint_pos_ranges / 2)
        joint_limit_penalty = self.cfg.rew_scale_joint_limit_penalty * torch.sum(
            torch.maximum(torch.abs(normalized_joint_pos) - 0.8, torch.zeros_like(normalized_joint_pos))**2, 
            dim=-1
        )
        
        # Alive bonus - small constant reward for not terminating
        alive_bonus = self.cfg.rew_scale_alive
        
        # Calculate total reward
        total_reward = (
            forward_reward + 
            lateral_vel_penalty + 
            action_smoothness_penalty + 
            energy_penalty + 
            joint_limit_penalty + 
            alive_bonus
        )
        
        # For debugging, store reward components
        self.extras["log"] = {
            "forward_reward": forward_reward.mean().item(),
            "lateral_vel_penalty": lateral_vel_penalty.mean().item(),
            "action_smoothness_penalty": action_smoothness_penalty.mean().item(),
            "energy_penalty": energy_penalty.mean().item(),
            "joint_limit_penalty": joint_limit_penalty.mean().item(),
            "total_reward": total_reward.mean().item(),
        }
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time-based termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # # Terminate based on robot state
        # root_pos = self.snake_robot.data.root_pos_w
        # root_quat = self.snake_robot.data.root_quat_w
        
        # # Calculate up vector (z-axis) in world coordinates
        # forward = torch.zeros((self.num_envs, 3), device=self.device)
        # forward[:, 0] = 1.0  # Forward is along x-axis
        
        # # 1. Terminate if robot is flipped (head too low)
        # head_too_low = root_pos[:, 2] < 0.05
        
        # # 2. Terminate if robot has flipped over significantly
        # up = torch.zeros((self.num_envs, 3), device=self.device)
        # up[:, 2] = 1.0  # Up is along z-axis
        # up_world = quat_rotate(root_quat, up)
        # too_tilted = up_world[:, 2] < 0.0  # z component negative means flipped
        
        # terminated = head_too_low | too_tilted
        self.joint_pos = self.snake_robot.data.joint_pos
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_lower_limits, dim=1) | \
                        torch.any(self.joint_pos > self.joint_pos_upper_limits, dim=1)
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = []  # Use empty list instead of tensor to avoid type error
        super()._reset_idx(env_ids)
        
        # Reset joint positions to a neutral pose with small noise
        n_envs = len(env_ids) if env_ids is not None else self.num_envs
        
        # Start with default positions
        joint_pos = self.snake_robot.data.default_joint_pos[env_ids].clone()
        
        # Option: Add a sinusoidal pattern as starting position
        # This can help the robot start with a sensible snake-like posture
        if True:
            for i in range(self.cfg.action_space):
                # Phase offset to create a sinusoidal wave along the body
                # Each joint is 90 degrees out of phase with the previous one
                phase_offset = i * (math.pi / 2)
                # Amplitude of the wave
                amplitude = 0.2
                # Apply sinusoidal pattern
                joint_pos[:, i] = amplitude * torch.sin(torch.tensor(phase_offset))
        
        # Add small random noise to initial positions
        joint_pos += torch.randn_like(joint_pos) * 0.05
        
        # Ensure joints are within limits
        joint_pos = torch.clamp(
            joint_pos,
            self.joint_pos_lower_limits[env_ids],
            self.joint_pos_upper_limits[env_ids]
        )
        
        # Zero velocities
        joint_vel = torch.zeros_like(joint_pos)
        
        # Reset root state
        default_root_state = self.snake_robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write states to simulation
        self.snake_robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.snake_robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.snake_robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset action buffer
        if env_ids is not None:
            self.dof_targets[env_ids] = joint_pos
            self.prev_actions[env_ids] = joint_pos
        else:
            self.dof_targets = joint_pos
            self.prev_actions = joint_pos
            
        # Reset observation history for the reset environments
        if self.use_history:
            # Get the current observation for these environments
            joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits[env_ids]) / self.joint_pos_ranges[env_ids] - 1.0
            root_pos = default_root_state[:, :3]
            root_quat = default_root_state[:, 3:7]
            root_lin_vel = torch.zeros_like(root_pos)  # Zero velocity on reset
            
            # Create the initial observation
            initial_obs = torch.cat(
                (
                    joint_pos_normalized,   # Normalized joint positions
                    joint_vel * 0.1,        # Scaled joint velocities (zeros)
                    root_pos,               # Root position
                    root_quat,              # Root orientation
                    root_lin_vel,           # Root linear velocity (zeros)
                ),
                dim=-1,
            )
            
            # Fill the entire history with the initial observation
            if len(env_ids) > 0:  # Only if there are environments to reset
                for t in range(self.history_length):
                    self.obs_history[env_ids, t, :] = initial_obs

    # --- Override the step method ---
    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Override the main environment step to collect plotting data."""

        # Before taking the step, record the current time and commanded position
        current_time = self.sim.current_time
        commanded_position = None
        
        if self.track_positions and len(self.position_tracking_data["timesteps"]) < self.max_tracking_points:
            commanded_position = self.dof_targets[self.tracking_env_id, self.tracking_joint_id].item()

        # --- Call the original DirectRLEnv step logic ---
        # This handles: _pre_physics_step, sim.step(), _get_observations, _get_rewards, _get_dones, _reset_idx etc.
        obs_dict, rew, terminated, truncated, extras = super().step(actions)
        # --- End original step logic ---

        # --- Collect BOTH commanded and actual position data AFTER physics step ---
        if self.track_positions and commanded_position is not None:
            # Get actual joint positions after the physics step
            actual_pos = self.snake_robot.data.joint_pos[self.tracking_env_id, self.tracking_joint_id].item()

            # Debug: print collection info occasionally
            if self.env_step_counter % 100 == 0:
                print(f"Collecting data point {len(self.position_tracking_data['timesteps'])} at time {self.sim.current_time:.2f}s")
                print(f"  Commanded: {commanded_position:.4f}, Actual: {actual_pos:.4f}")
            
            # Save both data points
            self.position_tracking_data["timesteps"].append(current_time)
            self.position_tracking_data["commanded_positions"].append(commanded_position)
            self.position_tracking_data["actual_positions"].append(actual_pos)

            # No longer triggering plot saving periodically - will save on Ctrl+C via signal handler

        return obs_dict, rew, terminated, truncated, extras
    # --- End override ---
    
    def save_position_tracking_plot(self):
        print("#############################################")
        print("Saving position tracking plot")
        print("#############################################")
        
        # Save the tracking data to a file
        if not self.position_tracking_data["timesteps"]:
            print("No position tracking data to plot!")
            return
            
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os
        
        # Debug info about data collection
        timesteps_count = len(self.position_tracking_data["timesteps"])
        actual_count = len(self.position_tracking_data["actual_positions"])
        cmd_count = len(self.position_tracking_data["commanded_positions"])
        
        print(f"Plotting data: timesteps={timesteps_count}, commanded={cmd_count}, actual={actual_count}")
        
        # Check if data lengths match
        if timesteps_count != cmd_count or timesteps_count != actual_count:
            print("WARNING: Data array lengths don't match!")
            print(f"  timesteps: {timesteps_count}, commanded: {cmd_count}, actual: {actual_count}")
            # Adjust the arrays to the same length if needed
            min_len = min(timesteps_count, cmd_count, actual_count)
            self.position_tracking_data["timesteps"] = self.position_tracking_data["timesteps"][:min_len]
            self.position_tracking_data["commanded_positions"] = self.position_tracking_data["commanded_positions"][:min_len]
            self.position_tracking_data["actual_positions"] = self.position_tracking_data["actual_positions"][:min_len]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        times = np.array(self.position_tracking_data["timesteps"])
        commanded = np.array(self.position_tracking_data["commanded_positions"])
        actual = np.array(self.position_tracking_data["actual_positions"])
        
        # Only plot up to the length of the shortest array
        min_len = min(len(times), len(commanded), len(actual))
        print(f"Plotting {min_len} data points")
        
        plt.plot(times[:min_len], commanded[:min_len], 'b-', label=f'Commanded Position (Joint {self.tracking_joint_id})')
        plt.plot(times[:min_len], actual[:min_len], 'r-', label=f'Actual Position (Joint {self.tracking_joint_id})')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (rad)')
        plt.title(f'Joint Position Tracking - Env {self.tracking_env_id}, Joint {self.tracking_joint_id}')
        plt.legend()
        plt.grid(True)
        
        # Calculate position tracking error metrics
        if min_len > 0:
            error = commanded[:min_len] - actual[:min_len]
            rmse = np.sqrt(np.mean(np.square(error)))
            max_error = np.max(np.abs(error))
            avg_error = np.mean(np.abs(error))
            
            plt.figtext(0.02, 0.02, 
                      f'RMSE: {rmse:.4f} rad\nMax Error: {max_error:.4f} rad\nAvg Error: {avg_error:.4f} rad',
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "joint_tracking_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f'joint_tracking_env{self.tracking_env_id}_joint{self.tracking_joint_id}_{timestamp}.png')
        plt.savefig(filename)
        plt.close()
        
        # Optional: Save the raw data as CSV
        data = np.column_stack((times[:min_len], commanded[:min_len], actual[:min_len]))
        csv_filename = os.path.join(output_dir, f'joint_tracking_data_env{self.tracking_env_id}_joint{self.tracking_joint_id}_{timestamp}.csv')
        np.savetxt(
            csv_filename,
            data,
            delimiter=',',
            header='time,commanded_position,actual_position'
        )
        
        print(f"Saved position tracking data and plot to {filename} at t={self.sim.current_time:.2f}s")
        return filename