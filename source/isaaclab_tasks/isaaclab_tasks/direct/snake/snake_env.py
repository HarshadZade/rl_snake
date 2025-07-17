# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_matrix

from .oscillation_controller import OscillationController
from .snake_env_cfg import SnakeEnvCfg


class SnakeEnv(DirectRLEnv):
    cfg: SnakeEnvCfg

    def __init__(self, cfg: SnakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._render = render_mode is not None

        self.env_step_counter = 0

        # Define which link to track for target reaching
        # self.tracked_link_idx = torch.tensor([self.cfg.target_position.tracked_link_idx], device=self.device)

        # Store target position, used in:
        # 1. Constructing observation
        # 2. Reward function
        # 3. Reset (TODO: optionally, can vary)
        self.target_position = torch.tensor(self.cfg.target_position.target_pos, device=self.device)

        # Add a buffer to track whether each environment has reached the target, used in:
        # 1. Reward function
        self.target_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Track closest distance to target for each environment (initialize with large value)
        # Updated in:
        # 1. Reward Function
        # 2. Reset
        self.closest_distance = torch.ones(self.num_envs, device=self.device) * 100.0

        # Tracking initializations
        if self.cfg.position_tracking.enable:
            self.tracking_env_id = self.cfg.position_tracking.env_id
            self.tracking_joint_id = self.cfg.position_tracking.joint_id
            self.track_all_joints = self.cfg.position_tracking.track_all_joints
            self.max_tracking_points = self.cfg.position_tracking.max_points

            if self.track_all_joints:
                print(f"[Info] Velocity tracking enabled for all joints in Env {self.tracking_env_id}.")
            else:
                print(
                    f"[Info] Velocity tracking enabled for Env {self.tracking_env_id}, Joint {self.tracking_joint_id}."
                )

        # Initialize joint position limits
        joint_pos_limits = self.snake_robot.data.soft_joint_pos_limits
        self.joint_pos_lower_limits = joint_pos_limits[..., 0].to(
            self.device
        )  # Ellipsis (...) means all preceding dims
        self.joint_pos_upper_limits = joint_pos_limits[..., 1].to(self.device)

        # Get joint position range for normalization
        self.joint_pos_ranges = self.joint_pos_upper_limits - self.joint_pos_lower_limits + 1e-6

        # Initialize joint velocity targets and previous actions
        self.joint_vel_targets = torch.zeros((self.num_envs, self.snake_robot.num_joints), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.snake_robot.num_joints), device=self.device)

        # Initialize oscillation controller if enabled
        if self.cfg.enable_oscillation_controller:
            self.oscillation_controller = OscillationController(
                cfg=self.cfg.testing, num_joints=self.snake_robot.num_joints, device=self.device
            )
            print(f"[Info] Oscillation controller initialized with pattern: {self.cfg.testing.oscillation_type}")

        # Initialize observation history if enabled
        self.use_observation_history = self.cfg.observation_history.enable
        if self.use_observation_history:
            self.history_length = self.cfg.observation_history.history_length
            print(f"[Info] Observation history enabled with {self.history_length} frames.")

            # Initialize the observation history buffer with zeros
            # Shape: [num_envs, history_length, single_obs_size]
            # Get the size of a single observation using the helper method
            single_obs_size = self._get_single_observation_size()
            self.obs_history = torch.zeros((self.num_envs, self.history_length, single_obs_size), device=self.device)

        # Initialize frame visualization for virtual chassis
        self._setup_virtual_chassis_frame_markers()

        # Cache common data tensors (optional)
        self.joint_pos = self.snake_robot.data.joint_pos
        self.joint_vel = self.snake_robot.data.joint_vel
        self.root_state = self.snake_robot.data.root_state_w

    def _setup_scene(self):
        # Create snake robot articulation
        self.snake_robot = Articulation(self.cfg.robot)

        # add ground plane
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

        # Setup target visualization markers if enabled
        if self.cfg.target_position.show_marker:
            self._setup_target_markers()

    def _setup_target_markers(self):
        """Setup visualization markers for target positions."""
        # Configure the visualization markers
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/TargetMarkers",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=self.cfg.target_position.marker_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.target_position.marker_color),
                ),
            },
        )
        self.target_markers = VisualizationMarkers(marker_cfg)

        # Initialize marker positions and orientations
        self.marker_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.marker_orientations[..., 3] = 1.0  # Set to identity quaternion
        self.marker_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Update initial marker positions
        self._update_target_markers()

    def _update_target_markers(self, env_ids: Sequence[int] | None = None):
        """Update target marker positions.

        Args:
            env_ids: Optional list of environment IDs to update. If None, updates all environments.
        """
        if not hasattr(self, "target_markers"):
            return

        # Determine which environments to update
        if env_ids is None:
            env_ids = range(self.num_envs)

        # Update marker positions for specified environments
        for env_idx in env_ids:
            env_origin = self.scene.env_origins[env_idx]
            self.marker_positions[env_idx] = env_origin + torch.tensor(
                self.cfg.target_position.target_pos, device=self.device
            )

        # Update visualization
        self.target_markers.visualize(
            self.marker_positions, self.marker_orientations, marker_indices=self.marker_indices
        )

    def _setup_virtual_chassis_frame_markers(self):
        """Setup frame visualization markers for the virtual chassis."""

        # Configure frame markers (single frame showing X, Y, Z axes)
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/VirtualChassisAxes",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.5, 0.5, 0.5),
                ),
            },
        )
        self.virtual_chassis_axes = VisualizationMarkers(marker_cfg)

        # Initialize frame marker arrays
        # We need 1 frame marker per environment
        self.axis_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.axis_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.axis_orientations[..., 3] = 1.0  # Initialize to identity quaternion

        # Create marker indices: all environments use the same "frame" marker type (index 0)
        self.axis_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._compute_virtual_chassis()

    def _update_virtual_chassis_frame_visualization(self):
        """Update frame visualization for the virtual chassis in world frame."""
        if not hasattr(self, "virtual_chassis_axes"):
            return

        # Update frame marker positions and orientations for each environment
        for env_idx in range(self.num_envs):
            vc_position = self.virtual_chasis_com_world[env_idx]  # Virtual chassis center in world
            vc_rotation = self.virtual_chasis_rot_mat[env_idx]  # Virtual chassis rotation in world

            # Convert rotation matrix to quaternion for the frame marker
            vc_quaternion = quat_from_matrix(vc_rotation)

            env_origin = self.scene.env_origins[env_idx]

            # Update position and orientation for this environment's frame marker
            self.axis_positions[env_idx] = vc_position
            self.axis_orientations[env_idx] = vc_quaternion
            # Print center of mass data for debugging
            print(f"[Debug] Env {env_idx}: Origin: ", env_origin)
            print(f"[Debug] Env {env_idx}: Virtual chassis: ", self.axis_positions[env_idx])

        # Update the visualization
        self.virtual_chassis_axes.visualize(
            translations=self.axis_positions,
            orientations=self.axis_orientations,
            marker_indices=self.axis_indices,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.env_step_counter += 1

        # Store action for smoothness calculations in reward
        self.prev_actions = self.joint_vel_targets.clone()

        # Choose between oscillation control or policy control
        if self.cfg.enable_oscillation_controller:
            self._apply_oscillation_control(actions)
        else:
            self._apply_policy_control(actions)

    def _apply_oscillation_control(self, actions: torch.Tensor) -> None:
        """Apply oscillation patterns for testing snake locomotion."""
        # Generate velocity targets using the oscillation controller
        current_time = self.sim.current_time
        velocity_targets = self.oscillation_controller.generate_velocity_targets(current_time)

        # Apply to all environments
        self.joint_vel_targets[:] = velocity_targets.unsqueeze(0).expand(self.num_envs, -1)

        # Set zero actions for reward calculations
        self.actions = torch.zeros_like(actions)

    def _apply_policy_control(self, actions: torch.Tensor) -> None:
        """Apply velocity control based on policy actions.

        Args:
            actions: Policy actions in range [-1, 1]. Shape: [num_envs, num_joints]
        """
        # Process and clamp actions
        self.actions = actions.clone().clamp_(-1.0, 1.0)

        # Scale normalized actions to velocity targets
        # Map [-1, 1] to desired velocity range using action_scale
        velocity_targets = self.cfg.action_scale * self.actions

        # Set joint velocity targets directly
        self.joint_vel_targets[:] = velocity_targets

    def _apply_action(self) -> None:
        # Use velocity control instead of position control
        self.snake_robot.set_joint_velocity_target(self.joint_vel_targets)

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
                # Target position relative to end effector (shape: num_envs x 3)
                torch.zeros((self.num_envs, 3), device=self.device),
            ),
            dim=-1,
        )

        # Return the size of the last dimension (observation features)
        return single_obs.shape[-1]

    def _get_observations(self) -> dict:
        # Updates the virtual chasis com and rotation matrix
        self._compute_virtual_chassis()

        # Get joint positions and velocities
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel

        # Calculate joint positions normalized to [-1, 1]
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits) / self.joint_pos_ranges - 1.0

        # Normalize joint velocities to [-1, 1] based on velocity limits
        velocity_limit = torch.tensor(self.cfg.robot.actuators["snake_joints"].velocity_limit, device=self.device)
        joint_vel_normalized = joint_vel / velocity_limit  # This will be in [-1, 1] when velocity is at limits

        # Get end-effector position in world frame
        link_positions_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        last_link_idx = link_positions_w.shape[1] - 1
        end_effector_pos = link_positions_w[:, last_link_idx]  # Shape: [num_envs, 3]

        # Calculate target position relative to the end effector
        target_pos_world = self.scene.env_origins + self.target_position.unsqueeze(0)
        target_pos_relative = target_pos_world - end_effector_pos

        # Combine observations without root state information
        current_obs = torch.cat(
            (
                joint_pos_normalized,  # Normalized joint positions (9)
                joint_vel_normalized,  # Normalized joint velocities (9)
                target_pos_relative,  # Target position relative to end effector (3)
            ),
            dim=-1,
        )

        if self.use_observation_history:
            # Shift the history buffer (discard oldest, make room for newest)
            self.obs_history = self.obs_history.roll(-1, dims=1)

            # Insert the current observation as the newest entry
            self.obs_history[:, -1, :] = current_obs

            # Flatten the history for the policy
            policy_obs = self.obs_history.reshape(self.num_envs, -1)

            observations = {"policy": policy_obs}
        else:
            observations = {"policy": current_obs}

        # Update logs with tracking and observation data
        self._update_logs(observations)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Calculate rewards using an LQR-style quadratic cost function for fixed-base snake robot.
        Total reward = -(state_cost + control_cost) + alive_bonus + success_bonus
        where state_cost = x^T Q x and control_cost = u^T R u
        """
        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Get current state information
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        root_pos_w = self.snake_robot.data.root_pos_w  # Need root position to calculate target in world frame

        # Get end-effector (last link) position
        link_positions_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        last_link_idx = link_positions_w.shape[1] - 1
        end_effector_pos = link_positions_w[:, last_link_idx]  # Shape: [num_envs, 3]

        # --- State Costs (x^T Q x) ---

        # 1. Joint position cost (deviation from zero/neutral position)
        joint_pos_cost = self.cfg.lqr_reward.joint_pos_cost * torch.sum(joint_pos**2, dim=1)

        # 2. Joint velocity cost
        joint_vel_cost = self.cfg.lqr_reward.joint_vel_cost * torch.sum(joint_vel**2, dim=1)

        # 3. End-effector position cost (deviation from target)
        # Calculate target position in world frame for each environment (relative to root)
        target_pos_w = root_pos_w + self.target_position.unsqueeze(0)  # [num_envs, 3]
        end_effector_cost = self.cfg.lqr_reward.end_effector_cost * torch.sum(
            (end_effector_pos - target_pos_w) ** 2, dim=1
        )

        # Total state cost
        state_cost = joint_pos_cost + joint_vel_cost + end_effector_cost

        # --- Control Costs (u^T R u) ---
        # Use the commanded joint velocities as control inputs
        control_cost = self.cfg.lqr_reward.control_cost * torch.sum(self.joint_vel_targets**2, dim=1)

        # --- Additional Reward Terms ---

        # Check if target reached (within threshold)
        distance_to_target = torch.norm(end_effector_pos - target_pos_w, dim=1)
        threshold = self.cfg.target_position.success_distance_threshold
        newly_reached = (distance_to_target < threshold) & (~self.target_reached)
        self.target_reached = self.target_reached | newly_reached

        # Success bonus for reaching target
        success_bonus = torch.zeros_like(distance_to_target)
        success_bonus[newly_reached] = self.cfg.lqr_reward.success_bonus

        # Alive bonus
        alive_bonus = self.cfg.lqr_reward.alive_bonus

        # --- Total Reward ---
        # Negative cost plus bonuses
        total_reward = -(state_cost + control_cost) + success_bonus + alive_bonus

        # Update logs
        # Update closest distance tracker
        self.closest_distance = torch.minimum(self.closest_distance, distance_to_target)

        self.extras["log"].update({
            "Rewards/joint_pos_cost": joint_pos_cost.mean().item(),
            "Rewards/joint_vel_cost": joint_vel_cost.mean().item(),
            "Rewards/end_effector_cost": end_effector_cost.mean().item(),
            "Rewards/state_cost": state_cost.mean().item(),
            "Rewards/control_cost": control_cost.mean().item(),
            "Rewards/success_bonus": success_bonus.mean().item(),
            "Rewards/alive_bonus": alive_bonus,
            "Rewards/total_reward": total_reward.mean().item(),
            "Rewards/distance_to_target": distance_to_target.mean().item(),
            "Rewards/closest_distance": self.closest_distance.mean().item(),
            "Rewards/targets_reached": torch.sum(self.target_reached).item(),
        })

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time-based termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Get joint position bounds check
        self.joint_pos = self.snake_robot.data.joint_pos
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_lower_limits, dim=1) | torch.any(
            self.joint_pos > self.joint_pos_upper_limits, dim=1
        )

        # Combine all termination conditions
        # terminated = out_of_bounds | vel_violation | torque_violation
        terminated = out_of_bounds
        if "log" not in self.extras:  # Initialize if not present
            self.extras["log"] = {}

        # Add termination info to extras["log"]
        self.extras["log"].update({
            "terminations/joint_out_of_bounds": torch.sum(out_of_bounds).item(),
            "terminations/time_out": torch.sum(time_out).item(),
        })

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = []  # Use empty list instead of tensor to avoid type error
        super()._reset_idx(env_ids)

        # Reset target reached status for reset environments
        if len(env_ids) > 0:
            self.target_reached[env_ids] = False
            self.closest_distance[env_ids] = torch.ones(len(env_ids), device=self.device) * 100.0

            # Update marker positions for reset environments
            self._update_target_markers(env_ids)
        else:
            self.target_reached = torch.zeros_like(self.target_reached)
            self.closest_distance = torch.ones_like(self.closest_distance) * 100.0

            # Update all marker positions
            self._update_target_markers()

        # Reset joint positions to a neutral pose with small noise
        # n_envs = len(env_ids) if env_ids is not None else self.num_envs

        # Start with default positions
        joint_pos = self.snake_robot.data.default_joint_pos[env_ids].clone()

        # Create sinusoidal pattern for joint positions
        # This creates a snake-like posture that's good for starting position
        amplitude = 0.2  # radians (~11.5 degrees)
        phase_diff = math.pi / 2  # 90 degrees phase difference between joints

        for i in range(self.snake_robot.num_joints):
            # Phase offset increases along the body
            phase_offset = i * phase_diff
            # Apply sinusoidal pattern
            joint_pos[:, i] = amplitude * torch.sin(torch.tensor(phase_offset))

        # Add small random noise to initial positions
        joint_pos += torch.randn_like(joint_pos) * 0.05

        # Ensure joints are within limits
        joint_pos = torch.clamp(joint_pos, self.joint_pos_lower_limits[env_ids], self.joint_pos_upper_limits[env_ids])

        # Zero velocities
        joint_vel = torch.zeros_like(joint_pos)

        # Reset root state
        default_root_state = self.snake_robot.data.default_root_state[env_ids].clone()
        # Add env origins offset to maintain proper positioning in multi-env setup
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write states to simulation
        self.snake_robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.snake_robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.snake_robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset action buffer
        if env_ids is not None:
            self.joint_vel_targets[env_ids] = joint_vel
            self.prev_actions[env_ids] = joint_vel

        # Reset observation history for the reset environments
        if self.use_observation_history:
            # Calculate joint positions normalized to [-1, 1]
            joint_pos_normalized = (
                2.0 * (joint_pos - self.joint_pos_lower_limits[env_ids]) / self.joint_pos_ranges[env_ids] - 1.0
            )

            # Normalize joint velocities (which are zero at reset)
            velocity_limit = torch.tensor(self.cfg.robot.actuators["snake_joints"].velocity_limit, device=self.device)
            joint_vel_normalized = joint_vel / velocity_limit

            # Get end-effector position in world frame
            link_positions_w = self.snake_robot.data.body_pos_w[env_ids]  # Shape: [num_reset_envs, num_links, 3]
            last_link_idx = link_positions_w.shape[1] - 1
            end_effector_pos = link_positions_w[:, last_link_idx]  # Shape: [num_reset_envs, 3]

            # Calculate target position relative to the end effector
            target_pos_world = self.scene.env_origins[env_ids] + self.target_position.unsqueeze(0)
            target_pos_relative = target_pos_world - end_effector_pos

            # Create the initial observation with target position included
            initial_obs = torch.cat(
                (
                    joint_pos_normalized,  # Normalized joint positions (9)
                    joint_vel_normalized,  # Normalized joint velocities (9)
                    target_pos_relative,  # Target position relative to end effector (3)
                ),
                dim=-1,
            )

            # Fill the entire history with the initial observation
            if len(env_ids) > 0:  # Only if there are environments to reset
                for t in range(self.history_length):
                    self.obs_history[env_ids, t, :] = initial_obs

    def _update_logs(self, obs_dict: dict) -> None:
        """Updates and logs various metrics for debugging and analysis."""
        self._log_tracking_data()
        self._log_last_link_data()
        self._log_mass_information()
        self._log_torque_data()
        self._log_observation_data(obs_dict)

    def _log_tracking_data(self) -> None:
        """Logs joint position tracking data when enabled."""
        if not self.cfg.position_tracking.enable:
            return

        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        env_id = self.cfg.position_tracking.env_id
        if self.track_all_joints:
            # Initialize sum for average calculation
            total_abs_error = 0.0

            # Log commanded and actual velocities for each joint
            for joint_idx in range(self.snake_robot.num_joints):
                commanded_vel = self.joint_vel_targets[env_id, joint_idx]
                actual_vel = self.snake_robot.data.joint_vel[env_id, joint_idx]

                self.extras["log"].update({
                    f"Tracking/Joint{joint_idx}/CommandedVelocity": commanded_vel.item(),
                    f"Tracking/Joint{joint_idx}/ActualVelocity": actual_vel.item(),
                })
                # Calculate and log error metrics
                error = commanded_vel - actual_vel
                abs_error = abs(error.item())
                total_abs_error += abs_error

                self.extras["log"].update({
                    f"Tracking/Joint{joint_idx}/Error": error.item(),
                    f"Tracking/Joint{joint_idx}/AbsError": abs_error,
                })

            # Calculate and log average absolute error across all joints
            avg_abs_error = total_abs_error / self.snake_robot.num_joints
            self.extras["log"]["Tracking/AverageAbsoluteError"] = avg_abs_error

    def _log_last_link_data(self) -> None:
        """Logs position data for the last link of the snake."""
        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Get all link positions in world frame
        link_positions_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        last_link_idx = link_positions_w.shape[1] - 1  # Get the index of the last link

        # Get position of last link for the visualization environment
        env_id = self.cfg.observation_visualization.env_id if self.cfg.observation_visualization.enable else 0
        last_link_pos_world = link_positions_w[env_id, last_link_idx]  # Shape: [3]

        # Get root position in world frame
        root_pos_world = self.snake_robot.data.root_pos_w[env_id]  # Shape: [3]

        # Calculate position relative to robot base
        last_link_pos_relative = last_link_pos_world - root_pos_world

        # Log the world frame position components
        self.extras["log"].update({
            "LastLink/WorldPosition/X": last_link_pos_world[0].item(),
            "LastLink/WorldPosition/Y": last_link_pos_world[1].item(),
            "LastLink/WorldPosition/Z": last_link_pos_world[2].item(),
        })

        # Log the robot-base-relative position components
        self.extras["log"].update({
            "LastLink/RelativePosition/X": last_link_pos_relative[0].item(),
            "LastLink/RelativePosition/Y": last_link_pos_relative[1].item(),
            "LastLink/RelativePosition/Z": last_link_pos_relative[2].item(),
        })

        # Calculate and log distances in world frame
        world_distance_from_origin = torch.norm(last_link_pos_world).item()
        world_planar_distance = torch.norm(last_link_pos_world[:2]).item()
        self.extras["log"]["LastLink/World/DistanceFromOrigin"] = world_distance_from_origin
        self.extras["log"]["LastLink/World/PlanarDistance"] = world_planar_distance

        # Calculate and log distances relative to robot base
        relative_distance = torch.norm(last_link_pos_relative).item()
        relative_planar_distance = torch.norm(last_link_pos_relative[:2]).item()
        self.extras["log"]["LastLink/Relative/DistanceFromBase"] = relative_distance
        self.extras["log"]["LastLink/Relative/PlanarDistance"] = relative_planar_distance

    def _log_mass_information(self) -> None:
        """Logs mass information for individual links and total mass."""
        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Get masses for all links
        link_masses = self.snake_robot.data.default_mass  # Shape: [num_envs, num_bodies]
        total_mass = torch.sum(link_masses, dim=1)  # Shape: [num_envs]

        # Log masses for visualization env
        env_id = self.cfg.observation_visualization.env_id if self.cfg.observation_visualization.enable else 0

        # Log individual link masses
        for link_idx in range(link_masses.shape[1]):
            self.extras["log"][f"Masses/Link{link_idx}"] = link_masses[env_id, link_idx].item()

        # Log total mass
        self.extras["log"]["Masses/TotalRobotMass"] = total_mass[env_id].item()

    def _log_torque_data(self) -> None:
        """Logs computed and applied torque data for joints."""
        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Get torque data
        joint_torques_computed = self.snake_robot.data.computed_torque  # Shape: [num_envs, num_joints]
        joint_torques_applied = self.snake_robot.data.applied_torque  # Shape: [num_envs, num_joints]

        # Get environment ID for logging
        env_id = self.cfg.observation_visualization.env_id if self.cfg.observation_visualization.enable else 0

        # Log torques for each joint
        for joint_idx in range(self.snake_robot.num_joints):
            self.extras["log"][f"Torques/Joint{joint_idx}/computed"] = joint_torques_computed[env_id, joint_idx].item()
            self.extras["log"][f"Torques/Joint{joint_idx}/applied"] = joint_torques_applied[env_id, joint_idx].item()

    def _log_observation_data(self, obs_dict: dict) -> None:
        """Logs observation data for visualization when enabled."""
        if not self.cfg.observation_visualization.enable:
            return

        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        env_id = self.cfg.observation_visualization.env_id

        # Log joint positions and velocities
        if "joint_pos" in self.cfg.observation_visualization.components_to_plot:
            for joint_idx in range(self.snake_robot.num_joints):
                self.extras["log"][f"Observations/Joint{joint_idx}/Position"] = self.snake_robot.data.joint_pos[
                    env_id, joint_idx
                ].item()

        if "joint_vel" in self.cfg.observation_visualization.components_to_plot:
            for joint_idx in range(self.snake_robot.num_joints):
                self.extras["log"][f"Observations/Joint{joint_idx}/Velocity"] = self.snake_robot.data.joint_vel[
                    env_id, joint_idx
                ].item()

        # Log root position (world and local frame)
        if "root_pos" in self.cfg.observation_visualization.components_to_plot:
            for i, axis in enumerate(["X", "Y", "Z"]):
                self.extras["log"].update({
                    f"Observations/Root/WorldPosition{axis}": self.snake_robot.data.root_pos_w[env_id, i].item(),
                    f"Observations/Root/LocalPosition{axis}": self.snake_robot.data.root_link_pos_w[env_id, i].item(),
                })

        # Log root linear velocity
        if "root_lin_vel" in self.cfg.observation_visualization.components_to_plot:
            for i, axis in enumerate(["X", "Y", "Z"]):
                self.extras["log"][f"Observations/Root/LinearVelocity{axis}"] = self.snake_robot.data.root_lin_vel_w[
                    env_id, i
                ].item()

        # Log root orientation (quaternion)
        if "root_quat" in self.cfg.observation_visualization.components_to_plot:
            for i, component in enumerate(["W", "X", "Y", "Z"]):
                self.extras["log"][f"Observations/Root/Quaternion{component}"] = self.snake_robot.data.root_quat_w[
                    env_id, i
                ].item()

        # Log flattened policy observation
        if "flattened_policy_obs" in self.cfg.observation_visualization.components_to_plot:
            if self.use_observation_history:
                policy_obs = self.obs_history[env_id].reshape(-1)
            else:
                policy_obs = obs_dict["policy"][env_id]

            for i in range(len(policy_obs)):
                self.extras["log"][f"Observations/PolicyObs/Dim{i}"] = policy_obs[i].item()

    def _compute_mass_weighted_com_world_frame(self) -> torch.Tensor:
        # Get body center of mass positions in world frame and ensure correct device
        body_com_pos_w = self.snake_robot.data.body_com_pos_w.to(self.device)  # [num_envs, num_bodies, 3]

        # Get body masses and ensure correct device
        body_masses = self.snake_robot.data.default_mass.to(self.device)  # [num_envs, num_bodies]

        # Calculate mass-weighted center of mass
        # COM = Σ(mass_i * position_i) / Σ(mass_i)
        total_mass = torch.sum(body_masses, dim=1, keepdim=True)  # [num_envs, 1]

        # Weight each body COM position by its mass
        weighted_positions = body_com_pos_w * body_masses.unsqueeze(-1)  # [num_envs, num_bodies, 3]

        # Sum weighted positions and divide by total mass
        mass_weighted_com = torch.sum(weighted_positions, dim=1) / total_mass  # [num_envs, 3]

        return mass_weighted_com

    def _compute_virtual_chassis(self) -> None:
        """
        Compute virtual chassis pose using SVD method from Rollinson 2012.

        This method implements the virtual chassis computation as described in:
        "Virtual Chassis for Snake Robots: Definition and Applications" by Rollinson et al.

        The virtual chassis is defined as a body frame whose origin is at the robot's center
        of mass and whose axes are aligned with the robot's principal moments of inertia.

        Args:
            link_positions_root_frame: [num_envs, num_links, 3] - Link frame positions relative to root frame
            prev_rotation_matrix: [num_envs, 3, 3] - Previous rotation matrix to prevent sign flips
            use_mass_weighted_com: If True, uses actual mass-weighted COM; if False, uses geometric centroid

        Returns:
            rotation_matrix: [num_envs, 3, 3] - Virtual chassis rotation matrix relative to root frame
            center_of_mass: [num_envs, 3] - Center of mass position relative to root frame
        """
        # Step 1: Compute center of mass of all links
        center_of_mass_world = self._compute_mass_weighted_com_world_frame()  # [num_envs, 3]

        # Ensure center_of_mass is on correct device
        self.virtual_chasis_com_world = center_of_mass_world.to(self.device)

        # Step 2: Create position matrix P relative to center of mass
        # P[i] = link_positions[i] - center_of_mass for each environment
        # P = link_positions_root_frame - center_of_mass.unsqueeze(1)  # [num_envs, num_links, 3]

        # Step 3: Compute SVD for each environment
        # We need to handle each environment separately for SVD
        self.virtual_chasis_rot_mat = torch.zeros((self.num_envs, 3, 3), device=self.device, dtype=torch.float32)

        # for env_idx in range(num_envs):
        #     # Get position matrix for this environment
        #     P_env = P[env_idx]  # [num_links, 3]

        #     # Compute SVD: P = U * S * V^T
        #     # V contains the eigenvectors of P^T * P (principal axes)
        #     try:
        #         U, S, Vt = torch.linalg.svd(P_env, full_matrices=False)
        #         V = Vt.T  # Convert V^T to V: [3, 3]

        #         # Step 4: Ensure right-handed coordinate system
        #         # Third singular vector should be cross product of first and second
        #         v1, v2 = V[:, 0], V[:, 1]
        #         v3_expected = torch.linalg.cross(v1, v2)

        #         # Ensure third column matches expected direction
        #         if torch.dot(V[:, 2], v3_expected) < 0:
        #             V[:, 2] = -V[:, 2]

        #         # Step 5: Handle sign consistency with previous timestep
        #         if prev_rotation_matrix is not None:
        #             prev_V = prev_rotation_matrix[env_idx].to(self.device)

        #             # Enforce positive dot products with previous frame to prevent flips
        #             for i in range(2):  # Only check first two vectors
        #                 if torch.dot(V[:, i], prev_V[:, i]) < 0:
        #                     V[:, i] = -V[:, i]

        #             # Recompute third vector to maintain right-handed system
        #             V[:, 2] = torch.linalg.cross(V[:, 0], V[:, 1])

        #         rotation_matrices[env_idx] = V

        #     except Exception as e:
        #         # Fallback to identity matrix if SVD fails
        #         print(f"Warning: SVD failed for environment {env_idx}, using identity matrix: {e}")
        #         rotation_matrices[env_idx] = torch.eye(3, device=self.device, dtype=torch.float32)

        # Update the virtual chassis frame marker if enabled
        self._update_virtual_chassis_frame_visualization()
