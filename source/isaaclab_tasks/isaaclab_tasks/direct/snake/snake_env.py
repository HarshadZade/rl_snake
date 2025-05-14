# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
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
    episode_length_s = 50.0
    action_scale = 1.0  # rad/s - velocity control scale  #TODO: tune this
    action_space = 9    # 9 joints
    observation_space = 31  # Updated: 9 (joints) + 9 (vels) + 3 (pos) + 4 (quat) + 3 (vel) + 3 (target)
    state_space = 0
    link_length = 4.0  #TODO: Get this from the USD instead of hardcoding # Length of each link in meters, used for height termination

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # -- Target Position Configuration --
    @configclass
    class TargetPositionCfg:
        """Configuration for target position task."""
        # Target position relative to the root
        target_pos: tuple = (5, 3, 3)
        # Which link to track for reaching the target (0 is root, higher numbers for other links)
        tracked_link_idx: int = 9  # Default to the 9th link (adjust based on model)
        # Scale for position-based reward
        position_reward_scale: float = 10.0
        # Scale for distance threshold (when to consider target reached)
        success_distance_threshold: float = 0.5  # in meters
        # Success bonus reward
        success_bonus: float = 100.0

    target_position: TargetPositionCfg = TargetPositionCfg()
    # -- End Target Position Configuration --

    # -- Robot Configuration (Loading from USD)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot", # Standard prim path pattern
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_realistic_dim_v0.usd",
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
                effort_limit=1000.0,   # <<< Tune based on your robot's specs
                velocity_limit=10.0,  # <<< Tune based on your robot's specs
                stiffness=0.0,       # <<< Tune: Use >0 for position/velocity control
                damping=1000.0,        # <<< Tune: Use >0 for position/velocity control (helps stability)
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
    # Action scale now determines how much the target velocity changes per RL step
    action_scale = 1.0  # rad/s - For velocity control, scale can be larger than position control
    
    # reward scales 
    rew_scale_forward_velocity = 0.0  # Not using this in position target approach
    rew_scale_joint_vel_penalty = -0.0001  # Small penalty on joint velocities (control cost)
    rew_scale_termination = 0.0
    rew_scale_alive = 1.0  # Small alive bonus
    rew_scale_action_smoothness_penalty = 0.0
    rew_scale_lateral_velocity_penalty = 0.0
    rew_scale_joint_limit_penalty = -5.0  # Keep joint limit penalty
    
    # Position target reward parameters
    rew_scale_target_distance = 10.0  # Weight for target distance tracking term
    rew_scale_control_cost = -0.001  # Weight for control cost term
    
    # --- ADD TESTING CONFIGURATION ---
    @configclass
    class TestingCfg:
        """Configuration for testing modes."""
        # Set to True to override RL actions with manual oscillation
        enable_manual_oscillation: bool = True
        
        # --- Sidewinding parameters ---
        # Amplitude in degrees (will be converted to radians)
        amplitude_x_deg: float = 30.0  # Amplitude for even joints
        amplitude_y_deg: float = 30.0  # Amplitude for odd joints
        
        # Angular frequency
        omega_x: float = 5.0 * math.pi / 6.0  # Angular frequency for even joints
        omega_y: float = 5.0 * math.pi / 6.0  # Angular frequency for odd joints
        
        # Phase offset per joint
        delta_x: float = 2.0 * math.pi / 3.0  # Phase offset per even joint
        delta_y: float = 2.0 * math.pi / 3.0  # Phase offset per odd joint
        
        # Phase difference between even and odd joints
        phi: float = 0.0

    testing: TestingCfg = TestingCfg()
    # --- END TESTING CONFIGURATION ---

    @configclass
    class PositionTrackingCfg:
        """Configuration for velocity tracking analysis."""
        enable: bool = False
        env_id: int = 0     # Which environment to track
        track_all_joints: bool = True  # Whether to track all joints or just one
        joint_id: int = 0   # Which joint to track (if not tracking all)
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
    
    # --- ADD OBSERVATION VISUALIZATION CONFIG ---
    @configclass
    class ObservationVisualizationCfg:
        """Configuration for observation visualization."""
        enable: bool = False
        env_id: int = 0  # Which environment to visualize
        max_points: int = 1000  # Maximum number of data points to collect
        save_interval_s: float = 10.0  # How often to save plots (seconds)
        # Include all components by default
        components_to_plot: list = ["joint_pos", "joint_vel", "root_pos", "root_lin_vel", "root_quat", "flattened_policy_obs"]
    
    observation_visualization: ObservationVisualizationCfg = ObservationVisualizationCfg()
    # --- END OBSERVATION VISUALIZATION CONFIG ---

class SnakeEnv(DirectRLEnv):
    cfg: SnakeEnvCfg

    def __init__(self, cfg: SnakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._render = render_mode is not None
        
        self.action_scale = self.cfg.action_scale
        self.env_step_counter = 0
        
        # Define which link to track for target reaching
        self.tracked_link_idx = torch.tensor([self.cfg.target_position.tracked_link_idx], device=self.device)
        
        # Store target position
        self.target_position = torch.tensor(self.cfg.target_position.target_pos, device=self.device)
        
        # Add a buffer to track whether each environment has reached the target
        self.target_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Track closest distance to target for each environment (initialize with large value)
        self.closest_distance = torch.ones(self.num_envs, device=self.device) * 100.0
        
        # Print debug info about links at the first step
        self.printed_link_debug = False
        
        self.track_positions = self.cfg.position_tracking.enable
        if self.track_positions:
            self.tracking_env_id = self.cfg.position_tracking.env_id
            self.tracking_joint_id = self.cfg.position_tracking.joint_id
            self.track_all_joints = self.cfg.position_tracking.track_all_joints
            self.max_tracking_points = self.cfg.position_tracking.max_points
            
            if self.track_all_joints:
                print(f"[Info] Velocity tracking enabled for all joints in Env {self.tracking_env_id}.")
            else:
                print(f"[Info] Velocity tracking enabled for Env {self.tracking_env_id}, Joint {self.tracking_joint_id}.")
        
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
        
        # Initialize joint position limits
        self.joint_pos_limits = self.snake_robot.data.soft_joint_pos_limits
        self.joint_pos_lower_limits = self.joint_pos_limits[..., 0].to(self.device) # Ellipsis (...) means all preceding dims
        self.joint_pos_upper_limits = self.joint_pos_limits[..., 1].to(self.device)
        self.joint_pos_ranges = self.joint_pos_upper_limits - self.joint_pos_lower_limits + 1e-6
        self.joint_pos_mid = (self.joint_pos_lower_limits + self.joint_pos_upper_limits) / 2

        # Initialize velocity target buffers
        self.joint_vel_targets = torch.zeros((self.num_envs, self.snake_robot.num_joints), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.snake_robot.num_joints), device=self.device)

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
        self.prev_actions = self.joint_vel_targets.clone()
        
        # Check if manual oscillation test mode is enabled
        if self.cfg.testing.enable_manual_oscillation:
            # --- SIDEWINDING MOTION PATTERN ---
            current_time = self.sim.current_time

            # Parameters from config
            # Convert amplitude from degrees to radians
            amplitude_x_rad = math.radians(self.cfg.testing.amplitude_x_deg)
            amplitude_y_rad = math.radians(self.cfg.testing.amplitude_y_deg)
            
            # Angular frequencies
            omega_x = self.cfg.testing.omega_x
            omega_y = self.cfg.testing.omega_y
            
            # Phase offsets
            delta_x = self.cfg.testing.delta_x
            delta_y = self.cfg.testing.delta_y
            
            # Phase difference between patterns
            phi = self.cfg.testing.phi
            
            # Number of joints
            num_joints = self.snake_robot.num_joints
            
            # Create tensor to hold velocity targets
            velocity_targets = torch.zeros((num_joints,), device=self.device)
            
            # Calculate velocity for each joint (derivative of position function)
            for i in range(num_joints):
                if i % 2 == 0:  # Even joints
                    # velocity(n,t) = Ax * wx * cos(wx*t + n*deltax)
                    velocity_targets[i] = amplitude_x_rad * omega_x * torch.cos(
                        torch.tensor(omega_x * current_time + i * delta_x)
                    )
                else:  # Odd joints
                    # velocity(n,t) = Ay * wy * cos(wy*t + n*deltay + phi)
                    velocity_targets[i] = amplitude_y_rad * omega_y * torch.cos(
                        torch.tensor(omega_y * current_time + i * delta_y + phi)
                    )
            
            # Expand to all environments
            self.joint_vel_targets[:] = velocity_targets.unsqueeze(0).expand(self.num_envs, -1)
            
            # Set self.actions for potential use in reward calculations
            self.actions = torch.zeros_like(actions)
            
        else:
            # Process actions from the policy for VELOCITY CONTROL
            self.actions = actions.clone().clamp_(-1.0, 1.0)

            # Scale normalized actions to velocity targets
            # Map [-1, 1] to desired velocity range using action_scale
            velocity_targets = self.action_scale * self.actions
            
            # Set joint velocity targets directly - no need to accumulate like position
            self.joint_vel_targets[:] = velocity_targets

    def _apply_action(self) -> None:
        # Use velocity control instead of position control
        # self.joint_vel_targets[:] = 0.0
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
                # Root position (shape: num_envs x 3)
                torch.zeros((self.num_envs, 3), device=self.device),
                # Root orientation (shape: num_envs x 4)
                torch.zeros((self.num_envs, 4), device=self.device),
                # Root linear velocity (shape: num_envs x 3)
                torch.zeros((self.num_envs, 3), device=self.device),
                # Target position (shape: num_envs x 3)
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

        # Calculate joint positions normalized to [-1, 1]
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits) / self.joint_pos_ranges - 1.0
        
        # Get root state information in world frame
        root_pos_w = self.snake_robot.data.root_pos_w
        root_quat = self.snake_robot.data.root_quat_w
        root_lin_vel_w = self.snake_robot.data.root_lin_vel_w
        
        # Convert positions from world frame to spawn-relative frame
        root_pos = root_pos_w - self.scene.env_origins
        
        # Use the same spawn-relative frame for velocities
        # No rotation needed - we want to keep measuring velocity relative to the world axes
        # just like the position
        root_lin_vel = root_lin_vel_w  # Keep velocities in world frame orientation
        
        # Calculate target position relative to the robot's current position
        # This helps the robot understand where the target is in its local frame
        target_pos_relative = self.target_position.unsqueeze(0).expand(self.num_envs, -1)
        
        # Combine observations (current frame only) including target position
        current_obs = torch.cat(
            (
                joint_pos_normalized,   # Normalized joint positions
                joint_vel * 0.1,        # Scaled joint velocities
                root_pos,               # Root position (spawn-relative frame)
                root_quat,              # Root orientation
                root_lin_vel,           # Root linear velocity (world frame orientation)
                target_pos_relative,    # Target position (relative to root)
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
            
            observations = {"policy": policy_obs}
        else:
            # Just use the current observation if history is disabled
            observations = {"policy": current_obs}
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Calculate rewards based on distance to target position.
        Reward structure:
        1. Negative exponential of distance to target (closer = higher reward)
        2. Success bonus for reaching the target
        3. Control cost for joint velocities
        4. Joint limit penalties for safety
        """
        # Get current state information
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        
        # --- Target distance component ---
        # Get root position in world frame
        root_pos_w = self.snake_robot.data.root_pos_w  # [num_envs, 3]
        
        # Get tracked link position (for reaching the target)
        link_positions_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        
        # Ensure tracked link index is valid
        num_links = link_positions_w.shape[1]
        tracked_idx = torch.clamp(self.tracked_link_idx, 0, num_links - 1)[0]
        
        # Get the position of the tracked link
        tracked_link_pos = link_positions_w[:, tracked_idx]  # [num_envs, 3]
        
        # Calculate target position in world frame for each environment
        target_pos_w = root_pos_w + self.target_position.unsqueeze(0)  # [num_envs, 3]
        
        # Calculate distance to target
        distance_to_target = torch.norm(tracked_link_pos - target_pos_w, dim=1)  # [num_envs]
        
        # Update closest distance tracker
        self.closest_distance = torch.minimum(self.closest_distance, distance_to_target)
        
        # Check if target reached (within threshold)
        threshold = self.cfg.target_position.success_distance_threshold
        newly_reached = (distance_to_target < threshold) & (~self.target_reached)
        self.target_reached = self.target_reached | newly_reached
        
        # Calculate distance-based reward (exponential of negative distance)
        # This gives higher reward as robot gets closer to target
        distance_reward = self.cfg.rew_scale_target_distance * torch.exp(-distance_to_target)
        
        # Add success bonus for environments that just reached the target
        success_bonus = torch.zeros_like(distance_reward)
        success_bonus[newly_reached] = self.cfg.target_position.success_bonus
        
        # --- Control cost component ---
        # Penalize control effort (joint velocities)
        control_cost = self.cfg.rew_scale_control_cost * torch.sum(joint_vel**2, dim=1)
        
        # --- Joint limit penalty (safety constraint) ---
        normalized_joint_pos = (joint_pos - self.joint_pos_mid) / (self.joint_pos_ranges / 2)
        joint_limit_penalty = self.cfg.rew_scale_joint_limit_penalty * torch.sum(
            torch.maximum(torch.abs(normalized_joint_pos) - 0.5, torch.zeros_like(normalized_joint_pos))**2, 
            dim=1
        )
        
        # --- Alive bonus ---
        alive_bonus = self.cfg.rew_scale_alive
        
        # Total reward
        total_reward = distance_reward + success_bonus + control_cost + joint_limit_penalty + alive_bonus
        
        # Log components for debugging
        self.extras["log"] = {
            "distance_to_target": distance_to_target.mean().item(),
            "closest_distance": self.closest_distance.mean().item(),
            "distance_reward": distance_reward.mean().item(),
            "success_bonus": success_bonus.mean().item(),
            "targets_reached": torch.sum(self.target_reached).item(),
            "control_cost": control_cost.mean().item(),
            "joint_limit_penalty": joint_limit_penalty.mean().item(),
            "alive_bonus": alive_bonus,
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
        
        # # terminated = head_too_low | too_tilted
        
        # Get joint position bounds check
        self.joint_pos = self.snake_robot.data.joint_pos
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_lower_limits, dim=1) | \
                        torch.any(self.joint_pos > self.joint_pos_upper_limits, dim=1) #1.57-0.524 rad = 60 deg. restricting the max angles 
        
        # Get body position world coordinates
        body_pos_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        
        # Check if any link's height (z-coordinate) exceeds the link length
        # The 3rd component (index 2) of the position vector is the z coordinate (height)
        link_heights = body_pos_w[:, :, 2]  # Extract heights for all links
        too_high = torch.any(link_heights > self.cfg.link_length, dim=1)
        
        # Combine all termination conditions
        # terminated = out_of_bounds | too_high
        
        terminated = out_of_bounds 
        
        # Add termination reason info to extras
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        
        self.extras["episode"]["terminations/joint_out_of_bounds"] = torch.sum(out_of_bounds).item()
        self.extras["episode"]["terminations/link_too_high"] = torch.sum(too_high).item()
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = []  # Use empty list instead of tensor to avoid type error
        super()._reset_idx(env_ids)
        
        # Reset target reached status for reset environments
        if len(env_ids) > 0:
            self.target_reached[env_ids] = False
            self.closest_distance[env_ids] = torch.ones(len(env_ids), device=self.device) * 100.0
        else:
            self.target_reached = torch.zeros_like(self.target_reached)
            self.closest_distance = torch.ones_like(self.closest_distance) * 100.0
        
        # Reset joint positions to a neutral pose with small noise
        n_envs = len(env_ids) if env_ids is not None else self.num_envs
        
        # Start with default positions
        joint_pos = self.snake_robot.data.default_joint_pos[env_ids].clone()
        
        # Option: Add a sinusoidal pattern as starting position
        # This can help the robot start with a sensible snake-like posture
        if True:
            for i in range(self.snake_robot.num_joints):  # Fixed: use num_joints instead of action_space
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
            self.joint_vel_targets[env_ids] = joint_vel
            self.prev_actions[env_ids] = joint_vel
            
        # Reset observation history for the reset environments
        if self.use_history:
            # Get the current observation for these environments
            joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits[env_ids]) / self.joint_pos_ranges[env_ids] - 1.0
            root_pos = default_root_state[:, :3]
            root_quat = default_root_state[:, 3:7]
            root_lin_vel = torch.zeros_like(root_pos)  # Zero velocity on reset
            
            # Include target position in the observation
            target_pos_relative = self.target_position.unsqueeze(0).expand(len(env_ids), -1)
            
            # Create the initial observation with target position included
            initial_obs = torch.cat(
                (
                    joint_pos_normalized,   # Normalized joint positions
                    joint_vel * 0.1,        # Scaled joint velocities (zeros)
                    root_pos,               # Root position
                    root_quat,              # Root orientation
                    root_lin_vel,           # Root linear velocity (zeros)
                    target_pos_relative,    # Target position relative to root
                ),
                dim=-1,
            )
            
            # Fill the entire history with the initial observation
            if len(env_ids) > 0:  # Only if there are environments to reset
                for t in range(self.history_length):
                    self.obs_history[env_ids, t, :] = initial_obs

    # --- Override the step method ---
    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Get observations before stepping
        obs_dict = self._get_observations()
        
        # Apply actions
        self._pre_physics_step(actions)
        
        # Step physics
        for _ in range(self.cfg.decimation):
            self.sim.step(render=self._render)
            self.sim.render()
        
        # Get observations after stepping
        obs_dict = self._get_observations()
        
        # Get rewards and dones
        rew = self._get_rewards()
        terminated, truncated = self._get_dones()
        
        # Initialize extras if not present
        if "log" not in self.extras:
            self.extras["log"] = dict()
            
        # Log tracking data if enabled
        if self.cfg.position_tracking.enable:
            env_id = self.cfg.position_tracking.env_id
            if self.track_all_joints:
                # Log commanded and actual velocities for each joint
                for joint_idx in range(self.snake_robot.num_joints):
                    self.extras["log"][f"Tracking/Joint{joint_idx}/CommandedVelocity"] = self.joint_vel_targets[env_id, joint_idx].item()
                    self.extras["log"][f"Tracking/Joint{joint_idx}/ActualVelocity"] = self.snake_robot.data.joint_vel[env_id, joint_idx].item()
                    
                    # Calculate and log error metrics
                    error = self.joint_vel_targets[env_id, joint_idx] - self.snake_robot.data.joint_vel[env_id, joint_idx]
                    self.extras["log"][f"Tracking/Joint{joint_idx}/Error"] = error.item()
                    self.extras["log"][f"Tracking/Joint{joint_idx}/AbsError"] = abs(error.item())
            else:
                joint_idx = self.cfg.position_tracking.joint_id
                self.extras["log"][f"Tracking/Joint{joint_idx}/CommandedVelocity"] = self.joint_vel_targets[env_id, joint_idx].item()
                self.extras["log"][f"Tracking/Joint{joint_idx}/ActualVelocity"] = self.snake_robot.data.joint_vel[env_id, joint_idx].item()
                error = self.joint_vel_targets[env_id, joint_idx] - self.snake_robot.data.joint_vel[env_id, joint_idx]
                self.extras["log"][f"Tracking/Joint{joint_idx}/Error"] = error.item()
                self.extras["log"][f"Tracking/Joint{joint_idx}/AbsError"] = abs(error.item())
        
        # Log observation data if enabled
        if self.cfg.observation_visualization.enable:
            env_id = self.cfg.observation_visualization.env_id
            
            # Log joint positions and velocities
            if "joint_pos" in self.cfg.observation_visualization.components_to_plot:
                for joint_idx in range(self.snake_robot.num_joints):
                    self.extras["log"][f"Observations/Joint{joint_idx}/Position"] = self.snake_robot.data.joint_pos[env_id, joint_idx].item()
            
            if "joint_vel" in self.cfg.observation_visualization.components_to_plot:
                for joint_idx in range(self.snake_robot.num_joints):
                    self.extras["log"][f"Observations/Joint{joint_idx}/Velocity"] = self.snake_robot.data.joint_vel[env_id, joint_idx].item()
            
            # Log root position (world and local frame)
            if "root_pos" in self.cfg.observation_visualization.components_to_plot:
                # World frame position
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    self.extras["log"][f"Observations/Root/WorldPosition{axis}"] = self.snake_robot.data.root_pos_w[env_id, i].item()
                    self.extras["log"][f"Observations/Root/LocalPosition{axis}"] = self.snake_robot.data.root_link_pos_w[env_id, i].item()
            
            # Log root linear velocity
            if "root_lin_vel" in self.cfg.observation_visualization.components_to_plot:
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    self.extras["log"][f"Observations/Root/LinearVelocity{axis}"] = self.snake_robot.data.root_lin_vel_w[env_id, i].item()
            
            # Log root orientation (quaternion)
            if "root_quat" in self.cfg.observation_visualization.components_to_plot:
                for i, component in enumerate(['W', 'X', 'Y', 'Z']):
                    self.extras["log"][f"Observations/Root/Quaternion{component}"] = self.snake_robot.data.root_quat_w[env_id, i].item()
            
            # Log flattened policy observation
            if "flattened_policy_obs" in self.cfg.observation_visualization.components_to_plot:
                if self.use_history:
                    policy_obs = self.obs_history[env_id].reshape(-1)  # Flatten the history for the selected environment
                else:
                    policy_obs = obs_dict["policy"][env_id]  # Get the policy observation for the selected environment
                
                # Log each dimension of the policy observation
                for i in range(len(policy_obs)):
                    self.extras["log"][f"Observations/PolicyObs/Dim{i}"] = policy_obs[i].item()
        
        return obs_dict, rew, terminated, truncated, self.extras
    # --- End override ---
    
    def save_position_tracking_plot(self):
        pass

    def save_observation_plots(self):
        pass