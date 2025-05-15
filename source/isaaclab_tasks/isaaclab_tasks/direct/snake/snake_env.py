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
from isaaclab.utils.math import sample_uniform, quat_rotate

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains import TerrainImporter

@configclass
class SnakeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 50.0
    action_scale = 0.26  # rad/s # Action scale determines how much the target velocity changes per RL step
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
        target_pos: tuple = (-1, 0.2, 1)
        # Which link to track for reaching the target (0 is root, higher numbers for other links)
        tracked_link_idx: int = 9  # Default to the 9th link (adjust based on model)
        # Scale for distance threshold (when to consider target reached)
        success_distance_threshold: float = 0.5  # in meters

    target_position: TargetPositionCfg = TargetPositionCfg()
    # -- End Target Position Configuration --

    # -- LQR Style Reward Parameters --
    @configclass
    class LQRRewardCfg:
        """Configuration for LQR-style reward function for fixed-base snake robot."""
        # State cost matrix diagonal elements (Q matrix)
        joint_pos_cost: float = 0.0      # Cost on joint position deviation
        joint_vel_cost: float = 0.0      # Cost on joint velocity
        end_effector_cost: float = 5.0   # Cost on end-effector position deviation from target
        
        # Control cost matrix diagonal elements (R matrix)
        control_cost: float = 0.01       # Cost on control inputs (joint velocities)
        
        # Additional reward terms
        alive_bonus: float = 0.1         # Small bonus for staying alive
        success_bonus: float = 100.0     # Bonus for reaching target

    lqr_reward: LQRRewardCfg = LQRRewardCfg()

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
            },
            pos=(0.0, 0.0, 0.0375),  # Initial base position (adjust height based on robot)
            rot=(0.0, 0.0, 0.0, 1.0), # Initial base orientation
        ),
        actuators={
            # Define actuators for your joints #TODO: tune all these parameters
            "snake_joints": ImplicitActuatorCfg(
                # Use regex matching your joint names, or list them
                joint_names_expr=["joint_[1-9]"], # Example regex
                effort_limit=50.0,   # (Nm) <<< Tune 
                velocity_limit=0.262,  # (15deg/s)(rad/s) <<< Tune 
                stiffness=0.0,       # Kp
                damping=100.0,       # Kd
                                     # Tau = kp * (x - x0) + kd * (v - v0)
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

    # -- Testing Configuration --
    @configclass
    class TestingCfg:
        """Configuration for testing modes."""
        # Set to True to override RL actions with manual oscillation
        enable_manual_oscillation: bool = False
        # Type of manual oscillation ('sidewinding' or 'constant')
        oscillation_type: str = 'constant'
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
        # --- Constant velocity parameters ---
        constant_velocity: float = 0.262  # rad/s, constant velocity for all joints

    testing: TestingCfg = TestingCfg()
    # --- END TESTING CONFIGURATION ---

    @configclass
    class PositionTrackingCfg:
        """Configuration for velocity tracking analysis."""
        enable: bool = True
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
        enable: bool = True 
        env_id: int = 0  # Which environment to visualize
        max_points: int = 1000  # Maximum number of data points to collect
        save_interval_s: float = 10.0  # How often to save plots (seconds)
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
            # Create tensor to hold velocity targets
            num_joints = self.snake_robot.num_joints
            velocity_targets = torch.zeros((num_joints,), device=self.device)
            
            if self.cfg.testing.oscillation_type == 'sidewinding':
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
            
            elif self.cfg.testing.oscillation_type == 'constant':
                # Set all joints to the same constant velocity
                velocity_targets.fill_(self.cfg.testing.constant_velocity)
            
            else:
                raise ValueError(f"Unknown oscillation type: {self.cfg.testing.oscillation_type}")
            
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
        end_effector_cost = self.cfg.lqr_reward.end_effector_cost * torch.sum((end_effector_pos - target_pos_w)**2, dim=1)
        
        # Total state cost
        state_cost = (
            joint_pos_cost +
            joint_vel_cost +
            end_effector_cost
        )
        
        # --- Control Costs (u^T R u) ---
        # Use the commanded joint velocities as control inputs
        control_cost = self.cfg.lqr_reward.control_cost * torch.sum(self.joint_vel_targets**2, dim=1)
        
        # --- Additional Reward Terms ---
        
        # Check if target reached (within threshold)
        distance_to_target = torch.norm(end_effector_pos - target_pos_w, dim=1)
        threshold = self.cfg.target_position.success_distance_threshold
        newly_reached = (distance_to_target < threshold) & (~self.target_reached)
        self.target_reached = self.target_reached | newly_reached
        
        # Update closest distance tracker
        self.closest_distance = torch.minimum(self.closest_distance, distance_to_target)
        
        # Success bonus for reaching target
        success_bonus = torch.zeros_like(distance_to_target)
        success_bonus[newly_reached] = self.cfg.lqr_reward.success_bonus
        
        # Alive bonus
        alive_bonus = self.cfg.lqr_reward.alive_bonus
        
        # --- Total Reward ---
        # Negative cost plus bonuses
        total_reward = -(state_cost + control_cost) + success_bonus + alive_bonus
        
        # Update logs
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
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_lower_limits, dim=1) | \
                        torch.any(self.joint_pos > self.joint_pos_upper_limits, dim=1)
        
        # Get velocity limit violations - using limit from actuator config
        joint_vel = self.snake_robot.data.joint_vel
        velocity_limit = torch.tensor(self.cfg.robot.actuators["snake_joints"].velocity_limit, 
                                    device=self.device)
        vel_violation = torch.any(torch.abs(joint_vel) > velocity_limit, dim=1)
        
        # Get torque limit violations - using limit from actuator config
        joint_torques = self.snake_robot.data.applied_torque
        torque_limit = torch.tensor(self.cfg.robot.actuators["snake_joints"].effort_limit, 
                                  device=self.device)
        torque_violation = torch.any(torch.abs(joint_torques) > torque_limit, dim=1)
        
        # Combine all termination conditions
        terminated = out_of_bounds | vel_violation | torque_violation
        
        if "log" not in self.extras:  # Initialize if not present
            self.extras["log"] = {}
    
        # Add termination info to extras["log"]
        self.extras["log"].update({
            "terminations/joint_out_of_bounds": torch.sum(out_of_bounds).item(),
            "terminations/velocity_violations": torch.sum(vel_violation).item(),
            "terminations/torque_violations": torch.sum(torque_violation).item(),
            # Log max values for debugging
            "terminations/max_velocity": torch.max(torch.abs(joint_vel)).item(),
            "terminations/max_torque": torch.max(torch.abs(joint_torques)).item(),
            # Log the actual limits being used
            "terminations/velocity_limit": self.cfg.robot.actuators["snake_joints"].velocity_limit,
            "terminations/torque_limit": self.cfg.robot.actuators["snake_joints"].effort_limit,
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
        else:
            self.target_reached = torch.zeros_like(self.target_reached)
            self.closest_distance = torch.ones_like(self.closest_distance) * 100.0
        
        # Reset joint positions to a neutral pose with small noise
        n_envs = len(env_ids) if env_ids is not None else self.num_envs
        
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
        joint_pos = torch.clamp(
            joint_pos,
            self.joint_pos_lower_limits[env_ids],
            self.joint_pos_upper_limits[env_ids]
        )
        
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
                    root_pos,               # Fixed root position
                    root_quat,              # Fixed root orientation
                    root_lin_vel,           # Zero root velocity (fixed base)
                    target_pos_relative,    # Target position relative to root
                ),
                dim=-1,
            )
            
            # Fill the entire history with the initial observation
            if len(env_ids) > 0:  # Only if there are environments to reset
                for t in range(self.history_length):
                    self.obs_history[env_ids, t, :] = initial_obs

    def _update_logs(self, obs_dict: dict) -> None:
        """Update logs with tracking and observation data."""
        # Initialize log dict if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Log tracking data if enabled
        if self.cfg.position_tracking.enable:
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

        # Track last link position
        # Get all link positions in world frame
        link_positions_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        last_link_idx = link_positions_w.shape[1] - 1  # Get the index of the last link
        
        # Get position of last link for the visualization environment
        env_id = self.cfg.observation_visualization.env_id if self.cfg.observation_visualization.enable else 0
        # env_id = 3012
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

        # Log mass information
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

        # Log actuator torques
        joint_torques_computed = self.snake_robot.data.computed_torque  # Shape: [num_envs, num_joints]
        joint_torques_applied = self.snake_robot.data.applied_torque  # Shape: [num_envs, num_joints]
        
        # Log torques for each joint
        for joint_idx in range(self.snake_robot.num_joints):
            self.extras["log"][f"Torques/Joint{joint_idx}/computed"] = joint_torques_computed[env_id, joint_idx].item()
            self.extras["log"][f"Torques/Joint{joint_idx}/applied"] = joint_torques_applied[env_id, joint_idx].item()
            
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
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    self.extras["log"].update({
                        f"Observations/Root/WorldPosition{axis}": self.snake_robot.data.root_pos_w[env_id, i].item(),
                        f"Observations/Root/LocalPosition{axis}": self.snake_robot.data.root_link_pos_w[env_id, i].item(),
                    })
            
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
                    policy_obs = self.obs_history[env_id].reshape(-1)
                else:
                    policy_obs = obs_dict["policy"][env_id]
                
                for i in range(len(policy_obs)):
                    self.extras["log"][f"Observations/PolicyObs/Dim{i}"] = policy_obs[i].item()
