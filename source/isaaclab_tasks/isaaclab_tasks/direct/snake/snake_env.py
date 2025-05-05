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
    episode_length_s = 20.0
    action_scale = 0.5  # rad/s - velocity control scale  #TODO: tune this
    action_space = 9    # 9 joints
    observation_space = 18  # 9 joints * 2 values (normalized positions + scaled velocities)
    state_space = 0
    link_length = 4.0  #TODO: Get this from the USD instead of hardcoding # Length of each link in meters, used for height termination

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=40.0, replicate_physics=True)

    # -- Robot Configuration (Loading from USD)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot", # Standard prim path pattern
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_velocity-articulate-fixedjoint.usd",
            activate_contact_sensors=False, # Set to True if you need contact sensors #TODO: check this
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0, # Tune if needed
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=4, # Increase velocity iterations
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Define initial joint positions
            joint_pos={
                "joint_1": 0.0,
            },
            pos=(0.0, 0.0, 0.0),  # Initial base position (adjust height based on robot)
            rot=(0.0, 0.0, 0.0, 1.0), # Initial base orientation
        ),
        actuators={
            # Define actuators for your joints #TODO: tune all these parameters
            "snake_joints": ImplicitActuatorCfg(
                # Use regex matching your joint names, or list them
                joint_names_expr=["joint_[1-9]"], # Example regex
                effort_limit=100000.0,
                velocity_limit=5.0,      # Reduced from 10.0
                stiffness=10.0,          # Add stiffness for position holding (was 0.0)
                damping=1.0,             # Reduce damping to avoid over-damping (was 100000.0)
            ),
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
    rew_scale_forward_velocity = 0.0  # Not using this in fixed-base approach
    rew_scale_joint_vel_penalty = 0.0  # Not using this directly
    rew_scale_termination = -10.0  # Penalty for termination
    rew_scale_alive = 0.1  # Small alive bonus
    rew_scale_action_smoothness_penalty = 0.0  # Not using this
    rew_scale_lateral_velocity_penalty = 0.0  # Not using this
    rew_scale_joint_limit_penalty = -5.0  # Keep joint limit penalty
    
    # Position tracking reward parameters
    rew_scale_state_tracking = 10.0  # Weight for position tracking term (renamed from state_tracking)
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
        components_to_plot: list = ["joint_pos", "joint_vel", "root_pos", "root_lin_vel", "root_pos_w"]  # Which components to plot
    
    observation_visualization: ObservationVisualizationCfg = ObservationVisualizationCfg()
    # --- END OBSERVATION VISUALIZATION CONFIG ---

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
            self.track_all_joints = self.cfg.position_tracking.track_all_joints
            self.max_tracking_points = self.cfg.position_tracking.max_points
            
            # Initialize velocity tracking data structure 
            self.position_tracking_data = {
                "timesteps": [],
                "commanded_velocities": [],
                "actual_velocities": [],
            }
            
            if self.track_all_joints:
                print(f"[Info] Velocity tracking enabled for all joints in Env {self.tracking_env_id}.")
            else:
                print(f"[Info] Velocity tracking enabled for Env {self.tracking_env_id}, Joint {self.tracking_joint_id}.")
            print(f"       Plots will be saved when you terminate the simulation (Ctrl+C).")
            
            # Set up signal handler for SIGINT (Ctrl+C)
            import signal
            def signal_handler(sig, frame):
                print("\nCaught interrupt signal. Saving velocity tracking plot before exiting...")
                self.save_position_tracking_plot()
                # Also save root position debug plots
                if hasattr(self, 'visualize_observations') and self.visualize_observations:
                    print("Saving root position debug plots...")
                    self.save_root_position_debug_plot()
                print("Plots saved. Exiting...")
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
        
        # --- Initialize observation visualization ---
        self.visualize_observations = self.cfg.observation_visualization.enable
        if self.visualize_observations:
            self.visualization_env_id = self.cfg.observation_visualization.env_id
            self.max_visualization_points = self.cfg.observation_visualization.max_points
            self.components_to_plot = self.cfg.observation_visualization.components_to_plot
            
            # Initialize data structure for observation visualization
            self.observation_viz_data = {
                "timesteps": [],
                "joint_pos": [],
                "joint_vel": [],
            }
            
            print(f"[Info] Observation visualization enabled for Env {self.visualization_env_id}.")
            print(f"       Components being tracked: {self.components_to_plot}")
            print(f"       Plots will be saved when you terminate the simulation (Ctrl+C).")
            
            # Extend the signal handler to also save observation plots
            import signal
            original_sigint_handler = signal.getsignal(signal.SIGINT)
            
            def extended_signal_handler(sig, frame):
                print("\nCaught interrupt signal. Saving observation visualization plots before exiting...")
                self.save_observation_plots()
                
                # Explicitly save the root position debug plots as well
                print("Saving root position debug plots...")
                self.save_root_position_debug_plot()
                
                # Call the original handler if it exists
                if callable(original_sigint_handler):
                    original_sigint_handler(sig, frame)
                else:
                    print("Exiting...")
                    import sys
                    sys.exit(0)
            
            # Register the extended signal handler
            signal.signal(signal.SIGINT, extended_signal_handler)
        # --- End observation visualization initialization ---
 
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
        
        # Print environment origins for verification
        if self.cfg.scene.num_envs < 10:  # Only print if small number of envs
            for i in range(self.cfg.scene.num_envs):
                print(f"Env {i} origin: {self.scene.env_origins[i].cpu().numpy()}")
        
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
            
            # Apply velocity rate limiting for stability
            # Compute the change in velocity targets
            velocity_change = velocity_targets - self.joint_vel_targets
            
            # Limit the rate of change to 0.05 rad/s per step for stability
            max_change = 0.05
            limited_change = torch.clamp(velocity_change, -max_change, max_change)
            
            # Apply the limited change
            self.joint_vel_targets = self.joint_vel_targets + limited_change

    def _apply_action(self) -> None:
        # Use velocity control instead of position control
        self.joint_vel_targets[:] = 0.0
        self.snake_robot.set_joint_velocity_target(self.joint_vel_targets)
        
        # For debugging
        if self.env_step_counter % 100 == 0:
            # Log the joint positions and velocities of the first environment
            joint_pos = self.snake_robot.data.joint_pos[0].clone().cpu().numpy()
            joint_vel = self.snake_robot.data.joint_vel[0].clone().cpu().numpy()
            velocity_targets = self.joint_vel_targets[0].clone().cpu().numpy()
            print(f"Step {self.env_step_counter}:")
            print(f"  Joint positions: {joint_pos.round(3)}")
            print(f"  Joint velocities: {joint_vel.round(3)}")
            print(f"  Target velocities: {velocity_targets.round(3)}")
            # Check actuator configuration - for debugging only
            # Different IsaacLab versions may have different APIs, so use try/except
            try:
                print(f"  Actuator config from cfg: ")
                print(f"    Damping: {self.cfg.robot.actuators['snake_joints'].damping}")
                print(f"    Stiffness: {self.cfg.robot.actuators['snake_joints'].stiffness}")
                print(f"    Effort limit: {self.cfg.robot.actuators['snake_joints'].effort_limit}")
                print(f"    Velocity limit: {self.cfg.robot.actuators['snake_joints'].velocity_limit}")
            except:
                print("  Actuator configuration info not available")

    def _get_single_observation_size(self):
        """Calculate the size of a single observation (without history)."""
        # Get joint positions and velocities
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        
        # Calculate a single observation - for fixed base, only using joint data
        single_obs = torch.cat(
            (
                # Normalized joint positions (shape: num_envs x 9)
                torch.zeros_like(joint_pos),
                # Scaled joint velocities (shape: num_envs x 9)
                torch.zeros_like(joint_vel),
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
        
        # For fixed base, we only need joint positions and velocities
        # No need for root state information since the base is fixed
        current_obs = torch.cat(
            (
                joint_pos_normalized,   # Normalized joint positions
                joint_vel * 0.1,        # Scaled joint velocities
            ),
            dim=-1,
        )
        
        # Store observation data for visualization if enabled
        if self.visualize_observations and len(self.observation_viz_data["timesteps"]) < self.max_visualization_points:
            # Get root state for debugging/visualization even though not used in policy
            root_pos_w = self.snake_robot.data.root_pos_w
            root_quat = self.snake_robot.data.root_quat_w
            root_lin_vel_w = self.snake_robot.data.root_lin_vel_w
            
            # Convert positions from world frame to spawn-relative frame
            root_pos = root_pos_w - self.scene.env_origins
            
            # Keep velocities in world frame orientation
            root_lin_vel = root_lin_vel_w
            
            # Store timestep
            self.observation_viz_data["timesteps"].append(self.sim.current_time)
            
            # Store joint data
            self.observation_viz_data["joint_pos"].append(joint_pos_normalized[self.visualization_env_id].clone().cpu().numpy())
            self.observation_viz_data["joint_vel"].append((joint_vel * 0.1)[self.visualization_env_id].clone().cpu().numpy())
            
            # Initialize root state storage if not already done
            if "root_pos" not in self.observation_viz_data:
                self.observation_viz_data["root_pos"] = []
                self.observation_viz_data["root_quat"] = []
                self.observation_viz_data["root_lin_vel"] = []
                self.observation_viz_data["root_pos_w"] = []
                self.observation_viz_data["root_lin_vel_w"] = []
                self.observation_viz_data["env_origin"] = []
            
            # Store root state data for debugging
            self.observation_viz_data["root_pos"].append(root_pos[self.visualization_env_id].clone().cpu().numpy())
            self.observation_viz_data["root_quat"].append(root_quat[self.visualization_env_id].clone().cpu().numpy())
            self.observation_viz_data["root_lin_vel"].append(root_lin_vel[self.visualization_env_id].clone().cpu().numpy())
            self.observation_viz_data["root_pos_w"].append(root_pos_w[self.visualization_env_id].clone().cpu().numpy())
            self.observation_viz_data["root_lin_vel_w"].append(root_lin_vel_w[self.visualization_env_id].clone().cpu().numpy())
            self.observation_viz_data["env_origin"].append(self.scene.env_origins[self.visualization_env_id].clone().cpu().numpy())
            
            # Store flattened policy observation
            if "flattened_policy_obs" not in self.observation_viz_data:
                self.observation_viz_data["flattened_policy_obs"] = []
        
        if self.use_history:
            # Shift the history buffer (discard oldest, make room for newest)
            self.obs_history = self.obs_history.roll(-1, dims=1)
            
            # Insert the current observation as the newest entry
            self.obs_history[:, -1, :] = current_obs
            
            # Flatten the history for the policy
            # Shape goes from [num_envs, history_length, single_obs_size] 
            # to [num_envs, history_length * single_obs_size]
            policy_obs = self.obs_history.reshape(self.num_envs, -1)
            
            # Store the flattened policy observation for visualization
            if self.visualize_observations and len(self.observation_viz_data["timesteps"]) < self.max_visualization_points:
                self.observation_viz_data["flattened_policy_obs"].append(policy_obs[self.visualization_env_id].clone().cpu().numpy())
            
            observations = {"policy": policy_obs}
        else:
            # Just use the current observation if history is disabled
            observations = {"policy": current_obs}
            
            # Store the policy observation for visualization
            if self.visualize_observations and len(self.observation_viz_data["timesteps"]) < self.max_visualization_points:
                self.observation_viz_data["flattened_policy_obs"].append(current_obs[self.visualization_env_id].clone().cpu().numpy())
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Calculate rewards for fixed-base snake robot focused on position tracking.
        Reward = position_tracking_reward - control_cost + joint_limit_penalty + alive_bonus
        """
        # Get current state information
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        energy_penalty = self.cfg.rew_scale_joint_vel_penalty * torch.sum(joint_vel**2, dim=1)
        
        # --- Position tracking component ---
        # For fixed base control, we want to track target joint positions
        # We'll create a sinusoidal target pattern
        
        # Create time-based sinusoidal target positions
        current_time = self.sim.current_time
        num_joints = self.snake_robot.num_joints
        
        # Parameters for the oscillation pattern
        amplitude = 0.5  # 0.5 rad amplitude
        frequency = 0.5  # 0.5 Hz
        phase_offset_per_joint = torch.pi / 4  # 45 degrees phase difference between joints
        
        # Calculate target positions for each joint
        target_joint_pos = torch.zeros_like(joint_pos)
        for i in range(num_joints):
            # Even joints use sine, odd joints use cosine for alternating pattern
            if i % 2 == 0:
                target_joint_pos[:, i] = amplitude * torch.sin(
                    torch.tensor(2 * torch.pi * frequency * current_time + i * phase_offset_per_joint, device=self.device)
                )
            else:
                target_joint_pos[:, i] = amplitude * torch.cos(
                    torch.tensor(2 * torch.pi * frequency * current_time + i * phase_offset_per_joint, device=self.device)
                )
        
        # Calculate position tracking error (squared difference between actual and target)
        position_error = torch.sum((joint_pos - target_joint_pos) ** 2, dim=1)
        
        # Convert error to reward using exponential (higher when error is lower)
        position_tracking_reward = self.cfg.rew_scale_state_tracking * torch.exp(-5.0 * position_error)
        
        # --- Control cost component ---
        # Penalize control effort (joint velocities)
        control_cost = self.cfg.rew_scale_control_cost * torch.sum(joint_vel**2, dim=1)
        
        # --- Joint limit penalty (safety constraint) ---
        normalized_joint_pos = (joint_pos - self.joint_pos_mid) / (self.joint_pos_ranges / 2)
        joint_limit_penalty = self.cfg.rew_scale_joint_limit_penalty * torch.sum(
            torch.maximum(torch.abs(normalized_joint_pos) - 0.8, torch.zeros_like(normalized_joint_pos))**2, 
            dim=1
        )
        
        # --- Alive bonus ---
        alive_bonus = self.cfg.rew_scale_alive
        
        # Total reward
        total_reward = position_tracking_reward + control_cost + joint_limit_penalty + alive_bonus
        
        # Log components for debugging
        self.extras["log"] = {
            "position_tracking_reward": position_tracking_reward.mean().item(),
            "position_error": position_error.mean().item(),
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
        self.joint_pos = self.snake_robot.data.joint_pos
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_lower_limits, dim=1) | \
                        torch.any(self.joint_pos > self.joint_pos_upper_limits, dim=1)
        
        # Get joint position bounds check
        self.joint_pos = self.snake_robot.data.joint_pos
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_lower_limits + 0.524, dim=1) | \
                        torch.any(self.joint_pos > self.joint_pos_upper_limits - 0.524, dim=1) #1.57-0.524 rad = 60 deg. restricting the max angles 
        
        # Get body position world coordinates
        body_pos_w = self.snake_robot.data.body_pos_w  # Shape: [num_envs, num_links, 3]
        
        # Check if any link's height (z-coordinate) exceeds the link length
        # The 3rd component (index 2) of the position vector is the z coordinate (height)
        link_heights = body_pos_w[:, :, 2]  # Extract heights for all links
        too_high = torch.any(link_heights > self.cfg.link_length, dim=1)
        
        # Combine all termination conditions
        # terminated = out_of_bounds | too_high
        
        terminated = out_of_bounds 
        
        # Add logging information for debugging
        if self.env_step_counter % 100 == 0:  # Only log occasionally to avoid overhead
            num_too_high = torch.sum(too_high).item()
            if num_too_high > 0:
                # print(f"[Step {self.env_step_counter}] Terminating {num_too_high} environments due to links being too high")
                
                # For the first environment with high links, log details about which links are too high
                if too_high[0]:
                    first_env_heights = link_heights[0]
                    high_links = (first_env_heights > self.cfg.link_length).nonzero().flatten()
                    high_link_heights = first_env_heights[high_links]
                    # print(f"  Env 0: Links {high_links.cpu().numpy()} are too high with heights {high_link_heights.cpu().numpy()}")
        
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
        
        # Reset joint positions to a neutral pose with small noise
        n_envs = len(env_ids) if env_ids is not None else self.num_envs
        
        # Start with default positions (all zeros)
        joint_pos = self.snake_robot.data.default_joint_pos[env_ids].clone()
        
        # Start with all zeros - no sinusoidal pattern for stability
        # This gives the robot a chance to stabilize from a neutral position
        
        # Add very small random noise to initial positions (reduced from 0.05)
        joint_pos += torch.randn_like(joint_pos) * 0.01
        
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
        
        # For fixed-base robots, the base position must be exactly at the environment origin
        # Otherwise, all robots might be drawn to a common point (e.g., first robot's position)
        for i, env_id in enumerate(env_ids):
            default_root_state[i, :3] = self.scene.env_origins[env_id]
        
        # Write states to simulation
        self.snake_robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.snake_robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.snake_robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset action buffer - ensure zeros for stability
        if env_ids is not None:
            self.joint_vel_targets[env_ids] = torch.zeros_like(joint_vel)
            self.prev_actions[env_ids] = torch.zeros_like(joint_vel)
        else:
            self.joint_vel_targets = torch.zeros_like(joint_vel)
            self.prev_actions = torch.zeros_like(joint_vel)
            
        # Reset observation history for the reset environments
        if self.use_history:
            # Get the current observation for these environments
            joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits[env_ids]) / self.joint_pos_ranges[env_ids] - 1.0
            
            # Create the initial observation - FIXED: Removed root state components
            initial_obs = torch.cat(
                (
                    joint_pos_normalized,   # Normalized joint positions
                    joint_vel * 0.1,        # Scaled joint velocities (zeros)
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

        # Before taking the step, record the current time and commanded velocity
        current_time = self.sim.current_time
        
        # Store commanded velocities before physics step if tracking is enabled
        if self.track_positions and len(self.position_tracking_data["timesteps"]) < self.max_tracking_points:
            if self.track_all_joints:
                # Store commanded velocities for all joints in selected environment
                commanded_velocities = self.joint_vel_targets[self.tracking_env_id].clone().cpu().numpy()
            else:
                # Store just the single joint's commanded velocity
                commanded_velocities = self.joint_vel_targets[self.tracking_env_id, self.tracking_joint_id].item()

        # --- Call the original DirectRLEnv step logic ---
        # This handles: _pre_physics_step, sim.step(), _get_observations, _get_rewards, _get_dones, _reset_idx etc.
        obs_dict, rew, terminated, truncated, extras = super().step(actions)
        # --- End original step logic ---

        # --- Collect BOTH commanded and actual velocity data AFTER physics step ---
        if self.track_positions and len(self.position_tracking_data["timesteps"]) < self.max_tracking_points:
            if self.track_all_joints:
                # Get actual joint velocities for all joints after the physics step
                actual_velocities = self.snake_robot.data.joint_vel[self.tracking_env_id].clone().cpu().numpy()
                
                # Debug: print collection info occasionally
                if self.env_step_counter % 100 == 0:
                    print(f"Collecting data point {len(self.position_tracking_data['timesteps'])} at time {self.sim.current_time:.2f}s")
            else:
                # Get single joint's actual velocity after the physics step
                actual_velocities = self.snake_robot.data.joint_vel[self.tracking_env_id, self.tracking_joint_id].item()
                
                # Debug: print collection info occasionally
                if self.env_step_counter % 100 == 0:
                    print(f"Collecting data point {len(self.position_tracking_data['timesteps'])} at time {self.sim.current_time:.2f}s")
                    print(f"  Commanded velocity: {commanded_velocities:.4f}, Actual velocity: {actual_velocities:.4f}")
            
            # Save both data points
            self.position_tracking_data["timesteps"].append(current_time)
            self.position_tracking_data["commanded_velocities"].append(commanded_velocities)
            self.position_tracking_data["actual_velocities"].append(actual_velocities)

        return obs_dict, rew, terminated, truncated, extras
    # --- End override ---
    
    def save_position_tracking_plot(self):
        print("#############################################")
        print("Saving velocity tracking plot")
        print("#############################################")
        
        # Save the tracking data to a file
        if not self.position_tracking_data["timesteps"]:
            print("No velocity tracking data to plot!")
            return
            
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "joint_tracking_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get number of collected datapoints
        timesteps_count = len(self.position_tracking_data["timesteps"])
        
        if self.track_all_joints:
            # For multi-joint tracking
            
            # Convert data to numpy arrays for plotting
            times = np.array(self.position_tracking_data["timesteps"])
            
            # Number of joints
            num_joints = self.snake_robot.num_joints
            
            print(f"Plotting data: timesteps={timesteps_count}, joints={num_joints}")
            
            # Setup the plot - one figure with subplots for each joint
            fig, axes = plt.subplots(num_joints, 1, figsize=(12, 3*num_joints), sharex=True)
            
            # Create arrays to store error metrics
            rmse_values = np.zeros(num_joints)
            max_error_values = np.zeros(num_joints)
            avg_error_values = np.zeros(num_joints)
            
            # Extract data for each joint and plot
            for joint_idx in range(num_joints):
                # Extract data for this joint
                commanded = np.array([cmd[joint_idx] for cmd in self.position_tracking_data["commanded_velocities"]])
                actual = np.array([act[joint_idx] for act in self.position_tracking_data["actual_velocities"]])
                
                # Plot on the corresponding subplot
                ax = axes[joint_idx]
                ax.plot(times, commanded, 'b-', label='Commanded')
                ax.plot(times, actual, 'r-', label='Actual')
                
                # Calculate error metrics for this joint
                error = commanded - actual
                rmse_values[joint_idx] = np.sqrt(np.mean(np.square(error)))
                max_error_values[joint_idx] = np.max(np.abs(error))
                avg_error_values[joint_idx] = np.mean(np.abs(error))
                
                # Add metrics to the plot
                ax.text(0.02, 0.85, 
                      f'RMSE: {rmse_values[joint_idx]:.4f} rad/s\nMax Err: {max_error_values[joint_idx]:.4f} rad/s\nAvg Err: {avg_error_values[joint_idx]:.4f} rad/s',
                      transform=ax.transAxes,
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
                
                # Add labels and legend
                ax.set_ylabel(f'Joint {joint_idx} (rad/s)')
                ax.set_title(f'Joint {joint_idx} Velocity Tracking')
                ax.grid(True)
                if joint_idx == 0:
                    ax.legend(loc='upper right')
            
            # Common labels
            axes[-1].set_xlabel('Time (s)')
            fig.suptitle(f'Joint Velocity Tracking - All Joints in Env {self.tracking_env_id}')
            plt.tight_layout()
            
            # Save the multi-joint plot
            filename = os.path.join(output_dir, f'all_joints_velocity_tracking_env{self.tracking_env_id}_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            
            # Save a summary of the error metrics
            summary_filename = os.path.join(output_dir, f'velocity_tracking_metrics_summary_{timestamp}.txt')
            with open(summary_filename, 'w') as f:
                f.write(f"Joint Velocity Tracking Metrics Summary - Env {self.tracking_env_id}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write(f"{'Joint':<10}{'RMSE (rad/s)':<15}{'Max Error (rad/s)':<20}{'Avg Error (rad/s)':<20}\n")
                f.write("-" * 65 + "\n")
                for i in range(num_joints):
                    f.write(f"{i:<10}{rmse_values[i]:<15.4f}{max_error_values[i]:<20.4f}{avg_error_values[i]:<20.4f}\n")
                
                # Add averages across all joints
                f.write("-" * 65 + "\n")
                f.write(f"{'Average':<10}{np.mean(rmse_values):<15.4f}{np.mean(max_error_values):<20.4f}{np.mean(avg_error_values):<20.4f}\n")
            
            # Optional: Save the raw data as CSV for all joints
            # Create a header for the CSV file
            header = 'time'
            for i in range(num_joints):
                header += f',cmd_vel_joint{i},act_vel_joint{i}'
            
            # Create the data array for CSV
            csv_data = np.zeros((len(times), 1 + 2*num_joints))
            csv_data[:, 0] = times
            
            for t in range(len(times)):
                for j in range(num_joints):
                    csv_data[t, 1 + 2*j] = self.position_tracking_data["commanded_velocities"][t][j]
                    csv_data[t, 2 + 2*j] = self.position_tracking_data["actual_velocities"][t][j]
            
            csv_filename = os.path.join(output_dir, f'all_joints_velocity_tracking_data_env{self.tracking_env_id}_{timestamp}.csv')
            np.savetxt(csv_filename, csv_data, delimiter=',', header=header)
            
            print(f"Saved multi-joint velocity tracking data and plot to {filename}")
            print(f"Saved metrics summary to {summary_filename}")
            print(f"Saved raw data to {csv_filename}")
            
        else:
            # Single joint tracking (original code)
            actual_count = len(self.position_tracking_data["actual_velocities"])
            cmd_count = len(self.position_tracking_data["commanded_velocities"])
            
            print(f"Plotting data: timesteps={timesteps_count}, commanded={cmd_count}, actual={actual_count}")
            
            # Check if data lengths match
            if timesteps_count != cmd_count or timesteps_count != actual_count:
                print("WARNING: Data array lengths don't match!")
                print(f"  timesteps: {timesteps_count}, commanded: {cmd_count}, actual: {actual_count}")
                # Adjust the arrays to the same length if needed
                min_len = min(timesteps_count, cmd_count, actual_count)
                self.position_tracking_data["timesteps"] = self.position_tracking_data["timesteps"][:min_len]
                self.position_tracking_data["commanded_velocities"] = self.position_tracking_data["commanded_velocities"][:min_len]
                self.position_tracking_data["actual_velocities"] = self.position_tracking_data["actual_velocities"][:min_len]
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            times = np.array(self.position_tracking_data["timesteps"])
            commanded = np.array(self.position_tracking_data["commanded_velocities"])
            actual = np.array(self.position_tracking_data["actual_velocities"])
            
            # Only plot up to the length of the shortest array
            min_len = min(len(times), len(commanded), len(actual))
            print(f"Plotting {min_len} data points")
            
            plt.plot(times[:min_len], commanded[:min_len], 'b-', label=f'Commanded Velocity (Joint {self.tracking_joint_id})')
            plt.plot(times[:min_len], actual[:min_len], 'r-', label=f'Actual Velocity (Joint {self.tracking_joint_id})')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocity (rad/s)')
            plt.title(f'Joint Velocity Tracking - Env {self.tracking_env_id}, Joint {self.tracking_joint_id}')
            plt.legend()
            plt.grid(True)
            
            # Calculate velocity tracking error metrics
            if min_len > 0:
                error = commanded[:min_len] - actual[:min_len]
                rmse = np.sqrt(np.mean(np.square(error)))
                max_error = np.max(np.abs(error))
                avg_error = np.mean(np.abs(error))
                
                plt.figtext(0.02, 0.02, 
                          f'RMSE: {rmse:.4f} rad/s\nMax Error: {max_error:.4f} rad/s\nAvg Error: {avg_error:.4f} rad/s',
                          bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            # Save the plot
            filename = os.path.join(output_dir, f'joint_velocity_tracking_env{self.tracking_env_id}_joint{self.tracking_joint_id}_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            
            # Optional: Save the raw data as CSV
            data = np.column_stack((times[:min_len], commanded[:min_len], actual[:min_len]))
            csv_filename = os.path.join(output_dir, f'joint_velocity_tracking_data_env{self.tracking_env_id}_joint{self.tracking_joint_id}_{timestamp}.csv')
            np.savetxt(
                csv_filename,
                data,
                delimiter=',',
                header='time,commanded_velocity,actual_velocity'
            )
            
            print(f"Saved velocity tracking data and plot to {filename} at t={self.sim.current_time:.2f}s")
        
        return filename

    # Add a new method to save observation plots
    def save_observation_plots(self):
        print("#############################################")
        print("Saving observation visualization plots")
        print("#############################################")
        
        # Check if we have data to plot
        if not self.observation_viz_data["timesteps"]:
            print("No observation data to plot!")
            return
        
        # First call the specialized root position debugging plot function
        # This will generate detailed plots for debugging root position issues
        self.save_root_position_debug_plot()
            
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os
        import numpy as np
        
        # Continue with regular observation plots...
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "observation_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert lists to numpy arrays for easier plotting
        times = np.array(self.observation_viz_data["timesteps"])
        
        # Count of data points collected
        num_points = len(times)
        print(f"Plotting {num_points} observation data points")
        
        # --- Plot 1: Joint Positions ---
        if "joint_pos" in self.components_to_plot:
            joint_positions = np.array(self.observation_viz_data["joint_pos"])
            num_joints = joint_positions.shape[1]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            for j in range(num_joints):
                ax.plot(times, joint_positions[:, j], label=f'Joint {j+1}')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Joint Position [-1, 1]')
            ax.set_title(f'Normalized Joint Positions Over Time - Env {self.visualization_env_id}')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # Save the plot
            filename = os.path.join(output_dir, f'joint_positions_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved joint position plot to {filename}")
            
        # --- Plot 2: Joint Velocities ---
        if "joint_vel" in self.components_to_plot:
            joint_velocities = np.array(self.observation_viz_data["joint_vel"])
            num_joints = joint_velocities.shape[1]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            for j in range(num_joints):
                ax.plot(times, joint_velocities[:, j], label=f'Joint {j+1}')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Scaled Joint Velocity')
            ax.set_title(f'Scaled Joint Velocities Over Time - Env {self.visualization_env_id}')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # Save the plot
            filename = os.path.join(output_dir, f'joint_velocities_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved joint velocity plot to {filename}")
            
        # --- Plot 3: Root Position (World vs Local Frame) ---
        if "root_pos" in self.components_to_plot and "root_pos_w" in self.observation_viz_data:
            root_positions = np.array(self.observation_viz_data["root_pos"])
            root_positions_w = np.array(self.observation_viz_data["root_pos_w"])
            env_origins = np.array(self.observation_viz_data["env_origin"])
            
            # Create a 2x3 subplot grid (2 rows, 3 columns)
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Row 1: World frame position and environment origin
            axes[0, 0].plot(times, root_positions_w[:, 0], 'b-', label='X Position (World)')
            axes[0, 0].plot(times, env_origins[:, 0], 'b--', label='X Origin')
            axes[0, 0].set_ylabel('X Position (m)')
            axes[0, 0].set_title('X Position - World Frame')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(times, root_positions_w[:, 1], 'g-', label='Y Position (World)')
            axes[0, 1].plot(times, env_origins[:, 1], 'g--', label='Y Origin')
            axes[0, 1].set_ylabel('Y Position (m)')
            axes[0, 1].set_title('Y Position - World Frame')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[0, 2].plot(times, root_positions_w[:, 2], 'r-', label='Z Position (World)')
            axes[0, 2].plot(times, env_origins[:, 2], 'r--', label='Z Origin')
            axes[0, 2].set_ylabel('Z Position (m)')
            axes[0, 2].set_title('Z Position - World Frame')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # Row 2: Local frame position
            axes[1, 0].plot(times, root_positions[:, 0], 'b-', label='X Position (Local)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('X Position (m)')
            axes[1, 0].set_title('X Position - Local Frame')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(times, root_positions[:, 1], 'g-', label='Y Position (Local)')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Y Position (m)')
            axes[1, 1].set_title('Y Position - Local Frame')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            axes[1, 2].plot(times, root_positions[:, 2], 'r-', label='Z Position (Local)')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Z Position (m)')
            axes[1, 2].set_title('Z Position - Local Frame')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(output_dir, f'root_position_comparison_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved root position comparison plot to {filename}")
            
            # Also save the original single plot for local frame
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(times, root_positions[:, 0], label='X Position')
            ax.plot(times, root_positions[:, 1], label='Y Position')
            ax.plot(times, root_positions[:, 2], label='Z Position')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (m)')
            ax.set_title(f'Root Position (Local Frame) Over Time - Env {self.visualization_env_id}')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # Save the plot
            filename = os.path.join(output_dir, f'root_position_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            
        # --- Plot 4: Root Linear Velocity (World vs Local Frame) ---
        if "root_lin_vel" in self.components_to_plot and "root_lin_vel_w" in self.observation_viz_data:
            root_lin_vels = np.array(self.observation_viz_data["root_lin_vel"])
            root_lin_vels_w = np.array(self.observation_viz_data["root_lin_vel_w"])
            
            # Create a 2x3 subplot grid
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Row 1: World frame velocities
            axes[0, 0].plot(times, root_lin_vels_w[:, 0], 'b-')
            axes[0, 0].set_ylabel('X Velocity (m/s)')
            axes[0, 0].set_title('X Velocity - World Frame')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(times, root_lin_vels_w[:, 1], 'g-')
            axes[0, 1].set_ylabel('Y Velocity (m/s)')
            axes[0, 1].set_title('Y Velocity - World Frame')
            axes[0, 1].grid(True)
            
            axes[0, 2].plot(times, root_lin_vels_w[:, 2], 'r-')
            axes[0, 2].set_ylabel('Z Velocity (m/s)')
            axes[0, 2].set_title('Z Velocity - World Frame')
            axes[0, 2].grid(True)
            
            # Row 2: Local frame velocities
            axes[1, 0].plot(times, root_lin_vels[:, 0], 'b-')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('X Velocity (m/s)')
            axes[1, 0].set_title('X Velocity - Local Frame')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(times, root_lin_vels[:, 1], 'g-')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Y Velocity (m/s)')
            axes[1, 1].set_title('Y Velocity - Local Frame')
            axes[1, 1].grid(True)
            
            axes[1, 2].plot(times, root_lin_vels[:, 2], 'r-')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Z Velocity (m/s)')
            axes[1, 2].set_title('Z Velocity - Local Frame')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(output_dir, f'root_velocity_comparison_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved root velocity comparison plot to {filename}")
            
            # Also save the original single plot for local frame
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(times, root_lin_vels[:, 0], label='X Velocity')
            ax.plot(times, root_lin_vels[:, 1], label='Y Velocity')
            ax.plot(times, root_lin_vels[:, 2], label='Z Velocity')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title(f'Root Linear Velocity (Local Frame) Over Time - Env {self.visualization_env_id}')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # Save the plot
            filename = os.path.join(output_dir, f'root_linear_velocity_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            
        # --- Plot 5: Root Orientation (Quaternion) ---
        if "root_quat" in self.components_to_plot:
            root_quats = np.array(self.observation_viz_data["root_quat"])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(times, root_quats[:, 0], label='Quat W')
            ax.plot(times, root_quats[:, 1], label='Quat X')
            ax.plot(times, root_quats[:, 2], label='Quat Y')
            ax.plot(times, root_quats[:, 3], label='Quat Z')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Quaternion Values')
            ax.set_title(f'Root Orientation Quaternion Over Time - Env {self.visualization_env_id}')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # Save the plot
            filename = os.path.join(output_dir, f'root_orientation_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved root orientation plot to {filename}")
            
        # --- Plot 6: Policy Observation (flattened) ---
        # This is more complex, but useful to verify the shape and pattern
        # of observations actually passed to the policy
        if "flattened_policy_obs" in self.components_to_plot:
            flattened_obs = np.array(self.observation_viz_data["flattened_policy_obs"])
            
            # To visualize this complex data, let's create a heatmap of the observations over time
            plt.figure(figsize=(15, 10))
            
            # Create a 2D heatmap where:
            # - X-axis: observation dimension
            # - Y-axis: time
            # - Color: observation value
            plt.imshow(flattened_obs, aspect='auto', interpolation='none', cmap='viridis')
            
            plt.colorbar(label='Observation Value')
            plt.xlabel('Observation Dimension')
            plt.ylabel('Time Step')
            plt.title(f'Policy Observation Heatmap Over Time - Env {self.visualization_env_id}')
            
            # Save the plot
            filename = os.path.join(output_dir, f'policy_observations_heatmap_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved policy observation heatmap to {filename}")
            
            # Also save a line plot for the first few dimensions
            max_dims_to_plot = min(20, flattened_obs.shape[1])  # Plot at most 20 dimensions
            
            plt.figure(figsize=(15, 10))
            for i in range(max_dims_to_plot):
                plt.plot(times, flattened_obs[:, i], label=f'Dim {i}')
                
            plt.xlabel('Time (s)')
            plt.ylabel('Observation Value')
            plt.title(f'First {max_dims_to_plot} Policy Observation Dimensions - Env {self.visualization_env_id}')
            plt.legend(loc='upper right', ncol=4)
            plt.grid(True)
            
            # Save the plot
            filename = os.path.join(output_dir, f'policy_observations_line_{timestamp}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved policy observation line plot to {filename}")
        
        # Save raw data for further analysis
        raw_data_filename = os.path.join(output_dir, f'observation_raw_data_{timestamp}.npz')
        np.savez(
            raw_data_filename,
            times=times,
            joint_positions=np.array(self.observation_viz_data["joint_pos"]),
            joint_velocities=np.array(self.observation_viz_data["joint_vel"]),
            root_positions=np.array(self.observation_viz_data["root_pos"]), 
            root_quaternions=np.array(self.observation_viz_data["root_quat"]),
            root_linear_velocities=np.array(self.observation_viz_data["root_lin_vel"]),
            policy_observations=np.array(self.observation_viz_data["flattened_policy_obs"]),
            # Add world frame data if available
            root_positions_world=np.array(self.observation_viz_data.get("root_pos_w", [[0, 0, 0]])),
            root_linear_velocities_world=np.array(self.observation_viz_data.get("root_lin_vel_w", [[0, 0, 0]])),
            env_origins=np.array(self.observation_viz_data.get("env_origin", [[0, 0, 0]]))
        )
        print(f"Saved raw observation data to {raw_data_filename}")
        
        return output_dir

    # Add a specialized method just for plotting root positions to debug the current issue
    def save_root_position_debug_plot(self):
        """Create detailed plots specifically for debugging root position issues."""
        print("#############################################")
        print("Saving detailed root position debug plots")
        print("#############################################")
        
        if "root_pos_w" not in self.observation_viz_data or len(self.observation_viz_data["root_pos_w"]) == 0:
            print("No root position data to plot!")
            return
            
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os
        import numpy as np
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "root_position_debug")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert lists to numpy arrays for easier plotting
        times = np.array(self.observation_viz_data["timesteps"])
        root_pos_w = np.array(self.observation_viz_data["root_pos_w"])
        env_origins = np.array(self.observation_viz_data["env_origin"])
        
        # Calculate position errors from expected environment origin
        position_errors = root_pos_w - env_origins
        error_magnitudes = np.linalg.norm(position_errors, axis=1)
        
        # 1. Plot Root Position Deviations in all three axes
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot X, Y, Z position deviations
        axes[0].plot(times, position_errors[:, 0], 'r-', label='X Position Error')
        axes[0].set_ylabel('X Error (m)')
        axes[0].set_title(f'Root Position Error from Origin - Env {self.visualization_env_id}')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(times, position_errors[:, 1], 'g-', label='Y Position Error')
        axes[1].set_ylabel('Y Error (m)')
        axes[1].grid(True)
        axes[1].legend()
        
        axes[2].plot(times, position_errors[:, 2], 'b-', label='Z Position Error')
        axes[2].set_ylabel('Z Error (m)')
        axes[2].grid(True)
        axes[2].legend()
        
        # Plot total error magnitude
        axes[3].plot(times, error_magnitudes, 'k-', label='Total Error Magnitude')
        axes[3].set_ylabel('Error Magnitude (m)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True)
        axes[3].legend()
        
        plt.tight_layout()
        error_plot_path = os.path.join(output_dir, f'position_errors_{timestamp}.png')
        plt.savefig(error_plot_path)
        plt.close()
        
        # 2. Plot actual positions vs. environment origins
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        axes[0].plot(times, root_pos_w[:, 0], 'r-', label='Actual X Position')
        axes[0].plot(times, env_origins[:, 0], 'r--', label='Origin X')
        axes[0].set_ylabel('X Position (m)')
        axes[0].set_title(f'Root Position vs Origin - Env {self.visualization_env_id}')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(times, root_pos_w[:, 1], 'g-', label='Actual Y Position')
        axes[1].plot(times, env_origins[:, 1], 'g--', label='Origin Y')
        axes[1].set_ylabel('Y Position (m)')
        axes[1].grid(True)
        axes[1].legend()
        
        axes[2].plot(times, root_pos_w[:, 2], 'b-', label='Actual Z Position')
        axes[2].plot(times, env_origins[:, 2], 'b--', label='Origin Z')
        axes[2].set_ylabel('Z Position (m)')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        positions_plot_path = os.path.join(output_dir, f'positions_vs_origins_{timestamp}.png')
        plt.savefig(positions_plot_path)
        plt.close()
        
        # 3. Plot X-Y trajectory to see if robot is drifting
        plt.figure(figsize=(10, 10))
        plt.plot(root_pos_w[:, 0], root_pos_w[:, 1], 'b-', label='XY Trajectory')
        plt.plot(env_origins[0, 0], env_origins[0, 1], 'ro', label='Environment Origin')
        plt.plot(root_pos_w[0, 0], root_pos_w[0, 1], 'go', label='Start Position')
        plt.plot(root_pos_w[-1, 0], root_pos_w[-1, 1], 'mo', label='End Position')
        
        # Add arrows to show direction of movement
        # Add arrows every 20 points 
        arrow_indices = np.arange(0, len(times), 20)
        if len(arrow_indices) > 1:
            for i in arrow_indices[:-1]:
                plt.arrow(root_pos_w[i, 0], root_pos_w[i, 1],
                         root_pos_w[i+1, 0] - root_pos_w[i, 0],
                         root_pos_w[i+1, 1] - root_pos_w[i, 1],
                         head_width=0.1, head_length=0.2, fc='k', ec='k')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'XY Trajectory - Env {self.visualization_env_id}')
        plt.grid(True)
        plt.axis('equal')  # Equal aspect ratio
        plt.legend()
        
        trajectory_plot_path = os.path.join(output_dir, f'xy_trajectory_{timestamp}.png')
        plt.savefig(trajectory_plot_path)
        plt.close()
        
        # 4. Save additional statistics
        stats_file_path = os.path.join(output_dir, f'position_stats_{timestamp}.txt')
        with open(stats_file_path, 'w') as f:
            f.write(f"Root Position Statistics - Env {self.visualization_env_id}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("Position Error Statistics (meters):\n")
            f.write(f"{'Axis':<5}{'Mean':<10}{'Max':<10}{'Min':<10}{'Std':<10}\n")
            f.write(f"{'X':<5}{np.mean(position_errors[:, 0]):<10.6f}{np.max(position_errors[:, 0]):<10.6f}{np.min(position_errors[:, 0]):<10.6f}{np.std(position_errors[:, 0]):<10.6f}\n")
            f.write(f"{'Y':<5}{np.mean(position_errors[:, 1]):<10.6f}{np.max(position_errors[:, 1]):<10.6f}{np.min(position_errors[:, 1]):<10.6f}{np.std(position_errors[:, 1]):<10.6f}\n")
            f.write(f"{'Z':<5}{np.mean(position_errors[:, 2]):<10.6f}{np.max(position_errors[:, 2]):<10.6f}{np.min(position_errors[:, 2]):<10.6f}{np.std(position_errors[:, 2]):<10.6f}\n")
            f.write(f"{'Mag':<5}{np.mean(error_magnitudes):<10.6f}{np.max(error_magnitudes):<10.6f}{np.min(error_magnitudes):<10.6f}{np.std(error_magnitudes):<10.6f}\n\n")
            
            f.write("Environment Origin:\n")
            f.write(f"X: {env_origins[0, 0]}, Y: {env_origins[0, 1]}, Z: {env_origins[0, 2]}\n\n")
            
            f.write("Initial Root Position:\n")
            f.write(f"X: {root_pos_w[0, 0]}, Y: {root_pos_w[0, 1]}, Z: {root_pos_w[0, 2]}\n\n")
            
            f.write("Final Root Position:\n")
            f.write(f"X: {root_pos_w[-1, 0]}, Y: {root_pos_w[-1, 1]}, Z: {root_pos_w[-1, 2]}\n")
        
        print(f"Saved root position debug plots to {output_dir}")
        return output_dir
