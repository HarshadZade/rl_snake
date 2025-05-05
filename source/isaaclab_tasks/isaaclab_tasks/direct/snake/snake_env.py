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
    # env - simplified for oscillation testing only
    decimation = 1  
    episode_length_s = 20.0  # Longer episodes for testing oscillation
    action_space = 1    # 1 joint for 2-link robot
    observation_space = 3  # Needed for compatibility with RL infrastructure
    state_space = 0     # Required for RL framework compatibility

    # simulation - smoother physics
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1)

    # scene - fewer environments for better visualization
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)

    # -- Robot Configuration (unchanged)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_2_link-fixedbase.usd",
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
            joint_pos={
                "joint_1": 0.0,
            },
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "snake_joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=10000.0,
                velocity_limit=3.0,   # Lower velocity limit for stability
                stiffness=100000.0,
                damping=500.0,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Oscillation configuration
    @configclass
    class OscillationCfg:
        """Configuration for oscillation testing."""
        # Enable manual oscillation
        enable: bool = True
        # Oscillation parameters
        amplitude_deg: float = 20.0
        frequency_hz: float = 0.2
        # Max position change per step (rad)
        max_step: float = 0.02
        # Safety margin from joint limits (0-1)
        safety_margin: float = 0.2
        # Print debug info every N steps
        print_interval: int = 20

    oscillation: OscillationCfg = OscillationCfg()
    
    # Minimal reward parameters (to keep compatibility)
    rew_scale_target_tracking = 1.0
    rew_scale_joint_vel_penalty = -0.0001

class SnakeEnv(DirectRLEnv):    
    cfg: SnakeEnvCfg

    def __init__(self, cfg: SnakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.env_step_counter = 0
        print("Initializing Snake Environment for oscillation testing")
        
        # Get joint limits
        self.joint_pos_limits = self.snake_robot.data.soft_joint_pos_limits
        self.joint_pos_lower_limits = self.joint_pos_limits[..., 0].to(self.device)
        self.joint_pos_upper_limits = self.joint_pos_limits[..., 1].to(self.device)
        
        # Apply safety margin to limits
        margin_factor = self.cfg.oscillation.safety_margin
        original_range = self.joint_pos_upper_limits - self.joint_pos_lower_limits
        margin = original_range * margin_factor
        
        self.safe_lower_limits = self.joint_pos_lower_limits + margin
        self.safe_upper_limits = self.joint_pos_upper_limits - margin
        
        # Use first environment's values for printing
        # print(f"Original joint limits: {self.joint_pos_lower_limits.item():.3f} to {self.joint_pos_upper_limits.item():.3f}")
        # print(f"Safe joint limits: {self.safe_lower_limits.item():.3f} to {self.safe_upper_limits.item():.3f}")
        
        self.joint_pos_ranges = self.joint_pos_upper_limits - self.joint_pos_lower_limits
        self.joint_pos_mid = (self.joint_pos_lower_limits + self.joint_pos_upper_limits) / 2
        
        # Initialize with mid-position
        self.dof_targets = torch.ones((self.num_envs, self.snake_robot.num_joints), device=self.device) * self.joint_pos_mid
        
        # Minimal storage for positions and velocities (for debugging)
        self.joint_pos = self.snake_robot.data.joint_pos
        self.joint_vel = self.snake_robot.data.joint_vel
        
        # For oscillation visualization
        self.target_joint_pos = self.dof_targets.clone()
        
        print(f"Environment initialized with {self.num_envs} environments")
        print(f"Number of joints: {self.snake_robot.num_joints}")
        print(f"Starting at mid position: {self.joint_pos_mid[0, 0].item():.3f}")  # Index first env, first joint

    def _setup_scene(self):
        # Create robot
        self.snake_robot = Articulation(self.cfg.robot)
        
        # Setup terrain
        self.cfg.terrain.num_envs = self.cfg.scene.num_envs
        self.cfg.terrain.env_spacing = self.cfg.scene.env_spacing
        self._terrain = TerrainImporter(self.cfg.terrain)
        
        # Setup environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["snake_robot"] = self.snake_robot
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Handle oscillation - the main function for this environment."""
        self.env_step_counter += 1
        
        # OSCILLATION - implement smooth sinusoidal motion
        try:
            # Get current time
            current_time = self.sim.current_time
            
            # Convert oscillation parameters to radians
            amplitude_rad = math.radians(self.cfg.oscillation.amplitude_deg)
            omega = 2.0 * math.pi * self.cfg.oscillation.frequency_hz
            
            # Calculate target position from sine wave
            offset = amplitude_rad * math.sin(omega * current_time)
            # Access first environment and first joint for mid position value
            target_pos_scalar = self.joint_pos_mid[0, 0].item() + float(offset)  # Ensure scalar float
            
            # Create tensor with the calculated target position
            target_tensor = torch.full_like(self.dof_targets, target_pos_scalar)
            
            # Clamp to safe limits
            target_tensor = torch.clamp(target_tensor, self.safe_lower_limits, self.safe_upper_limits)
            
            # Get current positions
            current_pos = self.snake_robot.data.joint_pos
            
            # Gradual movement (rate-limiting)
            max_step = self.cfg.oscillation.max_step
            delta = target_tensor - current_pos
            limited_delta = torch.clamp(delta, -max_step, max_step)
            
            # Apply movement
            self.dof_targets = current_pos + limited_delta
            
            # For visualization purposes
            self.target_joint_pos = target_tensor
            
            # Print status periodically
            if self.env_step_counter % self.cfg.oscillation.print_interval == 0:
                print(f"Time: {current_time:.2f}s | Target: {target_pos_scalar:.4f} rad | "
                      f"Current: {current_pos[0, 0].item():.4f} rad | "
                      f"Delta: {limited_delta[0, 0].item():.4f} rad")
                
        except Exception as e:
            # Emergency fallback
            print(f"ERROR in oscillation: {e}")
            # Reset to mid position
            self.dof_targets = torch.ones_like(self.dof_targets) * self.joint_pos_mid
    
    def _apply_action(self) -> None:
        """Apply position targets to the robot."""
        self.snake_robot.set_joint_position_target(self.dof_targets)

    def _get_observations(self) -> dict:
        """Minimal observation function - just for compatibility."""
        # Get joint positions and velocities
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        
        # Simple normalization
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_lower_limits) / self.joint_pos_ranges - 1.0
        joint_vel_normalized = torch.clamp(joint_vel / 5.0, -1.0, 1.0)  # Simple velocity normalization
        target_pos_normalized = 2.0 * (self.target_joint_pos - self.joint_pos_lower_limits) / self.joint_pos_ranges - 1.0
        
        # Combine observations
        obs = torch.cat([joint_pos_normalized, joint_vel_normalized, target_pos_normalized], dim=-1)
        
        # Safety check
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            obs = torch.zeros_like(obs)
            print("WARNING: Fixed NaN/Inf in observations")
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Minimal reward function - just for compatibility."""
        # Simple position tracking reward
        joint_pos = self.snake_robot.data.joint_pos
        tracking_error = torch.sum((joint_pos - self.target_joint_pos)**2, dim=-1)
        reward = torch.ones((self.num_envs,), device=self.device)  # Constant reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimal termination function - just for compatibility."""
        # Only terminate on timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)  # No early termination
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset to mid position."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            
        try:
            super()._reset_idx(env_ids)
            
            # Always reset to exactly mid position
            joint_pos = torch.ones((len(env_ids), self.snake_robot.num_joints), device=self.device) * self.joint_pos_mid
            joint_vel = torch.zeros_like(joint_pos)
            
            # Write to simulation
            self.snake_robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
            # Reset targets
            if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
                self.dof_targets[env_ids] = joint_pos
                self.target_joint_pos[env_ids] = joint_pos
            else:
                self.dof_targets = joint_pos
                self.target_joint_pos = joint_pos
                
            # Safely print the mid position value by indexing into the tensor first
            print(f"Reset to mid position: {self.joint_pos_mid[0, 0].item():.3f}")
            
        except Exception as e:
            print(f"ERROR in reset: {e}")
            # Emergency reset
            try:
                joint_pos = torch.zeros((len(env_ids), self.snake_robot.num_joints), device=self.device)
                joint_vel = torch.zeros_like(joint_pos)
                self.snake_robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
                
                if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
                    self.dof_targets[env_ids] = joint_pos
                    self.target_joint_pos[env_ids] = joint_pos
                else:
                    self.dof_targets = joint_pos
                    self.target_joint_pos = joint_pos
            except:
                print("CRITICAL ERROR in emergency reset")