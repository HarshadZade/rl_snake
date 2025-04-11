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
    episode_length_s = 10.0
    action_scale = 0.01  # rad #TODO: tune this
    action_space = 9    # 9 joints
    observation_space = 28 #TODO: fix this
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

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
            pos=(0.0, 0.0, 0.1),  # Initial base position (adjust height based on robot)
            rot=(0.0, 0.0, 0.0, 1.0), # Initial base orientation
        ),
        actuators={
            # Define actuators for your joints #TODO: tune all these parameters
            "snake_joints": ImplicitActuatorCfg(
                # Use regex matching your joint names, or list them
                joint_names_expr=["joint_[1-9]"], # Example regex
                effort_limit=100.0,   # <<< Tune based on your robot's specs
                velocity_limit=10.0,  # <<< Tune based on your robot's specs
                stiffness=1000.0,       # <<< Tune: Use >0 for position/velocity control
                damping=10.0,        # <<< Tune: Use >0 for position/velocity control (helps stability)
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
            static_friction=1.0,
            dynamic_friction=1.0,
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
    rew_scale_forward_velocity = 1.0
    rew_scale_action_penalty = -0.005 #-0.005
    rew_scale_joint_vel_penalty = -0.001 #-0.001
    rew_scale_termination = -2.0 #-2.0
    rew_scale_alive = 0.1
    rew_scale_action_smoothness_penalty = -0.05 #-0.05
    rew_scale_lateral_velocity_penalty = -0.05 #-0.05
    rew_scale_joint_limit_penalty = -0.1 #-0.1

class SnakeEnv(DirectRLEnv):
    cfg: SnakeEnvCfg

    def __init__(self, cfg: SnakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        
        #TODO: Need to make sure all the required information is used and set here!!
        # Currently its just some random stuff!
        # Get joint limits
        self.joint_pos_limits = self.snake_robot.data.soft_joint_pos_limits[0]
        self.joint_pos_ranges = self.joint_pos_limits[:, 1] - self.joint_pos_limits[:, 0]
        self.joint_pos_mid = (self.joint_pos_limits[:, 0] + self.joint_pos_limits[:, 1]) / 2

        # Initialize action history
        # Start with default positions, cloned to avoid modifying the default tensor
        self.dof_targets = self.snake_robot.data.default_joint_pos.clone().to(device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Ensure it has the correct shape for all envs
        if self.num_envs > 1:
             self.dof_targets = self.dof_targets.repeat(self.num_envs, 1)

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
        # Store action for smoothness calculations in reward
        self.prev_actions = self.dof_targets.clone()

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

        # Clamp the targets to the joint limits
        self.dof_targets[:] = torch.clamp(new_targets, self.joint_pos_limits[:, 0], self.joint_pos_limits[:,1])


    def _apply_action(self) -> None:
        self.snake_robot.set_joint_position_target(self.dof_targets)

    def _get_observations(self) -> dict:
        # Get joint positions and velocities
        joint_pos = self.snake_robot.data.joint_pos
        joint_vel = self.snake_robot.data.joint_vel
        
        # # Calculate joint positions normalized to [-1, 1]
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_pos_limits[:, 0]) / self.joint_pos_ranges - 1.0
        
        # Get root state information
        root_pos = self.snake_robot.data.root_pos_w
        root_quat = self.snake_robot.data.root_quat_w
        root_lin_vel = self.snake_robot.data.root_lin_vel_w
        
        # Combine observations
        obs = torch.cat(
            (
                joint_pos_normalized,   # Normalized joint positions
                joint_vel * 0.1,        # Scaled joint velocities
                root_pos,               # Root position
                root_quat,              # Root orientation
                root_lin_vel,           # Root linear velocity
            ),
            dim=-1,
        )
        
        observations = {"policy": obs}
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
        out_of_bounds = torch.any(self.joint_pos < self.joint_pos_limits[:, 0]) | \
                        torch.any(self.joint_pos > self.joint_pos_limits[:, 1])
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.snake_robot._ALL_INDICES
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
            self.joint_pos_limits[:, 0],
            self.joint_pos_limits[:, 1]
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