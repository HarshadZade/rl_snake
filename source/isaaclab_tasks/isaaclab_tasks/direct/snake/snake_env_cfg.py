# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class SnakeEnvCfg(DirectRLEnvCfg):
    """Configuration for the snake robot environment."""

    # Base configuration for snake robot
    fixed_base = False

    # Set to True to override RL actions with oscillation controller for floating snake robot
    enable_oscillation_controller: bool = False

    # Disable the controller if using fixed base
    if fixed_base:
        enable_oscillation_controller = False

    enable_virtual_chassis: bool = True

    # Length of each episode in seconds
    episode_length_s = 50.0

    # Action scale determines how much the target velocity changes per RL step
    action_scale = 3.0  # rad/s
    action_space = 9  # 9 joints
    observation_space = 21  # Updated: 9 (joints) + 9 (vels) + 3 (target relative position)
    state_space = 0

    # Distance between joints in meters (extracted from USD file)
    # Each link center is 0.2m apart based on USD positioning
    link_length = 0.2

    # Simulation configuration
    # Number of physics steps per rendering step
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # -- Target Position Configuration --
    @configclass
    class TargetPositionCfg:
        """Configuration for target position task."""

        # Target position relative to the root
        target_pos: tuple = (0.5, 0.5, 0.05)  # in meters (-1.8, 0, 0) in local frame
        # Which link to track for reaching the target (0 is root, higher numbers for other links)
        tracked_link_idx: int = 9  # Default to the 9th link (adjust based on model)
        # Scale for distance threshold (when to consider target reached)
        success_distance_threshold: float = 0.1  # in meters
        # Visual marker configuration
        marker_radius: float = 0.1  # Radius of the target sphere in meters
        marker_color: tuple = (1.0, 0.0, 0.0)  # RGB color (red)
        # Whether to show the target marker
        show_marker: bool = True  # Set to False to hide the target marker

    target_position: TargetPositionCfg = TargetPositionCfg()
    # -- End Target Position Configuration --

    # -- LQR Style Reward Parameters --
    @configclass
    class LQRRewardCfg:
        """Configuration for LQR-style reward function for fixed-base snake robot."""

        # State cost matrix diagonal elements (Q matrix)
        joint_pos_cost: float = 0.01  # Cost on joint position deviation
        joint_vel_cost: float = 0.01  # Cost on joint velocity
        target_cost: float = 5.0  # Cost on end-effector position deviation from target

        # Control cost matrix diagonal elements (R matrix)
        control_cost: float = 0.01  # Cost on control inputs (joint velocities)

        # Additional reward terms
        alive_bonus: float = 0.1  # Small bonus for staying alive
        success_bonus: float = 100.0  # Bonus for reaching target

    lqr_reward: LQRRewardCfg = LQRRewardCfg()

    if fixed_base:
        usd_path = "./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_realistic_fixed_dim_v0.usda"
    else:
        usd_path = "./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_realistic_floating_dim_v0.usda"

    # -- Robot Configuration (Loading from USD)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",  # Standard prim path pattern
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,  # Set to True if you need contact sensors #TODO: check this
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,  # Tune if needed
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Define initial joint positions
            joint_pos={
                "joint_1": 0.0,
            },
            pos=(0.0, 0.0, 0.075),  # Initial base position (adjust height based on robot) 0.15, 0.075, 0.075 m
            rot=(0.0, 0.0, 0.0, 1.0),  # Initial base orientation
        ),
        actuators={
            # Define actuators for your joints #TODO: tune all these parameters
            "snake_joints": ImplicitActuatorCfg(
                # Use regex matching your joint names, or list them
                joint_names_expr=["joint_[1-9]"],  # Example regex
                effort_limit_sim=50.0,  # (Nm) <<< Tune
                velocity_limit_sim=10.0,  # (286deg/s)(rad/s) <<< Tune
                stiffness=0.0,  # Kp
                damping=5.0,  # Kd
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
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # # reset
    # joint_angle_range = [-1.57, 1.57] # rad

    # -- Testing Configuration --
    @configclass
    class TestingCfg:
        """Configuration for testing modes."""

        # Type of manual oscillation ('sidewinding' or 'constant')
        oscillation_type: str = "sidewinding"  # 'sidewinding' or 'constant'
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
        env_id: int = 0  # Which environment to track
        track_all_joints: bool = True  # Whether to track all joints or just one
        joint_id: int = 0  # Which joint to track (if not tracking all)
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
        components_to_plot: list = [
            "joint_pos",
            "joint_vel",
            "root_pos",
            "root_lin_vel",
            "root_quat",
            "flattened_policy_obs",
        ]

    observation_visualization: ObservationVisualizationCfg = ObservationVisualizationCfg()
    # --- END OBSERVATION VISUALIZATION CONFIG ---

    # --- PROGRESS TRACKING CONFIG ---
    @configclass
    class ProgressTrackingCfg:
        """Configuration for progress tracking termination condition."""

        enable: bool = True  # Whether to enable progress tracking termination
        progress_check_window: int = 50  # Number of steps to wait before checking progress from episode start
        min_progress_threshold: float = (
            0.05  # Minimum distance robot should move toward goal since episode start (meters)
        )

    progress_tracking: ProgressTrackingCfg = ProgressTrackingCfg()
    # --- END PROGRESS TRACKING CONFIG ---
