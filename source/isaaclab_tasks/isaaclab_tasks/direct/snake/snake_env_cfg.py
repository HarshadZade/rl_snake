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

    # Length of each episode in seconds
    episode_length_s = 50.0

    # Action scale determines how much the target velocity changes per RL step
    action_scale = 0.26  # rad/s
    action_space = 9  # 9 joints
    observation_space = 21  # Updated: 9 (joints) + 9 (vels) + 3 (target relative position)
    state_space = 0

    # TODO: Get this from the USD instead of hardcoding # Length of each link in meters,
    # used for height termination
    link_length = 4.0

    # Simulation configuration
    # Number of physics steps per rendering step
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Set to True to override RL actions with oscillation control
    enable_oscillation_controller: bool = True

    # -- Target Position Configuration --
    @configclass
    class TargetPositionCfg:
        """Configuration for target position task."""

        # Target position relative to the root
        target_pos: tuple = (-1.2, 0.0, 0.8)  # in meters (-1.8, 0, 0) in local frame
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
        end_effector_cost: float = 5.0  # Cost on end-effector position deviation from target

        # Control cost matrix diagonal elements (R matrix)
        control_cost: float = 0.01  # Cost on control inputs (joint velocities)

        # Additional reward terms
        alive_bonus: float = 0.1  # Small bonus for staying alive
        success_bonus: float = 100.0  # Bonus for reaching target

    lqr_reward: LQRRewardCfg = LQRRewardCfg()

    # -- Robot Configuration (Loading from USD)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",  # Standard prim path pattern
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                "./source/isaaclab_tasks/isaaclab_tasks/direct/snake/usd_files/snake_realistic_floating_dim_v0.usda"
            ),
            activate_contact_sensors=False,  # Set to True if you need contact sensors #TODO: check this
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,  # Tune if needed
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
            pos=(0.0, 0.0, 0.0375),  # Initial base position (adjust height based on robot) 0.15, 0.075, 0.075 m
            rot=(0.0, 0.0, 0.0, 1.0),  # Initial base orientation
        ),
        actuators={
            # Define actuators for your joints #TODO: tune all these parameters
            "snake_joints": ImplicitActuatorCfg(
                # Use regex matching your joint names, or list them
                joint_names_expr=["joint_[1-9]"],  # Example regex
                effort_limit=50.0,  # (Nm) <<< Tune
                velocity_limit=0.262,  # (15deg/s)(rad/s) <<< Tune
                stiffness=0.0,  # Kp
                damping=100.0,  # Kd
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
