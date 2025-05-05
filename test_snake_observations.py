#!/usr/bin/env python3

"""Test script to visualize snake observations.

This script initializes the Snake environment and runs it for a specified number of steps
to collect and visualize observation data.
"""

import os
import sys
import time
import signal
import gymnasium as gym
import numpy as np
import torch

# Ensure our paths are correct
sys.path.append(os.path.abspath("."))

# Import snake environment
from source.isaaclab_tasks.isaaclab_tasks.direct.snake.snake_env import SnakeEnv, SnakeEnvCfg

# Create a simplified test app for running the environment
def main():
    print("Initializing Snake environment test for observation visualization...")
    
    # Create a modified config with fewer environments for testing
    cfg = SnakeEnvCfg()
    cfg.scene.num_envs = 4  # Use fewer environments for testing
    
    # Enable observation visualization
    cfg.observation_visualization.enable = True
    cfg.observation_visualization.env_id = 0
    cfg.observation_visualization.max_points = 1000
    cfg.observation_visualization.components_to_plot = [
        "joint_pos", "joint_vel", "root_pos", "root_lin_vel", 
        "root_quat", "flattened_policy_obs"
    ]
    
    # Create the environment (standalone mode)
    env = SnakeEnv(cfg)
    
    # Reset the environment
    obs_dict = env.reset()
    
    # Print observation info
    print(f"Observation shape: {obs_dict['policy'].shape}")
    
    # Run for a specified number of steps
    num_steps = 200
    print(f"Running environment for {num_steps} steps to collect observation data...")
    
    # Create dummy actions (zeros)
    actions = torch.zeros((cfg.scene.num_envs, env.snake_robot.num_joints), device=env.device)
    
    # Run the simulation
    for step in range(num_steps):
        # Step the environment
        obs_dict, rew, terminated, truncated, info = env.step(actions)
        
        # Print progress occasionally
        if step % 20 == 0:
            print(f"Step {step}/{num_steps}")
    
    # Force save the observation plots
    print("Saving observation plots...")
    output_dir = env.save_observation_plots()
    print(f"Plots saved to: {output_dir}")
    
    # Close the environment
    env.close()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    # Handle Ctrl+C gracefully to ensure plots are saved
    def signal_handler(sig, frame):
        print("Ctrl+C detected, exiting gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the main function
    main() 