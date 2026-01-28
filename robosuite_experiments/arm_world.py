"""
Wrapper for robosuite environment to interface with QuickLAP learning framework.
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from two_block_env import TwoBlockPickPlace


class ArmWorld:
    """
    Wrapper around robosuite environment for pick-and-place task.
    
    Manages the Panda robot, red block (target), and green block (obstacle).
    Provides a simplified interface for the learning framework.
    """
    
    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        seed=None,
    ):
        """
        Initialize the robosuite environment.
        
        Args:
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable off-screen rendering
            use_camera_obs: Whether to include camera observations
            control_freq: Control frequency in Hz
            horizon: Episode length in timesteps
            seed: Random seed for reproducibility
        """
        self.has_renderer = has_renderer
        self.control_freq = control_freq
        self.horizon = horizon
        self.timestep = 0
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Load OSC controller config
        controller_config = load_composite_controller_config(
            controller="BASIC",  # OSC for arm, joint position for gripper
        )
        
        # Create custom environment with red and green blocks
        self.env = TwoBlockPickPlace(
            robots="Panda",
            gripper_types="default",
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            control_freq=control_freq,
            horizon=horizon,
            use_object_obs=True,  # We need object positions
            reward_shaping=False,  # We'll use our own reward function
        )
        
        # Reset to get initial observation
        self.obs = self.env.reset()
        
        # Panda joint limits (approximate)
        self.joint_limits = (
            np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        )
        
        # Define zone positions for task
        self.zone_a_pos = np.array([-0.3, -0.3, 0.97])  # Left side of table (red block spawns here)
        self.zone_b_pos = np.array([0.1, 0.3, 0.97])    # Right side of table (target zone)
        self.zone_c_pos = np.array([-0.25, 0.15, 0.97])  # Zone C (obstacle zone)
        
    def reset(self):
        """Reset the environment and return initial observation."""
        self.obs = self.env.reset()
        self.timestep = 0
        return self.get_observation()
    
    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Args:
            action: Control action for the robot
            
        Returns:
            observation: Current observation dict
            reward: Reward (we'll compute this separately)
            done: Whether episode is finished
            info: Additional information
        """
        self.obs, reward, done, info = self.env.step(action)
        self.timestep += 1
        
        # Override done based on our horizon
        done = self.timestep >= self.horizon
        
        return self.get_observation(), reward, done, info
    
    def get_observation(self):
        """
        Extract relevant observations from robosuite environment.
        
        Returns:
            Dictionary with structured observations for feature computation
        """
        obs_dict = {}
        
        # End-effector position and velocity
        obs_dict["ee_pos"] = self.obs["robot0_eef_pos"]
        obs_dict["ee_vel"] = self.obs.get("robot0_eef_vel", np.zeros(3))
        
        # Joint positions
        obs_dict["joint_pos"] = self.obs["robot0_joint_pos"]
        
        # Object positions from custom environment
        # Red block position (target object)
        obs_dict["red_block_pos"] = self.obs.get("red_block_pos", self.zone_a_pos)
        
        # Green block position (obstacle)
        obs_dict["green_block_pos"] = self.obs.get("green_block_pos", np.array([0.0, 0.0, 0.82]))
        
        # Zone positions
        obs_dict["zone_a_pos"] = self.zone_a_pos
        obs_dict["zone_b_pos"] = self.zone_b_pos
        obs_dict["zone_c_pos"] = self.zone_c_pos
        
        # Joint limits
        obs_dict["joint_limits"] = self.joint_limits
        
        # Raw observation for potential future use
        obs_dict["raw_obs"] = self.obs
        
        return obs_dict
    
    def render(self):
        """Render the environment if renderer is enabled."""
        if self.has_renderer:
            self.env.render()
    
    def close(self):
        """Clean up resources."""
        self.env.close()
    
    @property
    def action_dim(self):
        """Get action space dimension."""
        return self.env.action_dim
    
    @property
    def action_spec(self):
        """Get action space specification."""
        return self.env.action_spec

