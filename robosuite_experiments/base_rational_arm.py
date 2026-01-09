"""
Base rational arm agent that uses reward weights to guide behavior.
"""

import numpy as np
from arm_world import ArmWorld
import arm_feature_utils as feature_utils


class BaseRationalArm:
    """
    Base robotic arm agent that computes features and rewards.
    
    Uses linear combination of features weighted by learned weights.
    Leverages robosuite's built-in OSC controller for control.
    """
    
    def __init__(
        self,
        world: ArmWorld,
        weights=None,
        seed=None,
    ):
        """
        Initialize the rational arm agent.
        
        Args:
            world: ArmWorld environment instance
            weights: Feature weights for reward computation
            seed: Random seed for reproducibility
        """
        self.world = world
        
        # TODO: This is old
        if weights is None:
            weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
        
        self.weights = np.array(weights, dtype=np.float32)
        
        if seed is not None:
            np.random.seed(seed)
    
    def features(self, obs):
        """
        Compute features from current observation.
        
        Uses centralized feature computation from arm_feature_utils.
        
        Args:
            obs: Observation dictionary from environment
            
        Returns:
            numpy array of feature values [5 features]
        """
        return feature_utils.compute_features(obs, include_red_dist=True, include_zones=False)
    
    def reward_fn(self, obs):
        """
        Compute reward as weighted sum of features.
        
        Args:
            obs: Observation dictionary from environment
            
        Returns:
            Scalar reward value
        """
        feats = self.features(obs)
        return np.dot(self.weights, feats)
    
    def get_action(self, obs):
        """
        Get action for the current observation.
        
        This is a placeholder - in practice, we'll use a simple policy
        or scripted behavior. For the base class, we return a zero action.
        
        Args:
            obs: Observation dictionary from environment
            
        Returns:
            Action array
        """
        # Default: no action (will be overridden in subclasses)
        return np.zeros(self.world.action_dim)
    
    def get_feature_descriptions(self):
        """
        Get natural language descriptions of features for LLM-based learners.
        
        Uses centralized descriptions from arm_feature_utils.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return feature_utils.get_feature_descriptions(include_red_dist=True, include_zones=False)

