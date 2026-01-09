"""
Robosuite-compatible PHRI learner that works with observation dictionaries.
"""

import time
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interact_drive.learner.phri_learner import PHRILearner


class RobosuitePHRILearner(PHRILearner):
    """
    PHRI Learner adapted for robosuite observation dictionaries.
    
    Overrides compute_features to handle dict observations instead of array states.
    """
    
    def compute_features(self, trajectory: dict[str, list]) -> np.ndarray:
        """
        Compute average features over trajectory.
        
        For robosuite, trajectory["state"] contains observation dictionaries,
        not numpy arrays. We pass them directly to the arm's features method.
        
        Args:
            trajectory: Dict with "state" (list of obs dicts) and "control" keys
            
        Returns:
            Summed features across trajectory (as numpy array)
        """
        total_features = np.zeros(len(self.car.weights))
        states = trajectory["state"]
        controls = trajectory["control"]
        
        # Debug print
        print(f"Computing features for {len(states)} states")
        
        for obs_dict in states:
            # Pass observation dict directly (not wrapped in list or tensor)
            features = self.car.features(obs_dict)
            if features is not None:
                total_features += features
        
        return total_features
    
    def update_weights(self, planned_trajectory, human_trajectory):
        """
        Update reward weights based on human correction.
        
        Overrides parent to handle numpy arrays instead of TensorFlow tensors.
        
        Args:
            planned_trajectory: Original trajectory from planner
            human_trajectory: List of {state, control} from human input
        """
        # Extract relevant features (returns numpy arrays)
        robot_features = self.compute_features(planned_trajectory)
        human_features = self.compute_features(human_trajectory)
        
        print("Planned trajectory features:", robot_features)
        print("Human trajectory features:", human_features)
        
        # Log data
        self.trajectory_lengths.append(
            {"robot": len(planned_trajectory["state"]), "human": len(human_trajectory["state"])}
        )
        
        # Compute update
        feature_diff = human_features - robot_features
        
        self.feature_differences.append(
            {
                "delta_phi": feature_diff,  # Already numpy
                "robot_features": robot_features,  # Already numpy
                "human_features": human_features,  # Already numpy
            }
        )
        old_weights = self.car.weights.copy()
        
        feature_specific_lr = np.ones(len(self.car.weights)) * self.learning_rate
        new_weights = self.car.weights + feature_specific_lr * feature_diff
        
        # Log update (pass numpy arrays directly)
        self.log_update(
            planned_trajectory,
            human_trajectory,
            robot_features,
            human_features,
            old_weights,
            new_weights,
        )
        self.car.weights = new_weights
        
        print("\nFeature Analysis:")
        print(f"Robot trajectory length: {len(planned_trajectory['state'])}")
        print(f"Human trajectory length: {len(human_trajectory['state'])}")
        print(f"Robot features: {robot_features}")
        print(f"Human features: {human_features}")
        print(f"Delta phi: {feature_diff}")
        print("Updated weights:", new_weights)

