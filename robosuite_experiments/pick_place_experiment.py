"""
Pick-and-place experiment for evaluating learning methods with robosuite.
"""

import json
import os
import sys
from datetime import datetime
from typing import Callable, Any

import numpy as np

# Add parent directory to path so we can import from interact_drive
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_world import ArmWorld
from intervention_arm import InterventionArm


class PickPlaceExperiment:
    """
    Experiment for pick-and-place task with human intervention.
    
    Red block is the target object to move from zone A to zone B.
    Green block is an obstacle on the path.
    Human intervenes by pulling toward green block while saying "Go faster".
    """
    
    def __init__(
        self,
        exp_name: str = "pick_place",
        save_dir: str | None = None,
        verbose: bool = False,
        horizon: int = 300,  # Increased from 150 to give more time for task
    ):
        """
        Initialize the experiment.
        
        Args:
            exp_name: Name for this experiment
            save_dir: Directory to save results (auto-generated if None)
            verbose: Whether to print detailed logs
            horizon: Episode length in timesteps
        """
        self.exp_name = exp_name
        self.verbose = verbose
        self.horizon = horizon
        
        # Set up save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            save_dir = os.path.join("logs", f"robosuite_{exp_name}_{timestamp}")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.step_data = []
    
    def setup_world(
        self,
        learner_factory: Callable,
        seed: int,
        utterance: str = "Go faster"
    ) -> tuple[ArmWorld, InterventionArm]:
        """
        Set up the robosuite environment and intervention arm.
        
        Args:
            learner_factory: Factory function to create learner
            seed: Random seed for reproducibility
            utterance: What human says during intervention
            
        Returns:
            Tuple of (world, arm)
        """
        # Create world
        world = ArmWorld(
            has_renderer=self.verbose,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=self.horizon,
            seed=seed,
        )
        
        # Create intervention arm with learner
        # Learner factory expects an arm-like object with weights and features
        # We'll pass a temporary arm to get the learner, then create the real arm
        
        # Expert weights: [red_dist, green_dist, velocity, collision, joints]
        expert_weights = np.array([1.0, 1.0, 10.0, 2.0, 2.0])
        base_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
        
        # Create arm first without learner
        arm = InterventionArm(
            world=world,
            learner=None,  # Will set after creation
            utterance=utterance,
            expert_weights=expert_weights,
            intervention_interval=(50, 70),
            base_weights=base_weights,
            seed=seed,
        )
        
        # Now create learner with the arm
        learner = learner_factory(arm)
        arm.learner = learner
        
        return world, arm
    
    def get_metrics(self, t: int, arm: InterventionArm, obs: dict) -> dict[str, Any]:
        """
        Collect metrics for current timestep.
        
        Args:
            t: Current timestep
            arm: InterventionArm instance
            obs: Current observation
            
        Returns:
            Dictionary of metrics
        """
        # Compute features and reward
        features = arm.features(obs)
        reward = arm.reward_fn(obs)
        
        # Distance metrics
        ee_pos = obs["ee_pos"]
        red_dist = np.linalg.norm(ee_pos - obs["red_block_pos"])
        green_dist = np.linalg.norm(ee_pos - obs["green_block_pos"])
        
        return {
            "timestep": t,
            "ee_pos": ee_pos.tolist(),
            "ee_vel": obs["ee_vel"].tolist(),
            "red_block_distance": float(red_dist),
            "green_block_distance": float(green_dist),
            "weights": arm.weights.tolist(),
            "features": features.tolist(),
            "reward": float(reward),
            "is_intervention": arm.is_intervention(),
        }
    
    def get_feature_trajectory(self) -> np.ndarray:
        """Extract feature trajectory from step data."""
        if not self.step_data:
            return np.array([])
        return np.array([step["features"] for step in self.step_data])
    
    def evaluate_performance(self, arm: InterventionArm) -> float:
        """
        Evaluate overall performance.
        
        Uses cumulative reward based on expert weights as metric.
        
        Args:
            arm: InterventionArm instance
            
        Returns:
            Performance metric (higher is better)
        """
        # Compute cumulative reward using expert weights
        feature_trajectory = self.get_feature_trajectory()
        
        if len(feature_trajectory) == 0:
            return 0.0
        
        # Sum of rewards using expert weights
        expert_rewards = feature_trajectory @ arm.expert_weights
        return float(np.sum(expert_rewards))
    
    def save_results(self, metric: float):
        """
        Save experiment results to files.
        
        Args:
            metric: Performance metric to save
        """
        results_file = os.path.join(self.save_dir, "experiment_results.json")
        
        results = {
            "experiment_name": self.exp_name,
            "performance_metric": metric,
            "steps": self.step_data,
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {results_file}")
    
    def run(
        self,
        learner_factory: Callable,
        seed: int = 0,
        utterance: str = "Go faster"
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """
        Run complete experiment episode.
        
        Args:
            learner_factory: Factory function to create learner
            seed: Random seed for reproducibility
            utterance: What human says during intervention
            
        Returns:
            Tuple of (performance_metric, feature_trajectory, learned_weights)
        """
        print(f"\nStarting {self.exp_name} experiment...")
        print(f"Seed: {seed}, Utterance: '{utterance}'")
        print(f"Results will be saved to: {self.save_dir}")
        
        # Set up environment and arm
        self.step_data = []
        world, arm = self.setup_world(learner_factory, seed, utterance)
        
        # Save experiment configuration
        config = {
            "experiment_name": self.exp_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "utterance": utterance,
            "horizon": self.horizon,
            "arm_config": {
                "initial_weights": arm.weights.tolist(),
                "expert_weights": arm.expert_weights.tolist(),
                "intervention_interval": [arm.intervention_start, arm.intervention_end],
            },
        }
        
        config_file = os.path.join(self.save_dir, "experiment_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Run episode
        obs = world.get_observation()
        final_learned_weights = None
        
        try:
            print("\nRunning simulation...")
            for t in range(self.horizon):
                # Get action from arm
                action = arm.get_action(obs)
                
                # Step environment
                obs, _, done, _ = world.step(action)
                
                # Collect metrics
                metrics = self.get_metrics(t, arm, obs)
                self.step_data.append(metrics)
                
                # Render if verbose
                if self.verbose:
                    world.render()
                
                # Print progress
                if t % 20 == 0 and not self.verbose:
                    print(f"Step {t}/{self.horizon}")
                
                if done:
                    break
            
            # Get final learned weights
            final_learned_weights = arm.weights.copy()
            
        except Exception as e:
            print(f"\nError during experiment: {e}")
            raise e
        
        finally:
            # Compute metrics
            feature_trajectory = self.get_feature_trajectory()
            performance_metric = self.evaluate_performance(arm)
            
            print(f"\nPerformance metric: {performance_metric:.2f}")
            print(f"Final weights: {arm.weights}")
            
            # Save results
            self.save_results(performance_metric)
            print(f"Experiment completed. Results saved to {self.save_dir}")
            
            # Clean up
            world.close()
        
        return performance_metric, feature_trajectory, final_learned_weights


def main():
    """Test the experiment with a simple learner."""
    from robosuite_phri_learner import RobosuitePHRILearner
    
    # Use verbose=False to avoid renderer issues on Mac
    # If you want visualization, run with: mjpython pick_place_experiment.py
    experiment = PickPlaceExperiment(verbose=False, horizon=100)
    
    def learner_factory(arm):
        return RobosuitePHRILearner(arm, log_file="robosuite_test.txt")
    
    metric, features, weights = experiment.run(learner_factory, seed=42)
    
    print(f"\nTest completed!")
    print(f"Final metric: {metric:.2f}")
    print(f"Final weights: {weights}")


if __name__ == "__main__":
    main()

