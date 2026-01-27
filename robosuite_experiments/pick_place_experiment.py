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
from hierarchical_mpc_arm import HierarchicalMPCArm
from robosuite.devices import Keyboard
from robosuite_experiments.arm_feature_utils import DEFAULT_EXPERT_WEIGHTS, DEFAULT_BASE_WEIGHTS


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
        use_physical_input: bool = False,
        input_sensitivity: float = 1.0,
    ):
        """
        Initialize the experiment.
        
        Args:
            exp_name: Name for this experiment
            save_dir: Directory to save results (auto-generated if None)
            verbose: Whether to print detailed logs
            horizon: Episode length in timesteps
            use_physical_input: Whether to enable keyboard input for human intervention
            input_sensitivity: Sensitivity for keyboard input (default: 1.0)
        """
        self.exp_name = exp_name
        self.verbose = verbose
        self.horizon = horizon
        self.use_physical_input = use_physical_input
        self.input_sensitivity = input_sensitivity
        
        # Set up save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            save_dir = os.path.join("logs", f"robosuite_{exp_name}_{timestamp}")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.step_data = []
        self.keyboard_device = None
    
    def setup_world(
        self,
        learner_factory: Callable,
        seed: int,
        utterance: str = "Go faster"
    ) -> tuple[ArmWorld, HierarchicalMPCArm]:
        """
        Set up the robosuite environment and intervention arm.
        
        Args:
            learner_factory: Factory function to create learner
            seed: Random seed for reproducibility
            utterance: What human says during intervention
            
        Returns:
            Tuple of (world, arm)
        """
        # Enable renderer if using physical input (need visual feedback)
        use_renderer = self.verbose or self.use_physical_input
        
        # Create world
        world = ArmWorld(
            has_renderer=use_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=self.horizon,
            seed=seed,
        )
        
        # Create intervention arm with learner
        # Learner factory expects an arm-like object with weights and features
        # We'll pass a temporary arm to get the learner, then create the real arm
        
        # Base weights for robot (initial policy)
        base_weights = DEFAULT_BASE_WEIGHTS.copy()
        
        if self.use_physical_input:
            # When using physical input, human IS the expert
            # No simulated expert weights - set to None to indicate human control
            expert_weights = None
        else:
            # Simulated expert weights: [red_dist, green_dist, velocity, collision, joints]
            expert_weights = DEFAULT_EXPERT_WEIGHTS.copy()
        
        # Create arm first without learner
        arm = HierarchicalMPCArm(
            world=world,
            learner=None,  # Will set after creation
            utterance=utterance,
            expert_weights=expert_weights,
            intervention_interval=(780, 790),
            base_weights=base_weights,
            seed=seed,
        )
        
        # Now create learner with the arm
        learner = learner_factory(arm)
        arm.learner = learner
        
        return world, arm
    
    def get_metrics(self, t: int, arm: HierarchicalMPCArm, obs: dict) -> dict[str, Any]:
        """
        Collect metrics for current timestep.
        
        Args:
            t: Current timestep
            arm: HierarchicalMPCArm instance
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
    
    def evaluate_performance(self, arm: HierarchicalMPCArm) -> float:
        """
        Evaluate overall performance.
        
        When using simulated expert: uses cumulative reward based on expert weights.
        When using physical input: uses cumulative reward based on current learned weights.
        
        Args:
            arm: HierarchicalMPCArm instance
            
        Returns:
            Performance metric (higher is better)
        """
        feature_trajectory = self.get_feature_trajectory()
        
        if len(feature_trajectory) == 0:
            return 0.0
        
        if self.use_physical_input or arm.expert_weights is None:
            # Physical input mode: evaluate using current learned weights
            # (no ground-truth expert weights since human is the expert)
            rewards = feature_trajectory @ arm.weights
        else:
            # Simulated expert mode: evaluate using expert weights
            rewards = feature_trajectory @ arm.expert_weights
            
        return float(np.sum(rewards))
    
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
        if self.use_physical_input:
            print("Physical input ENABLED (robosuite keyboard)")
        print(f"Results will be saved to: {self.save_dir}")
        
        # Set up environment and arm
        self.step_data = []
        world, arm = self.setup_world(learner_factory, seed, utterance)
        
        # Initialize robosuite's built-in keyboard device if enabled
        if self.use_physical_input:
            self.keyboard_device = Keyboard(
                env=world.env,
                pos_sensitivity=self.input_sensitivity,
                rot_sensitivity=self.input_sensitivity * 0.5,
            )
            # Wire up keyboard callback to the viewer
            world.env.viewer.add_keypress_callback(self.keyboard_device.on_press)
            print("Keyboard input initialized. Click on viewer window to focus!")
        
        # Save experiment configuration
        config = {
            "experiment_name": self.exp_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "utterance": utterance,
            "horizon": self.horizon,
            "use_physical_input": self.use_physical_input,
            "arm_config": {
                "initial_weights": arm.weights.tolist(),
                # expert_weights is None when using physical input (human is the expert)
                "expert_weights": arm.expert_weights.tolist() if arm.expert_weights is not None else "human",
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
                # Get robot's planned action (before human input)
                robot_action = arm.get_action(obs)
                action = robot_action.copy()
                
                # Add keyboard input if enabled (robosuite built-in device)
                # Only allow physical input during TRANSPORT/MOVE phase (when human guidance matters)
                # Skip first 10 frames to let keyboard device initialize (avoid false positives)
                in_transport_phase = arm.task_phase in ["transport", "move"]
                if self.use_physical_input and self.keyboard_device and t >= 10 and in_transport_phase:
                    # Get human input from keyboard device
                    # Returns dict with 'right_delta' (6,) and 'right_gripper' keys
                    device_action = self.keyboard_device.input2action()
                    
                    if device_action is not None:
                        right_delta = device_action.get("right_delta", np.zeros(6))
                        right_gripper = device_action.get("right_gripper", 0)
                        if isinstance(right_gripper, np.ndarray):
                            right_gripper = float(right_gripper.item()) if right_gripper.size == 1 else 0
                        
                        # Check if human is actively providing input (position/orientation only)
                        # Use higher thresholds to avoid false positives from device noise
                        # NOTE: We ignore gripper input during transport - robot must keep holding the block
                        delta_magnitude = np.linalg.norm(right_delta)
                        if delta_magnitude > 0.01:
                            # Apply human correction to position/orientation ONLY
                            # Do NOT override gripper - robot needs to keep it closed during transport
                            action[:6] += right_delta
                            
                            # Signal intervention for learning
                            arm.signal_physical_intervention(obs, robot_action, action)
                
                # Update intervention state (handles cooldown and triggers learning)
                if self.use_physical_input:
                    arm.update_physical_intervention_state()
                
                # Step environment
                obs, _, done, _ = world.step(action)
                
                # Collect metrics
                metrics = self.get_metrics(t, arm, obs)
                self.step_data.append(metrics)
                
                # Render if verbose (required for visual feedback with physical input)
                if self.verbose or self.use_physical_input:
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
            self.keyboard_device = None
            world.close()
        
        return performance_metric, feature_trajectory, final_learned_weights


def main():
    """Test the experiment with a simple learner."""
    import argparse
    from robosuite_phri_learner import RobosuitePHRILearner
    
    parser = argparse.ArgumentParser(description="Run pick-place experiment")
    parser.add_argument(
        "--physical-input", 
        action="store_true",
        help="Enable keyboard input for human intervention"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Episode length in timesteps"
    )
    args = parser.parse_args()
    
    # If you want visualization, run with: mjpython pick_place_experiment.py
    experiment = PickPlaceExperiment(
        verbose=False, 
        horizon=args.horizon,
        use_physical_input=args.physical_input,
    )
    
    def learner_factory(arm):
        return RobosuitePHRILearner(arm, log_file="robosuite_test.txt")
    
    metric, features, weights = experiment.run(learner_factory, seed=42)
    
    print(f"\nTest completed!")
    print(f"Final metric: {metric:.2f}")
    print(f"Final weights: {weights}")


if __name__ == "__main__":
    main()

