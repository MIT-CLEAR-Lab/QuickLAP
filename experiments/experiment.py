import json
import os
from datetime import datetime
from typing import Any, Callable

import numpy as np

from experiments.intervention_car import InterventionCar
from interact_drive.learner.phri_learner import PHRILearner
from interact_drive.world import CarWorld


class InterventionExperiment:
    def __init__(
        self,
        exp_name: str,
        save_dir: str | None = None,
        verbose: bool = False,
    ):
        """Initialize the cone avoidance experiment."""
        self.exp_name = exp_name
        self.verbose = verbose
        # Set up save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            save_dir = os.path.join("experiments", "results", exp_name, timestamp)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.step_data = []  # Initialize step data list

    def setup_world(
        self, learner_factory: Callable[[InterventionCar], PHRILearner], seed: int
    ) -> CarWorld:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_feature_trajectory(self) -> np.ndarray:
        """Return the feature trajectory extracted from step data."""
        if not self.step_data:
            return np.array([])

        return np.array([step["features"] for step in self.step_data])

    def get_metrics(self, t: int, car: InterventionCar) -> dict[str, Any]:
        """Log metrics for the current timestep."""
        # Calculate distance to nearest cone
        car_pos = car.state.numpy()[:2]
        cone_positions = [obs[1] for obs in car.env.obstacles]
        min_dist = min(
            np.linalg.norm(car_pos - np.array(cone_pos)) for cone_pos in cone_positions
        )

        # Get current features and reward
        current_features = car.features([car.state])
        current_reward = float(car.reward_fn([car.state]))

        # Store timestep data
        return {
            "timestep": t,
            "position": car_pos.tolist(),
            "velocity": float(car.state.numpy()[2]),
            "heading": float(car.state.numpy()[3]),
            "min_cone_distance": float(min_dist),
            "weights": car.weights.tolist(),
            "features": [float(f) for f in current_features.numpy()],
            "reward": current_reward,
            "is_intervention": car.is_intervention(),
        }

    def evaluate_intervention(self, car: InterventionCar) -> float:
        """Average reward after the last intervention period."""
        last_intervention = max(end for _, end in car.intervention_intervals)
        if last_intervention >= len(self.step_data):
            raise ValueError(
                "Last intervention period exceeds the length of step data."
            )
        features = np.array(
            [
                self.step_data[i]["features"]
                for i in range(last_intervention + 1, len(self.step_data))
            ]
        )
        return np.sum(car.expert_weights.reshape(1, -1) @ features.T)

    def save_results(self, metric: float):
        """Save all experiment results and generate visualizations."""
        # Save raw data
        results_file = os.path.join(self.save_dir, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump({"steps": self.step_data, "reward": metric}, f, indent=2)

    def run(
        self, learner_factory: Callable[[InterventionCar], PHRILearner], seed: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Run the complete experiment. Returns the final average reward metric."""
        print(f"\nStarting {self.exp_name} experiment...")
        print(f"Seed: {seed}")
        print(f"Results will be saved to: {self.save_dir}")

        # Set up world and car
        self.step_data = []  # Reset step data
        world = self.setup_world(learner_factory, seed)
        car = world.cars[0]
        assert isinstance(
            car, InterventionCar
        ), "The first car in the world must be an instance of InterventionCar"

        # Create experiment configuration record
        config = {
            "experiment_name": self.exp_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "car_config": {
                "planner_type": car.planner_type,
                "initial_weights": car.weights.tolist(),
                "initial_position": car.init_state.numpy().tolist(),
            },
            "world_config": {
                "obstacles": [{"type": t, "position": p} for t, p in world.obstacles],
                "num_lanes": len(world.lanes),
            },
            "intervention_config": {
                "expert_weights": car.expert_weights.tolist(),
                "intervention_intervals": car.intervention_intervals,
            },
        }

        # Save configuration
        config_file = os.path.join(self.save_dir, "experiment_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Run simulation
        final_learned_weights_to_return = None
        try:
            print("\nRunning simulation...")
            for t in range(car.world_horizon):
                world.step()
                self.step_data.append(self.get_metrics(t, car))
                if self.verbose:
                    world.render()

                # Print progress
                if t % 20 == 0:
                    print(f"Step {t}/{car.world_horizon} completed")

            if hasattr(car, "weights") and car.weights is not None:
                final_learned_weights_to_return = np.array(car.weights)
            else:
                # If car.weights is not available, it will remain None, which is acceptable
                print(
                    f"Warning: car.weights attribute not found or is None for {self.exp_name}. Returning None for learned_weights."
                )

        except Exception as e:
            print(f"\nError during experiment: {e}")
            raise e
        finally:
            # Save results even if there's an error
            feature_trajectory = self.get_feature_trajectory()
            metric = self.evaluate_intervention(car)
            print(f"\nExperiment metrics: {metric}")
            self.save_results(metric)
            print(f"\nExperiment completed. Results saved to {self.save_dir}")
            return metric, feature_trajectory, final_learned_weights_to_return
