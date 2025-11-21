import time

import numpy as np
import tensorflow as tf


class PHRILearner:
    """
    Physical Human-Robot Interaction Learner for autonomous vehicles.
    Updates reward weights based on human corrections during driving.
    """

    def __init__(
        self, car, learning_rate: float = 0.5, log_file: str = "learning_log.txt"
    ):
        """
        Args:
            car: The autonomous vehicle being controlled
            learning_rate: Step size for weight updates
        """
        self.car = car
        self.learning_rate = learning_rate
        self.correction_buffer = []
        self.update_history = []
        self.feature_differences = []
        self.trajectory_lengths = []
        self.update_counter = 0
        self.log_file = log_file

    def update_weights(self, planned_trajectory, human_trajectory):
        """Update reward weights based on human correction.

        Args:
            planned_trajectory: Original trajectory from planner
            human_trajectory: List of {state, control} from human input
        """

        # Extract relevant features
        robot_features = self.compute_features(planned_trajectory)
        human_features = self.compute_features(human_trajectory)

        print("Planned trajectory features:", robot_features)
        print("Human trajectory features:", human_features)

        # Log data
        self.trajectory_lengths.append(
            {"robot": len(planned_trajectory), "human": len(human_trajectory)}
        )

        # Compute update
        feature_diff = human_features - robot_features

        self.feature_differences.append(
            {
                "delta_phi": feature_diff.numpy(),
                "robot_features": robot_features.numpy(),
                "human_features": human_features.numpy(),
            }
        )
        old_weights = self.car.weights.copy()

        feature_specific_lr = np.ones(len(self.car.weights)) * self.learning_rate
        new_weights = self.car.weights + feature_specific_lr * feature_diff.numpy()

        self.log_update(
            planned_trajectory,
            human_trajectory,
            robot_features.numpy(),
            human_features.numpy(),
            old_weights,
            new_weights,
        )
        self.car.weights = new_weights

        print("\nFeature Analysis:")
        print(f"Robot trajectory length: {len(planned_trajectory['state'])}")
        print(f"Human trajectory length: {len(human_trajectory['state'])}")
        print(f"Robot features: {robot_features.numpy()}")
        print(f"Human features: {human_features.numpy()}")
        print(f"Delta phi: {feature_diff.numpy()}")
        print("Updated weights:", new_weights)

    def compute_features(self, trajectory: dict[str, list[np.ndarray]]) -> tf.Tensor:
        """Compute average features over trajectory."""
        total_features = tf.zeros(len(self.car.weights))
        states = trajectory["state"]
        controls = trajectory["control"]

        # Add debug prints
        print("Processing human trajectory:")
        print(f"States shape: {np.array(states).shape}")
        print(f"Controls shape: {np.array(controls).shape}")

        for state, control in zip(states, controls):
            state = [tf.convert_to_tensor(state, dtype=tf.float32)]
            control = tf.convert_to_tensor(control, dtype=tf.float32)
            features = self.car.features(state)
            if features is not None:
                total_features += features
        return total_features

    def log_update(
        self,
        planned_trajectory,
        human_trajectory,
        robot_features,
        human_features,
        old_weights,
        new_weights,
    ):
        """Log learning details to file"""
        with open(self.log_file, "a") as f:
            f.write(f"\n=== Update {self.update_counter} ===\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Log trajectories
            f.write("Planned Trajectory:\n")
            for i, state in enumerate(planned_trajectory):
                f.write(f"Step {i}: State: {state}\n")

            f.write("\nHuman Trajectory:\n")
            for i, state in enumerate(human_trajectory["state"]):
                f.write(
                    f"Step {i}: State: {state}, Control: {human_trajectory['control'][i]}\n"
                )

            # Log features
            f.write("\nFeatures:\n")
            feature_names = [
                "Speed",
                "Lane",
                "Collision",
                "OffRoad",
                "Cone",
                "Car",
                "Puddle",
            ]
            f.write("Robot features: ")
            for name, value in zip(feature_names, robot_features):
                f.write(f"{name}: {value:.4f} ")
            f.write("\nHuman features: ")
            for name, value in zip(feature_names, human_features):
                f.write(f"{name}: {value:.4f} ")

            # Log weights
            f.write("\n\nWeight Updates:\n")
            for i, (old_w, new_w) in enumerate(zip(old_weights, new_weights)):
                f.write(
                    f"{feature_names[i]}: {old_w:.4f} -> {new_w:.4f} (Δ = {new_w-old_w:.4f})\n"
                )

            f.write("\n" + "=" * 50 + "\n")

        self.update_counter += 1
