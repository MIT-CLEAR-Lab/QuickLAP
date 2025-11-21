from interact_drive.learner.phri_learner import PHRILearner


class OracleLearner(PHRILearner):
    """
    Oracle learner that uses predefined optimal weights without learning.
    Now accepts custom weights as input parameters.
    """

    def __init__(self, car, custom_weights, log_file: str = "oracle_log.txt"):
        """
        Initialize with zero learning rate to prevent updates.

        Args:
            car: The car object this learner controls
            custom_weights: Optional custom weights to use instead of defaults
            log_file: File to log updates to
        """
        print("*** OracleLearner with Custom Weights Initializing ***")

        # First call super to initialize the learner
        super().__init__(car, learning_rate=0.0, log_file=log_file)

        # Store custom weights if provided
        self.custom_weights = custom_weights

    def update_weights(self, planned_trajectory, human_trajectory):
        """Override to prevent weight updates, just log the data."""
        # Extract features for logging

        try:
            print(f"Expected shape: {self.car.weights_tf.shape}")
            print(f"Custom weights shape: {self.custom_weights}")
            print(f"Car weights shape: {self.custom_weights}")
            old_weights = self.car.weights
            self.car.weights = self.custom_weights

            print("\nFeature Analysis (Oracle):")
            print(f"Robot trajectory length: {len(planned_trajectory['state'])}")
            print(f"Human trajectory length: {len(human_trajectory['state'])}")
            print(f"Using oracle weights: {self.car.weights}")
            print(f"Old weights: {old_weights}")

            # Return the optimal weights and features for regret calculation if needed
            return {
                "optimal_weights": self.car.weights,
            }

        except Exception as e:
            print(f"Error in update_weights: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_optimal_weights(self):
        """Return the optimal weights for regret calculation."""
        if hasattr(self.car, "weights_tf"):
            return self.car.weights_tf.numpy()
        else:
            return self.car.weights
