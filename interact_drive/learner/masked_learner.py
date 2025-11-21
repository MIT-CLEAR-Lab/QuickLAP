import numpy as np

from interact_drive.car.car import Car
from interact_drive.learner.masked_selector import (
    MaskedLLMFeatureSelector,
    MaskedLLMFeatureSelectorPhiH,
    MaskedLLMFeatureSelectorPhiR,
    MaskedLLMFeatureSelectorDPhi,
    MaskedLLMFeatureSelectorDPhiSign,
)
from interact_drive.learner.phri_learner import PHRILearner


SELECTORS = {
    "default": MaskedLLMFeatureSelector,
    "phi_h": MaskedLLMFeatureSelectorPhiH,
    "phi_r": MaskedLLMFeatureSelectorPhiR,
    "d_phi": MaskedLLMFeatureSelectorDPhi,
    "d_phi_sign": MaskedLLMFeatureSelectorDPhiSign,
}


class MaskedLLMPHRILearner(PHRILearner):
    """Extended PHRILearner with LLM-based feature selection."""

    def __init__(
        self,
        car: Car,
        explanation: str,
        feature_descriptions: dict[str, str],
        learning_rate: float = 1.0,
        log_file: str = "learning_log_masked.txt",
        openai_api_key: str | None = None,
        selector: str = "default",
    ):
        super().__init__(car, learning_rate, log_file)
        self.explanation = explanation
        self.features_names = list(feature_descriptions.keys())
        self.feature_discriptions = feature_descriptions
        self.feature_selector = SELECTORS[selector](
            feature_descriptions, openai_api_key
        )

    def update_weights(
        self,
        planned_trajectory: dict[str, list[np.ndarray]],
        human_trajectory: dict[str, list[np.ndarray]],
    ) -> None:
        """Update weights with LLM-based feature selection."""
        # Get explanation from user
        explanation = self.explanation

        # Extract features
        robot_features = self.compute_features(planned_trajectory)
        human_features = self.compute_features(human_trajectory)

        print("Planned trajectory features:", robot_features)
        print("Human trajectory features:", human_features)

        # Create feature values dictionary
        robot_feature_values = {
            name: robot_features[i] for i, name in enumerate(self.features_names)
        }
        human_feature_values = {
            name: human_features[i] for i, name in enumerate(self.features_names)
        }

        # Get feature mask from LLM
        feature_mask = self.feature_selector.select_relevant_features(
            explanation, robot_feature_values, human_feature_values
        )
        print("Feature mask:", feature_mask)
        print("Old weights:", self.car.weights)

        # Apply mask to feature difference
        feature_diff = human_features - robot_features
        print("Feature difference:", feature_diff)
        print(
            f"New weights = {self.car.weights} + {self.learning_rate} * {feature_mask} * {feature_diff}"
        )
        print(
            "New weights =",
            self.car.weights + self.learning_rate * feature_mask * feature_diff.numpy(),
        )
        # Update weights using masked difference
        new_weights = (
            self.car.weights + self.learning_rate * feature_mask * feature_diff.numpy()
        )

        # Log the update
        self.log_update(
            planned_trajectory,
            human_trajectory,
            robot_features.numpy(),
            human_features.numpy(),
            self.car.weights,
            new_weights,
            explanation=explanation,
            feature_mask=feature_mask,
        )
        self.car.weights = new_weights

        print("\nFeature Analysis:")
        print(f"Robot trajectory length: {len(planned_trajectory['state'])}")
        print(f"Human trajectory length: {len(human_trajectory['state'])}")
        print(f"Robot features: {robot_features.numpy()}")
        print(f"Human features: {human_features.numpy()}")
        print(f"Delta phi: {feature_diff.numpy()}")
        print("Updated weights:", new_weights)

    def log_update(
        self,
        planned_trajectory: dict[str, list[np.ndarray]],
        human_trajectory: dict[str, list[np.ndarray]],
        robot_features: np.ndarray,
        human_features: np.ndarray,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        explanation: str = "",
        feature_mask: np.ndarray | None = None,
    ):
        """Extended logging with explanation and feature mask."""
        with open(self.log_file, "a") as f:
            # Original logging
            super().log_update(
                planned_trajectory,
                human_trajectory,
                robot_features,
                human_features,
                old_weights,
                new_weights,
            )

            # Additional logging for explanation and feature selection
            if explanation:
                f.write(f"\nIntervention Explanation: {explanation}\n")
            if feature_mask is not None:
                f.write(f"Selected Features: {feature_mask}\n")
            f.write("\n" + "=" * 50 + "\n")
