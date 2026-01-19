"""
Robosuite-compatible learners that work with observation dictionaries.

These extend the interact_drive learners but override compute_features
to handle robosuite's dict-based observations instead of TensorFlow tensors.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interact_drive.learner.masked_learner import MaskedLLMPHRILearner
from interact_drive.learner.adapt_gated_llm_learner import AdaptGatedLLMPHRILearner


class RobosuiteMaskedLLMPHRILearner(MaskedLLMPHRILearner):
    """
    Robosuite-compatible MaskedLLMPHRILearner.
    
    Overrides compute_features to handle dict observations and returns numpy arrays.
    """
    
    def compute_features(self, trajectory: dict[str, list]) -> np.ndarray:
        """
        Compute summed features over trajectory using observation dicts.
        
        Args:
            trajectory: Dict with "state" (list of obs dicts) and "control" keys
            
        Returns:
            Summed features as numpy array
        """
        total_features = np.zeros(len(self.car.weights))
        states = trajectory["state"]
        
        for obs_dict in states:
            features = self.car.features(obs_dict)
            if features is not None:
                total_features += features
        
        return total_features
    
    def update_weights(
        self,
        planned_trajectory: dict[str, list],
        human_trajectory: dict[str, list],
    ) -> None:
        """Update weights with LLM-based feature selection (numpy-compatible)."""
        explanation = self.explanation

        # Extract features (now returns numpy arrays)
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

        # Apply mask to feature difference (no .numpy() needed)
        feature_diff = human_features - robot_features
        print("Feature difference:", feature_diff)
        print(
            f"New weights = {self.car.weights} + {self.learning_rate} * {feature_mask} * {feature_diff}"
        )
        
        # Update weights using masked difference
        new_weights = self.car.weights + self.learning_rate * feature_mask * feature_diff
        print("New weights =", new_weights)

        # Log the update
        self.log_update(
            planned_trajectory,
            human_trajectory,
            robot_features,
            human_features,
            self.car.weights,
            new_weights,
            explanation=explanation,
            feature_mask=feature_mask,
        )
        self.car.weights = new_weights

        print("\nFeature Analysis:")
        print(f"Robot trajectory length: {len(planned_trajectory['state'])}")
        print(f"Human trajectory length: {len(human_trajectory['state'])}")
        print(f"Robot features: {robot_features}")
        print(f"Human features: {human_features}")
        print(f"Delta phi: {feature_diff}")
        print("Updated weights:", new_weights)


class RobosuiteAdaptGatedLLMPHRILearner(AdaptGatedLLMPHRILearner):
    """
    Robosuite-compatible AdaptGatedLLMPHRILearner.
    
    Overrides compute_features to handle dict observations and returns numpy arrays.
    """
    
    def compute_features(self, trajectory: dict[str, list]) -> np.ndarray:
        """
        Compute summed features over trajectory using observation dicts.
        
        Args:
            trajectory: Dict with "state" (list of obs dicts) and "control" keys
            
        Returns:
            Summed features as numpy array
        """
        total_features = np.zeros(len(self.car.weights))
        states = trajectory["state"]
        
        for obs_dict in states:
            features = self.car.features(obs_dict)
            if features is not None:
                total_features += features
        
        return total_features
    
    def update_weights(
        self,
        planned_trajectory: dict[str, list],
        human_trajectory: dict[str, list],
    ) -> None:
        """Update weights with LLM-based feature selection and direction (numpy-compatible)."""
        # Get explanation from user - either from speech or text
        if self.use_speech_input and self.audio_file_path:
            print("Converting speech to text...")
            explanation = self._speech_to_text(self.audio_file_path)
            print(f"Transcribed explanation: {explanation}")

            with open(f"{self.audio_file_path[:-4]}.txt", "a") as explanation_file:
                explanation_file.write(explanation + "\n")
        else:
            explanation = self.explanation

        # Extract features (now returns numpy arrays)
        robot_features = self.compute_features(planned_trajectory)
        human_features = self.compute_features(human_trajectory)

        print("Planned trajectory features:", robot_features)
        print("Human trajectory features:", human_features)

        # Create feature values dictionary
        robot_feature_values = {
            name: robot_features[i] if self.method != 3 else np.nan 
            for i, name in enumerate(self.features_names)
        }
        human_feature_values = {
            name: human_features[i] if self.method != 3 else np.nan 
            for i, name in enumerate(self.features_names)
        }

        # Get feature mask and change values from LLM
        gate, mu, confidence = (
            self.feature_selector.select_relevant_features_and_directions(
                explanation,
                robot_feature_values,
                human_feature_values,
                self.car.weights,
            )
        )

        # Calculate feature difference (no .numpy() needed)
        feature_diff = human_features - robot_features

        # Cap μ at 5 times the feature difference
        capped_mu = np.zeros_like(mu)
        for i in range(len(mu)):
            sign = 1 if mu[i] > 0 else -1 if mu[i] < 0 else 0
            max_magnitude = 5.0 * abs(feature_diff[i])
            capped_magnitude = min(abs(mu[i]), max_magnitude)
            capped_mu[i] = sign * capped_magnitude

        print("Feature gate:", gate)
        print("Raw feature direction (μ):", mu)
        print("Feature confidence:", confidence)
        print("Capped feature direction (μ):", capped_mu)
        print("Feature difference:", feature_diff)
        
        # Apply mask to feature difference
        SIGMA = 1.2
        eps = 1e-4
        alpha = self.learning_rate

        def beta_lang(m, sigma=1.0, p=1.0, eps=1e-3):
            """power‑law variance: beta = (sigma f(m))^2"""
            return sigma**2 * ((1.0 - m) / (m + eps)) ** p

        def weights_fn(alpha, beta):
            w_phi = alpha * beta / (alpha + beta)
            w_mu = alpha / (alpha + beta)
            return w_phi, w_mu

        g = gate
        c = confidence

        beta = beta_lang(c, SIGMA, 2.0, eps)
        w_phi, w_mu = weights_fn(alpha, beta)

        print(f"weight update from phi: {w_phi * feature_diff}")
        print(f"weight update from mu: {w_mu * mu}")
        print(f"combined weight update: {w_phi * feature_diff + w_mu * mu}")
        print(f"old weights: {self.car.weights}")
        
        # Combined update based on method
        if self.method == 1:
            new_weights = self.car.weights + g * (w_phi * feature_diff + w_mu * mu)
            print(f"method: {self.method} QUICKLAP UPDATE")
            print(f"new weights: {new_weights}")
        elif self.method == 0:
            new_weights = self.car.weights + alpha * feature_diff
            print(f"method: {self.method} PHRI UPDATE")
            print(f"new weights: {new_weights}")
        elif self.method == 2 or self.method == 3:
            new_weights = self.car.weights + g * (w_mu * mu)
            print(f"method: {self.method} LANGUAGE UPDATE")
            print(f"new weights: {new_weights}")
        elif self.method == -1:
            new_weights = self.car.weights
            print(f"method: {self.method} NO UPDATE")
            print(f"new weights: {new_weights}")

        # Log the update
        self.log_update(
            planned_trajectory,
            human_trajectory,
            robot_features,
            human_features,
            self.car.weights,
            new_weights,
            explanation=explanation,
            gate=gate,
            confidence=confidence,
            mu=mu,
            capped_mu=capped_mu,
            w_phi=w_phi,
            w_mu=w_mu,
        )
        self.car.weights = new_weights

        print("\nFeature Analysis:")
        print(f"Robot trajectory length: {len(planned_trajectory['state'])}")
        print(f"Human trajectory length: {len(human_trajectory['state'])}")
        print(f"Robot features: {robot_features}")
        print(f"Human features: {human_features}")
        print(f"Delta phi: {feature_diff}")
        print(f"Feature gate (g): {gate}")
        print(f"Feature confidence (c): {confidence}")
        print(f"Raw feature direction (μ): {mu}")
        print(f"Capped feature direction (μ): {capped_mu}")
        print(f"Weight for difference (w_phi): {w_phi}")
        print(f"Weight for direction (w_mu): {w_mu}")
        print("Updated weights:", new_weights)

