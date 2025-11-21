import numpy as np
import openai

from interact_drive.car.car import Car
from interact_drive.learner.adapt_gated_llm_selector import AdaptGatedLLMFeatureSelector
from interact_drive.learner.phri_learner import PHRILearner


class AdaptGatedLLMPHRILearner(PHRILearner):
    """Extended PHRILearner with LLM-based feature selection and direction guidance."""

    def __init__(
        self,
        car: Car,
        explanation: str,
        feature_descriptions: dict[str, str],
        learning_rate: float = 1.0,
        log_file: str = "learning_log_llm.txt",
        openai_api_key: str | None = None,
        use_speech_input: bool = False,
        audio_file_path: str | None = None,
        method: int = 1, # 0 for phri, 2 for quicklap, 3 for language
    ):
        super().__init__(car, learning_rate, log_file)
        self.explanation = explanation
        self.features_names = list(feature_descriptions.keys())
        self.feature_descriptions = feature_descriptions
        self.feature_selector = AdaptGatedLLMFeatureSelector(
            feature_descriptions, openai_api_key
        )
        self.use_speech_input = use_speech_input
        self.audio_file_path = audio_file_path
        self.openai_api_key = openai_api_key
        self.method = method

    def _speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech audio file to text using OpenAI's Whisper API."""
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            with open(audio_file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript.strip()
        except Exception as e:
            print(f"Error in speech-to-text conversion: {e}")
            # Fallback to the original explanation if speech-to-text fails
            return self.explanation

    def set_audio_file_path(self, audio_file_path: str) -> None:
        """Update the audio file path for speech-to-text conversion."""
        self.audio_file_path = audio_file_path

    def update_weights(
        self,
        planned_trajectory: dict[str, list[np.ndarray]],
        human_trajectory: dict[str, list[np.ndarray]],
    ) -> None:
        """Update weights with LLM-based feature selection and direction guidance."""
        # Get explanation from user - either from speech or text
        if self.use_speech_input and self.audio_file_path:
            print("Converting speech to text...")
            explanation = self._speech_to_text(self.audio_file_path)
            print(f"Transcribed explanation: {explanation}")

            with open(f"{self.audio_file_path[:-4]}.txt", "a") as explanation_file:
                explanation_file.write(explanation + "\n")
        else:
            explanation = self.explanation

        # Extract features
        robot_features = self.compute_features(planned_trajectory)
        human_features = self.compute_features(human_trajectory)

        print("Planned trajectory features:", robot_features)
        print("Human trajectory features:", human_features)

        # Create feature values dictionary
        robot_feature_values = {
            name: robot_features[i] if self.method != 3 else np.nan for i, name in enumerate(self.features_names)
        }
        human_feature_values = {
            name: human_features[i] if self.method != 3 else np.nan for i, name in enumerate(self.features_names)
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

        # Calculate feature difference for capping
        feature_diff = human_features - robot_features

        # Cap μ at 5 times the feature difference
        capped_mu = np.zeros_like(mu)
        for i in range(len(mu)):
            # Get the sign of μ
            sign = 1 if mu[i] > 0 else -1 if mu[i] < 0 else 0

            # Get the max allowed magnitude (5 * abs of feature difference)
            max_magnitude = 5.0 * abs(feature_diff[i])

            # Cap the magnitude while preserving sign
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

        def weights(alpha, beta):
            w_phi = alpha * beta / (alpha + beta)
            w_mu = alpha / (alpha + beta)
            return w_phi, w_mu

        g = gate
        c = confidence

        beta = beta_lang(c, SIGMA, 2.0, eps)
        w_phi, w_mu = weights(alpha, beta)

        print(f"weight update from phi: {w_phi * feature_diff.numpy()}")
        print(f"weight update from mu: {w_mu * mu}")
        print(f"combined weight update: {w_phi * feature_diff.numpy() + w_mu * mu}")
        print(f"old weights: {self.car.weights}")
        # Combined update

        if self.method == 1:
            new_weights = self.car.weights + g * (w_phi * feature_diff.numpy() + w_mu * mu)
            print(f"method: {self.method} QUICKLAP UPDATE")
            print(f"new weights: {new_weights}")
        elif self.method == 0:
            new_weights = self.car.weights + alpha * feature_diff.numpy()
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
            robot_features.numpy(),
            human_features.numpy(),
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
        print(f"Robot features: {robot_features.numpy()}")
        print(f"Human features: {human_features.numpy()}")
        print(f"Delta phi: {feature_diff.numpy()}")
        print(f"Feature gate (g): {gate}")
        print(f"Feature confidence (c): {confidence}")
        print(f"Raw feature direction (μ): {mu}")
        print(f"Capped feature direction (μ): {capped_mu}")
        print(f"Weight for difference (w_phi): {w_phi}")
        print(f"Weight for direction (w_mu): {w_mu}")
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
        gate: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        mu: np.ndarray | None = None,
        capped_mu: np.ndarray | None = None,
        w_phi: np.ndarray | None = None,
        w_mu: np.ndarray | None = None,
    ):
        """Extended logging with explanation, feature mask, and direction."""
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
            if gate is not None:
                f.write(f"Feature Gate (g): {gate}\n")
            if confidence is not None:
                f.write(f"Feature Confidence (c): {confidence}\n")
            if mu is not None:
                f.write(f"Raw Feature Change (μ): {mu}\n")
            if capped_mu is not None:
                f.write(f"Capped Feature Change (μ): {capped_mu}\n")
            if w_phi is not None and w_mu is not None:
                f.write(f"Weight for difference (w_phi): {w_phi}\n")
                f.write(f"Weight for direction (w_mu): {w_mu}\n")
            f.write("\n" + "=" * 50 + "\n")
