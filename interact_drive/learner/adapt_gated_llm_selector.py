import numpy as np
import openai
import json
import re


class AdaptGatedLLMFeatureSelector:
    """Uses LLM to determine relevant features and desired changes for interventions."""

    def __init__(
        self,
        feature_descriptions: dict[str, str],
        api_key: str | None,
        main_model: str = "gpt-4o",
        gate_model: str = "gpt-4o",
    ) -> None:
        """Initialize with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.feature_descriptions = feature_descriptions
        self.main_model = main_model
        self.gate_model = gate_model

    def select_relevant_features_and_directions(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
        current_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use LLM to determine which features are relevant and how they should change."""
        gate_prompt = self._construct_prompt(
            explanation, robot_feature_values, human_feature_values
        )
        gate_prompt += "\nFor absolutely EVERY feature above, determine:\n"
        gate_prompt += "1. How relevant is this feature to the intervention? (gate score 0.0 or 1.0)\n"

        gate_json = self._call_llm(
            gate_prompt,
            model=self.gate_model,
            temperature=0.1,
            system_msg=(
                "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
                "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
                "Your task is to determine which features are relevant to a given intervention explanation, given the "
                "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
                "that the human increased the feature value.\n"
                "If a feature is shown as not a number, the value is not available but it is not necessarily irrelevant.\n"
                "Note that a feature may be irrelevant even if it has a large change in value. Only output features that are relevant "
                "to the intevention explanation\n"
                "Output STRICT JSON with the single key 'gate': a list of attention gates "
                "scores (one per feature, 0.0 or 1.0). NO other keys."
            ),
            expect_keys=("gate",),
        )
        gate = np.asarray(gate_json["gate"], dtype=float)

        mc_prompt = self._construct_prompt(
            explanation, robot_feature_values, human_feature_values
        )
        ordered_feature_names = list(robot_feature_values.keys())

        mc_prompt += "\nCurrent Reward Weights after a Physical Intervention Update (these are the weights before applying the 'mu' change you will suggest):\n"
        if len(current_weights) == len(ordered_feature_names):
            for i, feature_name in enumerate(ordered_feature_names):
                mc_prompt += f"- {feature_name}: {current_weights[i]:.3f}\n"
        else:
            mc_prompt += "- Note: Could not display current weights due to a mismatch between number of features and weights provided.\n"

        # Updated questions for mu and confidence
        mc_prompt += "\nNow, for absolutely EVERY feature (considering the explanation, feature changes, and current weights):\n"
        mc_prompt += "1. What absolute change with direction (this will be your 'mu') would support this intervention? Consider the scale of the features, and the current weights.\n"
        mc_prompt += (
            "2. How confident are you in your decision? (confidence score 0.0-1.0)\n"
        )

        mc_json = self._call_llm(
            mc_prompt,
            model=self.main_model,
            temperature=0.3,
            system_msg=(
                "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
                "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
                "Your reward function is the sum of the features. You want to maximize the reward function."
                "Feature values are between 0 and 1. Look at the feature descriptions to understand the scale of the features."
                "Your task is to determine for EACH feature how much in magnitude should the weight of the feature be changed to support the intervention and human preference (mu between 0 and 6), and how confident you are in your decision (confidence between 0 and 1, be conservative), given the "
                "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
                "that the human increased the feature value.\n"
                "If a feature is shown as not a number, the value is not available but it is not necessarily irrelevant.\n"
                "FOR ABSOLUTELY EVERY FEATURE, return ONLY the values in this exact format. DO NOT MISS ANY FEATURES.\n"
                "OUTPUT (strict JSON, single line)"
                "    {"
                "    'mu':   [u1, u2, …, uN],"
                "    'confidence': [c1, c2, …, cN],"
                "    }"
            ),
            expect_keys=("mu", "confidence"),
        )
        mu = np.asarray(mc_json["mu"], dtype=float)
        confidence = np.asarray(mc_json["confidence"], dtype=float)

        return gate, mu, confidence

    def _construct_prompt(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
    ) -> str:
        """Construct prompt for the LLM."""

        prompt = f"""
                Human Driver Intervention Explanation:
                {explanation}

                Current Feature Values:
                """
        for feature, value in robot_feature_values.items():
            human_val = human_feature_values[feature]
            change = human_val - value
            direction = (
                "INCREASED"
                if change > 0
                else (
                    "DECREASED"
                    if change < 0
                    else "MAY HAVE CHANGED" if np.isnan(change) else "DID NOT CHANGE"
                )
            )

            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                # f"Robot value: {value:.3f}, Human value: {human_val:.3f}, Change: {change:+.3f}, the human ({direction} this feature)\n"
                f"feature change after intervention: {change:+.3f}, the human ({direction} this feature)\n"
            )

        return prompt

    def _call_llm(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        system_msg: str,
        expect_keys: tuple[str, ...],
    ) -> dict:
        """Shared wrapper around openai.chat.completions.create with JSON parsing."""
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):  # strip fences if any
            content = "\n".join(
                line
                for line in content.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        # sometimes +1.23 sneaks in → remove leading '+'
        content = re.sub(r"\+([0-9]+(\.[0-9]+)?)", r"\1", content)

        try:
            parsed = json.loads(content)
            missing = [k for k in expect_keys if k not in parsed]
            if missing:
                raise KeyError(f"Missing keys {missing}")
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output:\n{content}") from e
