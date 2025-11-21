import numpy as np
import openai


class MaskedLLMFeatureSelector:
    """Uses LLM to determine relevant features for interventions."""

    def __init__(
        self, feature_descriptions: dict[str, str], api_key: str | None
    ) -> None:
        """Initialize with OpenAI API key."""
        self.feature_descriptions = feature_descriptions
        self.client = openai.OpenAI(api_key=api_key)

    def select_relevant_features(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
    ) -> np.ndarray:
        """Use LLM to determine which features are relevant."""
        prompt = self.construct_prompt(
            explanation, robot_feature_values, human_feature_values
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": self.construct_system_prompt(),
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_text = response.choices[0].message.content
        assert response_text is not None
        return self.parse_response(response_text)

    def parse_response(self, response: str) -> np.ndarray:
        """Parse LLM response to get feature mask."""
        relevant_features = [f.strip() for f in response.split(",")]
        feature_mask = np.zeros(len(self.feature_descriptions))
        for i, feature in enumerate(self.feature_descriptions.keys()):
            if feature in relevant_features:
                feature_mask[i] = 1
        return feature_mask

    def construct_system_prompt(self) -> str:
        return (
            "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
            "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
            "Your task is to determine which features are relevant to a given intervention explanation, given the "
            "current feature values of the robot and human trajectories.\n"
            "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
        )

    def construct_prompt(
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
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Robot value: {value:.3f}, Human value: {human_feature_values[feature]:.3f}\n"
            )
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt


class MaskedLLMFeatureSelectorPhiH(MaskedLLMFeatureSelector):
    def construct_prompt(
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
        for feature, value in human_feature_values.items():
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Human value: {value:.3f}\n"
            )
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt

    def construct_system_prompt(self) -> str:
        return (
            "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
            "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
            "Your task is to determine which features are relevant to a given intervention explanation, given the "
            "current feature values of the human trajectory.\n"
            "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
        )


class MaskedLLMFeatureSelectorPhiR(MaskedLLMFeatureSelector):
    def construct_prompt(
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
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Robot value: {value:.3f}\n"
            )
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt

    def construct_system_prompt(self) -> str:
        return (
            "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
            "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
            "Your task is to determine which features are relevant to a given intervention explanation, given the "
            "current feature values of the robot trajectory.\n"
            "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
        )


class MaskedLLMFeatureSelectorDPhi(MaskedLLMFeatureSelector):
    def construct_prompt(
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
        for feature, value in human_feature_values.items():
            change = value - robot_feature_values[feature]
            direction = (
                "INCREASED"
                if change > 0
                else "DECREASED" if change < 0 else "UNCHANGED"
            )
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Feature change: {change:.3f}, the human ({direction} the reward weights of this feature)\n"
            )
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt

    def construct_system_prompt(self) -> str:
        return (
            "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
            "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
            "Your task is to determine which features are relevant to a given intervention explanation, given the "
            "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
            "that the human increased the feature value.\n"
            "Note that a feature may be irrelevant even if it has a large change in value. Only output features that are relevant "
            "to the intevention explanation\n"
            "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
        )


class MaskedLLMFeatureSelectorDPhiSign(MaskedLLMFeatureSelector):
    def construct_prompt(
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
        for feature, value in human_feature_values.items():
            change = (
                "increased" if value > robot_feature_values[feature] else "decreased"
            )
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Feature change: The human {change} this feature\n"
            )
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt

    def construct_system_prompt(self) -> str:
        return (
            "You are an expert in autonomous vehicle control analyzing driver interventions. In this task, a human "
            "driver has intervened to correct the behavior of a robot car and has provided an explanation of the intervention. "
            "Your task is to determine which features are relevant to a given intervention explanation, given the "
            "change in feature values of the human trajectory compared to the robot trajectory.\n"
            "Note that a feature may be irrelevant even if it has a large change in value. Only output features that are relevant "
            "to the intevention explanation\n"
            "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
        )
