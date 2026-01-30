"""
Robosuite-specific feature selectors that use arm manipulation prompts.

These selectors extend the interact_drive selectors but override the system prompts
to use arm manipulation language instead of vehicle language.
"""

import numpy as np
import json
import re
import openai
from datetime import datetime

from robosuite_experiments.prompts import (
    MASKED_SYSTEM_PROMPT,
    MASKED_SYSTEM_PROMPT_DPHI,
    ADAPT_GATED_GATE_SYSTEM_PROMPT,
    ADAPT_GATED_MU_CONFIDENCE_SYSTEM_PROMPT,
    USER_PROMPT_HEADER,
)

# Global LLM log file path (set when logging is enabled)
_LLM_LOG_FILE = None

# Method name mapping for logging
METHOD_NAMES = {
    0: "PHRI (physical only)",
    1: "adapt_gated_llm (QuickLAP: physical + language)",
    2: "quicklap_language_only (language with feature changes)",
    3: "no_feature_context_language_only (language without feature changes)",
    -1: "NO_UPDATE",
    "masked": "masked (feature masking)",
    "masked_dphi": "masked_dphi (feature masking with delta phi)",
}

REASON_GATE_PROMPT = """
The GOAL of the transport is to get to the TARGET ZONE. That is, a feature should get gate=1.0 ONLY if the human's explanation suggests they INTENDED to change it,
NOT simply because it affects task completion. Feature changes might be SIDE EFFECTS of the human's intent.
For each feature, determine:\n
1. Did this change BECAUSE it was the human's INTENTIONAL (Implicit or Explicit) goal based on their explanation? → gate = 1.0\n
2. Did this change as a SIDE EFFECT of achieving another goal? → gate = 0.0\n
Note: The user does not need to explicitly state their intent, it can be implicit. Analyze the user utterance and their FEATURE change (if provided) to determine the most likely intent.\n
IMPORTANT: Even if a feature changed significantly, if it was a side effect (not the human's stated intent), it should get gate = 0.0.\n
Similarly, even if a feature changed slightly, if it was the human's intent (implicit or explicit), it should get gate = 1.0.
"""
# WEIGHT_UPDATE_PROMPT = {
#     "green_clearance": "Increasing this weight makes the robot stay farther from the green block.",
#     "velocity": "Increasing this weight makes the robot move faster.",
#     "collision_safety": "Increasing this weight makes the robot stay farther from obstacles.",
#     "joint_safety": "Increasing this weight makes the robot stay farther from joint limits.",
#     "block_to_target_zone": "Increasing this weight makes the robot move the block closer to the target zone.",
#     "zone_c_clearance": "Increasing this weight makes the robot stay farther from zone C.",
#     "height_maintain": "Increasing this weight makes the robot maintain the target transport height better.",
# }
WEIGHT_UPDATE_PROMPT = {
    "green_clearance": "POSITIVE mu (+) makes the robot stay FARTHER from the green block. NEGATIVE mu (-) lets the robot get CLOSER to the green block.",
    "velocity": "POSITIVE mu (+) makes the robot move FASTER. NEGATIVE mu (-) makes the robot move SLOWER.",
    "collision_safety": "POSITIVE mu (+) makes the robot more cautious around obstacles. NEGATIVE mu (-) allows closer proximity.",
    "joint_safety": "POSITIVE mu (+) makes the robot avoid joint limits more. NEGATIVE mu (-) allows more extreme joint positions.",
    "block_to_target_zone": "POSITIVE mu (+) prioritizes moving the block to the target zone. NEGATIVE mu (-) deprioritizes task completion.",
    "zone_c_clearance": "POSITIVE mu (+) makes the robot stay FARTHER from zone C. NEGATIVE mu (-) lets the robot get CLOSER to zone C.",
    "height_maintain": "POSITIVE mu (+) makes the robot maintain transport height better. NEGATIVE mu (-) allows more vertical drift.",
}
def _log_llm_call(
    system_prompt: str,
    user_prompt: str,
    response: str,
    model: str = "gpt-4o",
    method: int | None = None,
):
    """Log an LLM call to the global log file."""
    global _LLM_LOG_FILE
    if _LLM_LOG_FILE is None:
        return
    
    method_name = METHOD_NAMES.get(method, f"unknown (method={method})") if method is not None else "N/A"
    
    # Try to extract reasoning from JSON response
    reasoning = None
    try:
        parsed = json.loads(response)
        reasoning = parsed.get("reasoning")
    except (json.JSONDecodeError, TypeError):
        pass
    
    with open(_LLM_LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"{'='*80}\n")
        f.write(f"\n--- SYSTEM PROMPT ---\n{system_prompt}\n")
        f.write(f"\n--- USER PROMPT ---\n{user_prompt}\n")
        f.write(f"\n--- RESPONSE ---\n{response}\n")
        if reasoning:
            f.write(f"\n--- REASONING ---\n{reasoning}\n")


def log_weight_update(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    feature_names: list[str],
    method: int,
):
    """Log the final weight update to the global log file."""
    global _LLM_LOG_FILE
    if _LLM_LOG_FILE is None:
        return
    
    method_name = METHOD_NAMES.get(method, f"unknown (method={method})")
    
    with open(_LLM_LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"WEIGHT UPDATE - Method: {method_name}\n")
        f.write(f"{'='*80}\n")
        f.write(f"\n--- OLD WEIGHTS ---\n")
        for i, name in enumerate(feature_names):
            f.write(f"  {name}: {old_weights[i]:.4f}\n")
        f.write(f"\n--- NEW WEIGHTS ---\n")
        for i, name in enumerate(feature_names):
            change = new_weights[i] - old_weights[i]
            f.write(f"  {name}: {new_weights[i]:.4f} (change: {change:+.4f})\n")
        f.write(f"\n")


class RobosuiteMaskedLLMFeatureSelector:
    """Uses LLM to determine relevant features for interventions (arm manipulation domain)."""
    
    # Method identifier for logging (masked selector = feature masking approach)
    METHOD_ID = "masked"

    def __init__(
        self, feature_descriptions: dict[str, str], api_key: str | None, log_llm: bool = False
    ) -> None:
        """Initialize with OpenAI API key."""
        global _LLM_LOG_FILE
        self.feature_descriptions = feature_descriptions
        self.client = openai.OpenAI(api_key=api_key)
        self.log_llm = log_llm
        if log_llm and _LLM_LOG_FILE is None:
            _LLM_LOG_FILE = f"logs/llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
        system_prompt = self.construct_system_prompt()

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_text = response.choices[0].message.content
        assert response_text is not None
        
        if self.log_llm:
            _log_llm_call(system_prompt, prompt, response_text, "gpt-4o", method=self.METHOD_ID)
        
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
        return MASKED_SYSTEM_PROMPT

    def construct_prompt(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
    ) -> str:
        """Construct prompt for the LLM."""
        prompt = f"""
{USER_PROMPT_HEADER}
{explanation}

Current Feature Values:
"""
        for feature, value in robot_feature_values.items():
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Robot value: {value:.3f}, Human value: {human_feature_values[feature]:.3f}\n"
            )
        prompt += REASON_GATE_PROMPT
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt


class RobosuiteMaskedLLMFeatureSelectorDPhi(RobosuiteMaskedLLMFeatureSelector):
    """Feature selector using delta phi (change in features) with arm manipulation prompts."""
    
    # Override method identifier for logging
    METHOD_ID = "masked_dphi"
    
    def construct_prompt(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
    ) -> str:
        """Construct prompt for the LLM."""
        prompt = f"""
        {USER_PROMPT_HEADER}
        {explanation}

        Current Feature Values:
        """
        for feature, value in human_feature_values.items():
            change = value - robot_feature_values[feature]
 
            prompt += (
                f"- {feature} ({self.feature_descriptions[feature]}): "
                f"Feature change: {change:.3f}\n"
            )
        prompt += REASON_GATE_PROMPT
        prompt += (
            "\nWhich features from the list above are relevant to this intervention?"
        )
        return prompt

    def construct_system_prompt(self) -> str:
        return MASKED_SYSTEM_PROMPT_DPHI


class RobosuiteAdaptGatedLLMFeatureSelector:
    """Uses LLM to determine relevant features and desired changes (arm manipulation domain)."""

    def __init__(
        self,
        feature_descriptions: dict[str, str],
        api_key: str | None,
        main_model: str = "gpt-4o",
        gate_model: str = "gpt-4o",
        method: int = 1,
        log_llm: bool = False,
    ) -> None:
        """Initialize with OpenAI API key."""
        global _LLM_LOG_FILE
        self.client = openai.OpenAI(api_key=api_key)
        self.feature_descriptions = feature_descriptions
        self.main_model = main_model
        self.gate_model = gate_model
        self.log_llm = log_llm
        self.method = method
        if log_llm and _LLM_LOG_FILE is None:
            _LLM_LOG_FILE = f"logs/llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def select_relevant_features_and_directions(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
        current_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use LLM to determine which features are relevant and how they should change."""
        gate_prompt = self._construct_prompt(
            explanation, robot_feature_values, human_feature_values
        )
        gate_prompt += "\nFor absolutely EVERY feature above, determine:\n"
        gate_prompt += REASON_GATE_PROMPT
        gate_prompt += "Which features from the list above are relevant to this intervention? (gate score 0.0 or 1.0)\n"

        gate_json = self._call_llm(
            gate_prompt,
            model=self.gate_model,
            temperature=0.1,
            system_msg=ADAPT_GATED_GATE_SYSTEM_PROMPT,
            expect_keys=("gate",),
        )
        gate = np.asarray(gate_json["gate"], dtype=float)

        mc_prompt = self._construct_prompt(
            explanation, robot_feature_values, human_feature_values
        )
        ordered_feature_names = list(robot_feature_values.keys())

        mc_prompt += "\nCurrent Reward Weights before the Intervention (these are the weights before applying the 'mu' change you will suggest):\n"
        mc_prompt += f"IMPORTANT: READ THE FOLLOWING VERY CAREFULLY AND REASON THOROUGHLY BEFORE ANSWERING.\n"
        if len(current_weights) == len(ordered_feature_names):
            for i, feature_name in enumerate(ordered_feature_names):
                mc_prompt += f"- {feature_name}: {WEIGHT_UPDATE_PROMPT[feature_name]} Current weight: {current_weights[i]:.3f}\n"
        else:
            mc_prompt += "- Note: Could not display current weights due to a mismatch between number of features and weights provided.\n"

        mc_prompt += "\nNow, for absolutely EVERY feature (considering the explanation, feature changes, and current weights):\n"
        mc_prompt += "1. What change in weight with direction (this will be your 'mu', your mu will be added to the current weight) would support this intervention? Consider the scale of the features, and the current weights.\n"
        mc_prompt += (
            "2. How confident are you in your decision? (confidence score 0.0-1.0)\n"
        )

        mc_json = self._call_llm(
            mc_prompt,
            model=self.main_model,
            temperature=0.0,
            system_msg=ADAPT_GATED_MU_CONFIDENCE_SYSTEM_PROMPT,
            expect_keys=("mu", "confidence"),
        )
        
        print(f"LLM response for mu and confidence: {mc_json}")
        
        try:
            mu = np.asarray(mc_json["mu"], dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"LLM returned non-numeric values in 'mu' field. "
                f"Expected list of numbers, got: {mc_json['mu']}\n"
                f"Full response: {mc_json}"
            ) from e
        
        try:
            confidence = np.asarray(mc_json["confidence"], dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"LLM returned non-numeric values in 'confidence' field. "
                f"Expected list of numbers, got: {mc_json['confidence']}\n"
                f"Full response: {mc_json}"
            ) from e

        return gate, mu, confidence

    def _construct_prompt(
        self,
        explanation: str,
        robot_feature_values: dict[str, float],
        human_feature_values: dict[str, float],
    ) -> str:
        """Construct prompt for the LLM."""

        prompt = f"""
                {USER_PROMPT_HEADER}
                {explanation}

                The following are FEATURE VALUES AND NOT REWARD WEIGHT CHANGES.Current Feature Values:
                """
        for feature, value in robot_feature_values.items():
            if self.method != 3:
                human_val = human_feature_values[feature]
                change = human_val - value


                prompt += (
                    f"- {feature} ({self.feature_descriptions[feature]}): "
                    f"feature change after intervention: {change:+.3f}\n"
                )
            else:
                prompt += (
                    f"- {feature} ({self.feature_descriptions[feature]}): "
                    f"Robot feature value: {value:.3f}\n"
                )

        prompt += "\nPlease respond in json format."
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
        
        if self.log_llm:
            _log_llm_call(system_msg, prompt, content, model, method=self.method)
        
        if content.startswith("```"):
            content = "\n".join(
                line
                for line in content.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        content = re.sub(r"\+([0-9]+(\.[0-9]+)?)", r"\1", content)

        try:
            parsed = json.loads(content)
            missing = [k for k in expect_keys if k not in parsed]
            if missing:
                raise KeyError(f"Missing keys {missing}")
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output:\n{content}") from e


