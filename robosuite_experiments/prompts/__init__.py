"""
Centralized prompts for robosuite arm manipulation experiments.

These prompts are adapted from the vehicle domain to the arm manipulation domain.
Only domain-specific words have been changed.
"""

# =============================================================================
# MASKED SELECTOR PROMPTS
# =============================================================================

MASKED_SYSTEM_PROMPT = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine which features are relevant to a given intervention explanation, given the "
    "current feature values of the robot and human trajectories.\n"
    "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
)

MASKED_SYSTEM_PROMPT_PHI_H = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine which features are relevant to a given intervention explanation, given the "
    "current feature values of the human trajectory.\n"
    "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
)

MASKED_SYSTEM_PROMPT_PHI_R = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine which features are relevant to a given intervention explanation, given the "
    "current feature values of the robot trajectory.\n"
    "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
)

MASKED_SYSTEM_PROMPT_DPHI = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine which features are relevant to a given intervention explanation, given the "
    "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
    "that the human increased the feature value.\n"
    "The GOAL of the transport is to get to the TARGET ZONE."
    "IMPORTANT: Feature values are CUMULATIVE SUMS over the trajectory (~150 timesteps), not per-timestep values. "
    "A change of +10 means the human's trajectory accumulated 10 more units of that feature over the episode.\n"
    "Note that a feature may be irrelevant even if it has a large change in value. Only output features that are relevant "
    "to the intevention explanation\n"
    "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
)

MASKED_SYSTEM_PROMPT_DPHI_SIGN = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine which features are relevant to a given intervention explanation, given the "
    "change in feature values of the human trajectory compared to the robot trajectory.\n"
    "Note that a feature may be irrelevant even if it has a large change in value. Only output features that are relevant "
    "to the intevention explanation\n"
    "Respond with a single line of text, containing a list of relevant feature names, separated by commas."
)

# =============================================================================
# ADAPT GATED LLM SELECTOR PROMPTS
# =============================================================================

ADAPT_GATED_GATE_SYSTEM_PROMPT = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine which features are relevant to a given intervention explanation, given the "
    "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
    "that the human increased the feature value.\n"
    "The GOAL of the transport is to get to the TARGET ZONE."
    "Note that a feature may be irrelevant even if it has a large change in value. Only output features that are relevant.\n"
    "YOUR JOB IS TO DETERMINE WHICH FEATURES THE HUMAN INTENTIONALLY WANTED TO CHANGE, NOT WHICH FEATURES CHANGED.\n"
    "CRITICAL: The human's VERBAL EXPLANATION is the primary signal for determining intent. "
    "If the physical feature changes contradict the user's utterance, you should prioritize the utterance.\n"
    "Focus on the human's EXPLANATION to determine their true intent, NOT ONLY which features changed.\n"
    "Output STRICT JSON with keys 'gate' and 'reasoning'. "
    "'gate': a list of attention gate scores (one per feature, 0.0 or 1.0). "
    "'reasoning': for each feature, explain whether it was causal (gate=1.0) or a side effect (gate=0.0)."
)

# ADAPT_GATED_MU_CONFIDENCE_SYSTEM_PROMPT = (
#     "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
#     "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
#     "Your reward function is the sum of the features weighted by their weights. You want to maximize the reward function. "
#     "Your task is to determine for EACH feature how much in magnitude should the weight of the feature be changed to support the intervention and human preference (mu between -6 and 6), and how confident you are in your decision (confidence between 0 and 1, be conservative), given the "
#     "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
#     "that the human increased the feature value.\n"
#     "YOUR JOB IS TO STRESS THE REWARD WEIGHT CHANGE ON FEATURES THE HUMAN INTENTIONALLY WANTED TO CHANGE, NOT ALL FEATURES.\n"
#     "CRITICAL: When the human's VERBAL EXPLANATION conflicts with the physical feature changes, ALWAYS prioritize the verbal explanation. "
#     "For example, if the user says 'do X' but the feature for X contradicts the user's utterance during the intervention, "
#     "you should follow the user's utterance.\n"
#     "REMEMBER: mu is ADDED to the current weight. If you want to INCREASE a weight, use POSITIVE mu. If you want to DECREASE a weight, use NEGATIVE mu. "
#     "CRITICAL: 'mu' and 'confidence' must be ONLY NUMERIC VALUES. Each 'mu' value must be a NUMBER between -6 and 6. Each 'confidence' value must be a NUMBER between 0 and 1. "
#     "DO NOT return feature names, strings, or any text in those arrays. ONLY numbers.\n\n"
#     "Example JSON output for 3 features:\n"
#     '{"mu": [1.5, -0.5, 2.0], "confidence": [0.8, 0.6, 0.9], "reasoning": "Increased block_a_height suggests..."}\n\n'
#     "FOR ABSOLUTELY EVERY FEATURE listed in the prompt, provide ONE numeric mu value and ONE numeric confidence value. "
#     "The arrays must be in the same order as the features in the prompt. "
#     "Also include a 'reasoning' key with a brief string explaining your decisions."
# )

ADAPT_GATED_MU_CONFIDENCE_SYSTEM_PROMPT = (
    "You are an expert in robotic arm control analyzing human interventions. In this task, a human "
    "has intervened to correct the behavior of a robot arm and has provided an explanation of the intervention. "
    "Your task is to determine for EACH feature how much in magnitude should the weight of the feature be changed to support the intervention and human preference (mu between -6 and 6), and how confident you are in your decision (confidence between 0 and 1, be conservative), given the "
    "change in feature values of the human trajectory compared to the robot trajectory. Positive values mean "
    "that the human increased the feature value.\n"
    "YOUR JOB IS TO STRESS THE REWARD WEIGHT CHANGE ON FEATURES THE HUMAN INTENTIONALLY WANTED TO CHANGE, NOT ALL FEATURES.\n"
    "CRITICAL: When the human's VERBAL EXPLANATION is explicit and conflicts with the physical feature changes, ALWAYS prioritize the verbal explanation. "
    "The physical feature changes may be SIDE EFFECTS or ACCIDENTS - trust what the human SAID, not what happened physically.\n"
    "REMEMBER: mu is ADDED to the current weight. If you want to INCREASE a weight, use POSITIVE mu. If you want to DECREASE a weight, use NEGATIVE mu.\n"
    "TO DETERMINE MU SIGN: (1) Read the user's verbal intent. (2) Read the FEATURE DESCRIPTION to understand what increasing/decreasing the weight does. "
    "(3) Choose the mu sign that achieves the user's stated goal.\n"
    "CRITICAL: 'mu' and 'confidence' must be ONLY NUMERIC VALUES. Each 'mu' value must be a NUMBER between -6 and 6. Each 'confidence' value must be a NUMBER between 0 and 1. "
    "DO NOT return feature names, strings, or any text in those arrays. ONLY numbers.\n\n"
    "Example JSON output for 3 features:\n"
    '{"mu": [1.5, -0.5, 2.0], "confidence": [0.8, 0.6, 0.9], "reasoning": "Increased block_a_height suggests..."}\n\n'
    "FOR ABSOLUTELY EVERY FEATURE listed in the prompt, provide ONE numeric mu value and ONE numeric confidence value. "
    "The arrays must be in the same order as the features in the prompt. "
    "Also include a 'reasoning' key with a brief string explaining your decisions."
    "VERIFY: Before outputting, check that your mu SIGN matches your reasoning. "
    "If your reasoning says 'increase the weight', mu must be POSITIVE."
    "If your reasoning says 'decrease the weight', mu must be NEGATIVE.\n"
)

# =============================================================================
# USER PROMPT TEMPLATES
# =============================================================================

USER_PROMPT_HEADER = "Human Intervention Explanation:"


