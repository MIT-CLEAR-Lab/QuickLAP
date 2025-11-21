import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import dotenv

from experiments.cone_avoid import ConeAvoidExperiment
from experiments.puddle_avoid import PuddleAvoidExperiment
from experiments.cone_car_avoid import ConeCarAvoidExperiment
from experiments.cone_car_avoid_four import ConeCarAvoidExperimentFour
from interact_drive.learner.phri_learner import PHRILearner
from interact_drive.learner.masked_learner import MaskedLLMPHRILearner
from interact_drive.learner.adapt_gated_llm_learner import AdaptGatedLLMPHRILearner
from interact_drive.learner.oracle_learner import OracleLearner


dotenv.load_dotenv()

SEED_RANGE = 10000

ORACLE_WEIGHTS = {
    "cone_avoid": np.array([10.0, 2.5, 20.0, 40.0]),
    "puddle_avoid": np.array([5.0, 1.5, 10.0, 20.0, 1.0]),
    "cone_car_avoid": np.array([15.0, 2.5, 20.0, 40.0, 50.0, 3.0]),
    "cone_car_avoid_four": np.array([15.0, 2.5, 20.0, 40.0, 50.0, 3.0]),
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    type=int,
    default=1,
    help="Number of Times to Seed & Run Each Experiment (default: 1)",
)
parser.add_argument(
    "--notes",
    type=str,
    default="",
    help="Notes for the experiment, to be saved in the log",
)
experiment_args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY")
EXPERIMENTS = {
    "cone_avoid": ConeAvoidExperiment,
    "puddle_avoid": PuddleAvoidExperiment,
    "cone_car_avoid": ConeCarAvoidExperiment,
    "cone_car_avoid_four": ConeCarAvoidExperimentFour,
}
UTTERANCES = [
    ("Be careful.", "Be_careful"),
    ("Watch out for that thing.", "Watch_out_thing"),
    ("Stay away from that thing.", "Stay_away_thing"),
    ("Avoid the obstacle.", "Avoid_obstacle"),
    ("Stay away from construction zones.", "Stay_away_construction"),
    ("Steer clear of the cone.", "Steer_clear_cone"),
]
LEARNER_FACTORIES = {
    "naive": lambda car, utterance: PHRILearner(car),
    "masked_dphi": lambda car, utterance: MaskedLLMPHRILearner(
        car,
        utterance,
        car.get_feature_descriptions(),
        openai_api_key=api_key,
        selector="d_phi",
    ),
    "adapt_gated_llm": lambda car, utterance: AdaptGatedLLMPHRILearner(
        car, utterance, car.get_feature_descriptions(), openai_api_key=api_key
    ),
    "quicklap_language_only": lambda car, utterance: AdaptGatedLLMPHRILearner(
        car,
        utterance,
        car.get_feature_descriptions(),
        openai_api_key=api_key,
        method=2,
    ),
    "no_feature_context_language_only": lambda car, utterance: AdaptGatedLLMPHRILearner(
        car,
        utterance,
        car.get_feature_descriptions(),
        openai_api_key=api_key,
        method=3,
    ),
}


def normalize_weights(weights):
    """L2-normalizes a weight vector."""
    weights_arr = np.array(weights, dtype=float)  # Ensure float for division
    norm = np.linalg.norm(weights_arr)
    if norm == 0:
        # Return a zero vector of the same shape; avoids division by zero
        # and is meaningful for a zero input vector.
        return weights_arr
    return weights_arr / norm


def calculate_weights_mse(learned_theta, gt_theta):
    """Calculates ||normalized_learned_theta - normalized_GT_theta||^2."""
    if learned_theta is None:
        print("Warning: Learned theta is None, cannot calculate MSE.")
        return np.nan  # Or handle as appropriate
    if gt_theta is None:  # Should not happen with ORACLE_WEIGHTS
        print("Warning: Ground truth theta is None, cannot calculate MSE.")
        return np.nan

    norm_learned_theta = normalize_weights(learned_theta)
    norm_gt_theta = normalize_weights(gt_theta)

    if norm_learned_theta.shape != norm_gt_theta.shape:
        print(
            f"Warning: Normalized weight dimension mismatch. Learned: {norm_learned_theta.shape}, GT: {norm_gt_theta.shape}. Cannot calculate MSE."
        )
        return np.nan

    return np.sum((norm_learned_theta - norm_gt_theta) ** 2)


def calculate_regret(optimal_weights, oracle_features, learner_features):
    """

    Args:
        optimal_weights: The optimal weights vector (θ*)
        oracle_features: Feature vectors from optimal trajectory (Φ(ξθ*))
        learner_features: Feature vectors from learner trajectory (Φ(ξact))

    Returns:
        float: The regret value
    """
    # Make sure features are numpy arrays
    if not isinstance(oracle_features, np.ndarray):
        oracle_features = np.array(oracle_features)
    if not isinstance(learner_features, np.ndarray):
        learner_features = np.array(learner_features)

    # Sum feature vectors across time steps
    oracle_sum = np.sum(oracle_features, axis=0)
    learner_sum = np.sum(learner_features, axis=0)

    # Calculate dot products
    oracle_value = np.dot(optimal_weights, oracle_sum)
    learner_value = np.dot(optimal_weights, learner_sum)

    # Calculate regret
    regret = oracle_value - learner_value

    return regret


if experiment_args.n > 1:
    # Seed experiments if running many for reproducibility
    np.random.seed(859)
experiment_seeds = np.random.randint(0, int(1e9), experiment_args.n).tolist()
results = defaultdict(lambda: defaultdict(list))
regrets = defaultdict(lambda: defaultdict(list))
oracle_data = defaultdict(lambda: defaultdict(dict))
weights_mse_results = defaultdict(lambda: defaultdict(list))
learned_weights_raw_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# First run Oracle experiments to establish optimal performance
for world_name, world_cls in EXPERIMENTS.items():
    print(f"\nRunning Oracle for {world_name}...")
    oracle_factory = lambda car: OracleLearner(
        car, custom_weights=ORACLE_WEIGHTS[world_name]
    )

    for seed in experiment_seeds:
        print(f"  Seed {seed}...")
        experiment = world_cls(exp_name=f"{world_name}_oracle")

        reward, feature_trajectory, learned_weights_for_run = experiment.run(
            oracle_factory, int(seed)
        )

        # Store results
        results[world_name]["oracle"].append(reward)

        # Store oracle data for regret calculation
        oracle_data[world_name][seed] = {
            "features": feature_trajectory,
            "weights": ORACLE_WEIGHTS[world_name],
        }
        print(f"  Stored Oracle data with {len(feature_trajectory)} feature vectors")

for world_name, world_cls in EXPERIMENTS.items():
    gt_weights_for_world = ORACLE_WEIGHTS[world_name]
    for learner_name, learner_factory in LEARNER_FACTORIES.items():
        for utterance_text, utterance_short_name in UTTERANCES:
            current_run_learner_name = f"{learner_name}_{utterance_short_name}"
            print(
                f"\nRunning {current_run_learner_name} for {world_name} with utterance: '{utterance_text}'..."
            )

            for seed in experiment_seeds:
                print(f"  Seed {seed}...")
                experiment = world_cls(
                    exp_name=f"{world_name}_{current_run_learner_name}"
                )

                reward, feature_trajectory, learned_weights_for_run = experiment.run(
                    lambda car: learner_factory(car, utterance_text), int(seed)
                )

                # Store results
                results[world_name][current_run_learner_name].append(reward)
                regrets[world_name][current_run_learner_name].append(
                    calculate_regret(
                        gt_weights_for_world,
                        oracle_data[world_name][seed]["features"],
                        feature_trajectory,
                    )
                )

                if seed in oracle_data[world_name]:
                    oracle_info = oracle_data[world_name][seed]

                    # Get Oracle weights and features
                    oracle_weights = oracle_info["weights"]
                    oracle_features = oracle_info["features"]

                    # Adjust feature trajectories to same length if needed
                    min_length = min(len(oracle_features), len(feature_trajectory))
                    print(
                        f"Lengths: {len(oracle_features)} (Oracle) vs {len(feature_trajectory)} (Learner)"
                    )

                    if learned_weights_for_run is not None:
                        mse = calculate_weights_mse(
                            learned_weights_for_run, gt_weights_for_world
                        )
                        if not np.isnan(mse):
                            weights_mse_results[world_name][
                                current_run_learner_name
                            ].append(mse)
                            print(f"  Calculated weights MSE: {mse:.4f}")
                        else:
                            print(
                                f"  Weights MSE calculation failed or resulted in NaN for {current_run_learner_name}, seed {seed}."
                            )
                        # Store raw learned weights (as list for JSON)
                        learned_weights_raw_log[world_name][current_run_learner_name][
                            str(seed)
                        ] = np.array(learned_weights_for_run).tolist()
                    else:
                        print(
                            f"  No learned weights returned for {current_run_learner_name}, seed {seed}. Cannot calculate MSE."
                        )

                else:
                    print(
                        f"  Warning: No Oracle data for {world_name}, seed {seed}, learner {learner_name}"
                    )

results_dict = {x: dict(results[x]) for x in results.keys()}
regrets_dict = {x: dict(regrets[x]) for x in regrets.keys()}
weights_mse_dict = {x: dict(weights_mse_results[x]) for x in weights_mse_results.keys()}
learned_weights_raw_dict = {
    world: {
        learner: dict(seeds_weights) for learner, seeds_weights in learner_runs.items()
    }
    for world, learner_runs in learned_weights_raw_log.items()
}

log_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_text = f"""
Experiment Date: {log_date}

Notes: {experiment_args.notes}

Oracle Weights Used:
Cone Avoid: {ORACLE_WEIGHTS["cone_avoid"]}
Puddle Avoid: {ORACLE_WEIGHTS["puddle_avoid"]}
Cone Car Avoid: {ORACLE_WEIGHTS["cone_car_avoid"]}
Cone Car Avoid Four: {ORACLE_WEIGHTS["cone_car_avoid_four"]}

Raw Rewards: {json.dumps(results_dict, indent=2)}
Raw Regrets: {json.dumps(regrets_dict, indent=2)}
Raw Weights MSE: {json.dumps(weights_mse_dict, indent=2)}
Raw Learned Weights (sample, see JSON for all): {json.dumps(learned_weights_raw_dict, indent=2)}
"""

# Print reward results
log_text += "\n\n=== REWARD RESULTS ===\n"
for world_name, world_results in results.items():
    log_text += f"\n{world_name}:\n"
    sorted_learner_names = sorted(world_results.keys())
    for learner_name in sorted_learner_names:
        learner_results = world_results[learner_name]
        mean_result = np.mean(learner_results)
        std_result = np.std(learner_results)
        log_text += f"{learner_name:{' '}<45}: {mean_result:.4f} ± {std_result:.4f}\n"

log_text += "\n\n=== REGRET RESULTS (R(ξθ*, ξact)) ===\n"
for world_name, world_regrets in regrets.items():
    log_text += f"\n{world_name}:\n"
    sorted_learner_names = sorted(world_regrets.keys())
    for learner_name in sorted_learner_names:
        learner_regrets = world_regrets[learner_name]
        mean_regret = np.mean(learner_regrets)
        std_regret = np.std(learner_regrets)
        log_text += f"{learner_name:{' '}<45}: {mean_regret:.4f} ± {std_regret:.4f}\n"

log_text += "\n\n=== NORMALIZED WEIGHTS MSE (||norm_θ_learned - norm_θ_GT||^2) ===\n"
for world_name, world_mses in weights_mse_results.items():
    log_text += f"\n{world_name}:\n"
    sorted_learner_names = sorted(world_mses.keys())
    for learner_name in sorted_learner_names:
        learner_mses_list = world_mses[learner_name]
        if learner_mses_list:  # Check if list is not empty
            mean_mse = np.mean(learner_mses_list)
            std_mse = np.std(learner_mses_list)
            log_text += f"{learner_name:{' '}<45}: {mean_mse:.4f} ± {std_mse:.4f}\n"
        else:
            log_text += f"{learner_name:{' '}<45}: No MSE values\n"

print(log_text)
os.makedirs(f"logs/{log_date}", exist_ok=True)
with open(f"logs/{log_date}/report.txt", "w") as f:  # Changed name to generic report
    f.write(log_text)

# Prepare JSON output
json_output = {
    "experiment_info": {
        "date": log_date,
        "notes": experiment_args.notes,
        "num_seeds_per_run": experiment_args.n,
        "seeds_used": experiment_seeds,
    },
    "oracle_weights_GT": {k: v.tolist() for k, v in ORACLE_WEIGHTS.items()},
    "results_reward": results_dict,
    "results_regret": regrets_dict,
    "results_weights_mse": weights_mse_dict,
    "learned_weights_raw": learned_weights_raw_dict,  # Store all learned weights
}

with open(f"logs/{log_date}/results.json", "w") as f:
    json.dump(json_output, f, indent=2)

print(f"\nResults saved to logs/{log_date}/")
