"""
Main script for running robosuite experiments with all learning methods.

Mirrors the structure of experiments/run_experiments.py for consistency.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import sys
import os
import numpy as np
import dotenv

# Add parent directory to path so we can import from interact_drive
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pick_place_experiment import PickPlaceExperiment
from robosuite_phri_learner import RobosuitePHRILearner
from robosuite_learners import (
    RobosuiteMaskedLLMPHRILearner,
    RobosuiteAdaptGatedLLMPHRILearner,
)
from interact_drive.learner.oracle_learner import OracleLearner


# Load environment variables for OpenAI API
dotenv.load_dotenv()

# Oracle (expert) weights for the pick-and-place task
# [green_clearance, velocity, collision, joints, block_to_zone, zone_c_clearance, height_maintain]
# Note: clearance features use POSITIVE weights (higher = stay farther = safer)
ORACLE_WEIGHTS = {
    "pick_place": np.array([10.0, 1.0, 2.0, 1.0, 25.0, 2.0, 0.0]),
}

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    type=int,
    default=1,
    help="Number of times to seed & run each experiment (default: 1)",
)
parser.add_argument(
    "--notes",
    type=str,
    default="",
    help="Notes for the experiment, to be saved in the log",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=1200,
    help="Episode length in timesteps (default: 1200)",
)
parser.add_argument(
    "--physical-input",
    action="store_true",
    help="Enable keyboard input for human intervention",
)
parser.add_argument(
    "--log-llm",
    action="store_true",
    help="Log LLM inputs and outputs for every experiment",
)
args = parser.parse_args()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Define experiments
EXPERIMENTS = {
    "pick_place": PickPlaceExperiment,
}

# Define utterances to test
UTTERANCES = [
    # ("Avoid the obstacle.", "Avoid_obstacle"),
    ("Steer clear of there.", "Steer_clear"),
    # ("Move away!!", "Move_away"),
    # ("AHH!", "AHH"),
    # ("Go left", "Go_left"),
    # ("Get away", "Get_away"),
]

# Define learner factories (using robosuite-compatible versions)
def get_learner_factories(log_llm=False):
    return {
        "naive": lambda arm, utterance: RobosuitePHRILearner(
            arm, log_file=f"logs/robosuite_naive_{utterance}.txt"
        ),
        "masked_dphi": lambda arm, utterance: RobosuiteMaskedLLMPHRILearner(
            arm,
            utterance,
            arm.get_feature_descriptions(),
            openai_api_key=api_key,
            selector="d_phi",
            log_llm=log_llm,
        ),
        "adapt_gated_llm": lambda arm, utterance: RobosuiteAdaptGatedLLMPHRILearner(
            arm, utterance, arm.get_feature_descriptions(), openai_api_key=api_key,
            log_llm=log_llm,
        ),
        "quicklap_language_only": lambda arm, utterance: RobosuiteAdaptGatedLLMPHRILearner(
            arm,
            utterance,
            arm.get_feature_descriptions(),
            openai_api_key=api_key,
            method=2,
            log_llm=log_llm,
        ),
        "no_feature_context_language_only": lambda arm, utterance: RobosuiteAdaptGatedLLMPHRILearner(
            arm,
            utterance,
            arm.get_feature_descriptions(),
            openai_api_key=api_key,
            method=3,
            log_llm=log_llm,
        ),
    }


def normalize_weights(weights):
    """L2-normalize a weight vector."""
    weights_arr = np.array(weights, dtype=float)
    norm = np.linalg.norm(weights_arr)
    if norm == 0:
        return weights_arr
    return weights_arr / norm


def calculate_weights_mse(learned_theta, gt_theta):
    """Calculate ||normalized_learned_theta - normalized_GT_theta||^2."""
    if learned_theta is None:
        print("Warning: Learned theta is None, cannot calculate MSE.")
        return np.nan
    if gt_theta is None:
        print("Warning: Ground truth theta is None, cannot calculate MSE.")
        return np.nan
    
    norm_learned_theta = normalize_weights(learned_theta)
    norm_gt_theta = normalize_weights(gt_theta)
    
    if norm_learned_theta.shape != norm_gt_theta.shape:
        print(
            f"Warning: Normalized weight dimension mismatch. "
            f"Learned: {norm_learned_theta.shape}, GT: {norm_gt_theta.shape}. "
            f"Cannot calculate MSE."
        )
        return np.nan
    
    return np.sum((norm_learned_theta - norm_gt_theta) ** 2)


def calculate_regret(optimal_weights, oracle_features, learner_features):
    """
    Calculate regret: difference in value between oracle and learner trajectories.
    
    Args:
        optimal_weights: The optimal weights vector (θ*)
        oracle_features: Feature vectors from optimal trajectory
        learner_features: Feature vectors from learner trajectory
        
    Returns:
        float: The regret value
    """
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


# Generate random seeds for experiments
if args.n > 1:
    np.random.seed(859)  # Fixed seed for reproducibility
experiment_seeds = np.random.randint(0, int(1e9), args.n).tolist()

# Storage for results
results = defaultdict(lambda: defaultdict(list))
regrets = defaultdict(lambda: defaultdict(list))
oracle_data = defaultdict(lambda: defaultdict(dict))
weights_mse_results = defaultdict(lambda: defaultdict(list))
learned_weights_raw_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# First, run Oracle experiments to establish optimal performance
print("\n" + "="*80)
print("RUNNING ORACLE EXPERIMENTS")
print("="*80)

for world_name, world_cls in EXPERIMENTS.items():
    print(f"\nRunning Oracle for {world_name}...")
    oracle_factory = lambda arm: OracleLearner(
        arm, custom_weights=ORACLE_WEIGHTS[world_name]
    )
    
    for seed in experiment_seeds:
        print(f"  Seed {seed}...")
        experiment = world_cls(
            exp_name=f"{world_name}_oracle",
            use_physical_input=args.physical_input,
            horizon=args.horizon,
        )
        
        reward, feature_trajectory, learned_weights = experiment.run(
            oracle_factory, int(seed)
        )
        
        # Store results
        results[world_name]["oracle"].append(reward)
        
        # Store oracle data for regret calculation
        oracle_data[world_name][seed] = {
            "features": feature_trajectory,
            "weights": ORACLE_WEIGHTS[world_name],
        }
        print(f"  Oracle reward: {reward:.2f}")
        print(f"  Stored {len(feature_trajectory)} feature vectors")

# Now run experiments with all learners and utterances
print("\n" + "="*80)
print("RUNNING LEARNER EXPERIMENTS")
print("="*80)

LEARNER_FACTORIES = get_learner_factories(log_llm=args.log_llm)

for world_name, world_cls in EXPERIMENTS.items():
    gt_weights = ORACLE_WEIGHTS[world_name]
    
    for learner_name, learner_factory in LEARNER_FACTORIES.items():
        for utterance_text, utterance_short_name in UTTERANCES:
            run_name = f"{learner_name}_{utterance_short_name}"
            print(f"\nRunning {run_name} for {world_name}")
            print(f"  Utterance: '{utterance_text}'")
            
            for seed in experiment_seeds:
                print(f"  Seed {seed}...")
                experiment = world_cls(
                    exp_name=f"{world_name}_{run_name}",
                    use_physical_input=args.physical_input,
                    horizon=args.horizon,
                )
                
                reward, feature_trajectory, learned_weights = experiment.run(
                    lambda arm: learner_factory(arm, utterance_text),
                    int(seed),
                    utterance=utterance_text,
                )
                
                # Store results
                results[world_name][run_name].append(reward)
                
                # Calculate regret
                if seed in oracle_data[world_name]:
                    oracle_features = oracle_data[world_name][seed]["features"]
                    regret_value = calculate_regret(
                        gt_weights, oracle_features, feature_trajectory
                    )
                    regrets[world_name][run_name].append(regret_value)
                    print(f"    Reward: {reward:.2f}, Regret: {regret_value:.2f}")
                
                # Calculate weights MSE
                if learned_weights is not None:
                    mse = calculate_weights_mse(learned_weights, gt_weights)
                    if not np.isnan(mse):
                        weights_mse_results[world_name][run_name].append(mse)
                        print(f"    Weights MSE: {mse:.4f}")
                    
                    # Store raw learned weights
                    learned_weights_raw_log[world_name][run_name][str(seed)] = (
                        learned_weights.tolist()
                    )
                else:
                    print(f"    No learned weights returned")

# Convert defaultdicts to regular dicts for JSON serialization
results_dict = {x: dict(results[x]) for x in results.keys()}
regrets_dict = {x: dict(regrets[x]) for x in regrets.keys()}
weights_mse_dict = {x: dict(weights_mse_results[x]) for x in weights_mse_results.keys()}
learned_weights_raw_dict = {
    world: {
        learner: dict(seeds_weights)
        for learner, seeds_weights in learner_runs.items()
    }
    for world, learner_runs in learned_weights_raw_log.items()
}

# Generate report
log_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_text = f"""
Robosuite Experiment Report
Date: {log_date}

Notes: {args.notes}

Oracle Weights Used:
Pick-Place: {ORACLE_WEIGHTS["pick_place"]}

Number of seeds: {args.n}
Seeds used: {experiment_seeds}

Raw Rewards: {json.dumps(results_dict, indent=2)}
Raw Regrets: {json.dumps(regrets_dict, indent=2)}
Raw Weights MSE: {json.dumps(weights_mse_dict, indent=2)}
"""

# Print reward results
log_text += "\n\n=== REWARD RESULTS ===\n"
for world_name, world_results in results.items():
    log_text += f"\n{world_name}:\n"
    for learner_name in sorted(world_results.keys()):
        learner_results = world_results[learner_name]
        mean_result = np.mean(learner_results)
        std_result = np.std(learner_results)
        log_text += f"  {learner_name:<50}: {mean_result:.4f} ± {std_result:.4f}\n"

# Print regret results
log_text += "\n\n=== REGRET RESULTS (R(ξθ*, ξact)) ===\n"
for world_name, world_regrets in regrets.items():
    log_text += f"\n{world_name}:\n"
    for learner_name in sorted(world_regrets.keys()):
        learner_regrets = world_regrets[learner_name]
        mean_regret = np.mean(learner_regrets)
        std_regret = np.std(learner_regrets)
        log_text += f"  {learner_name:<50}: {mean_regret:.4f} ± {std_regret:.4f}\n"

# Print weights MSE results
log_text += "\n\n=== NORMALIZED WEIGHTS MSE (||norm_θ_learned - norm_θ_GT||^2) ===\n"
for world_name, world_mses in weights_mse_results.items():
    log_text += f"\n{world_name}:\n"
    for learner_name in sorted(world_mses.keys()):
        learner_mses = world_mses[learner_name]
        if learner_mses:
            mean_mse = np.mean(learner_mses)
            std_mse = np.std(learner_mses)
            log_text += f"  {learner_name:<50}: {mean_mse:.4f} ± {std_mse:.4f}\n"
        else:
            log_text += f"  {learner_name:<50}: No MSE values\n"

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(log_text)

# Save results
os.makedirs(f"logs/robosuite_{log_date}", exist_ok=True)

with open(f"logs/robosuite_{log_date}/report.txt", "w") as f:
    f.write(log_text)

# Save JSON output
json_output = {
    "experiment_info": {
        "date": log_date,
        "notes": args.notes,
        "num_seeds_per_run": args.n,
        "seeds_used": experiment_seeds,
    },
    "oracle_weights_GT": {k: v.tolist() for k, v in ORACLE_WEIGHTS.items()},
    "results_reward": results_dict,
    "results_regret": regrets_dict,
    "results_weights_mse": weights_mse_dict,
    "learned_weights_raw": learned_weights_raw_dict,
}

with open(f"logs/robosuite_{log_date}/results.json", "w") as f:
    json.dump(json_output, f, indent=2)

print(f"\nResults saved to logs/robosuite_{log_date}/")

