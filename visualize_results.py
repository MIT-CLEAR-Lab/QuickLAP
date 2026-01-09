#!/usr/bin/env python3
"""
Visualization script for QuickLAP experimental results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_method_groups(results_dict):
    """Group methods by type (oracle, naive, masked_dphi, quicklap, quicklap_language_only, no_feature_context)."""
    groups = {
        'oracle': [],
        'naive': [],
        'masked_dphi': [],
        'quicklap': [],
        'quicklap_language_only': [],
        'no_feature_context_language_only': []
    }
    
    for method_name in results_dict.keys():
        if method_name == 'oracle':
            groups['oracle'].append(method_name)
        elif method_name.startswith('naive_'):
            groups['naive'].append(method_name)
        elif method_name.startswith('masked_dphi_'):
            groups['masked_dphi'].append(method_name)
        elif method_name.startswith('adapt_gated_llm_'):
            groups['quicklap'].append(method_name)
        elif method_name.startswith('quicklap_language_only_'):
            groups['quicklap_language_only'].append(method_name)
        elif method_name.startswith('no_feature_context_language_only_'):
            groups['no_feature_context_language_only'].append(method_name)
    
    return groups


def plot_metric_comparison(data, metric_name, environments, output_dir):
    """Create bar plots comparing different methods across environments."""
    n_envs = len(environments)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    colors = {
        'oracle': '#2ecc71',
        'naive': '#e74c3c',
        'masked_dphi': '#f39c12',
        'quicklap': '#3498db',
        'quicklap_language_only': '#9b59b6',
        'no_feature_context_language_only': '#1abc9c'
    }
    
    for idx, env in enumerate(environments):
        ax = axes[idx]
        env_data = data[env]
        
        # Group methods
        method_groups = extract_method_groups(env_data)
        
        # Prepare data for plotting
        method_means = {}
        method_stds = {}
        
        for group_name, methods in method_groups.items():
            if not methods:
                continue
            
            values = []
            for method in methods:
                if method in env_data:
                    values.extend(env_data[method])
            
            if values:
                method_means[group_name] = np.mean(values)
                method_stds[group_name] = np.std(values) if len(values) > 1 else 0
        
        # Sort by mean value for better visualization (except oracle first)
        sorted_methods = ['oracle'] if 'oracle' in method_means else []
        other_methods = sorted(
            [k for k in method_means.keys() if k != 'oracle'],
            key=lambda k: method_means[k],
            reverse=(metric_name == 'Reward')  # Higher is better for reward, lower for regret/MSE
        )
        sorted_methods.extend(other_methods)
        
        # Plot
        x_pos = np.arange(len(sorted_methods))
        means = [method_means[m] for m in sorted_methods]
        stds = [method_stds[m] for m in sorted_methods]
        bar_colors = [colors.get(m, '#95a5a6') for m in sorted_methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Formatting
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{env.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', '\n') for m in sorted_methods], rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'{metric_name} Comparison Across Environments', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / f'{metric_name.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prompt_sensitivity(data, metric_name, environments, output_dir):
    """Plot sensitivity to different prompts for each method group."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    method_types = ['naive', 'masked_dphi', 'quicklap', 'quicklap_language_only']
    method_type_mapping = {'adapt_gated_llm': 'quicklap'}  # Map old name to new display name
    
    for idx, env in enumerate(environments):
        ax = axes[idx]
        env_data = data[env]
        
        prompt_results = {}
        
        for method_name, values in env_data.items():
            if method_name == 'oracle':
                continue
            
            # Extract prompt name and map method types
            if method_name.startswith('adapt_gated_llm_'):
                prompt = method_name.replace('adapt_gated_llm_', '')
                mtype = 'quicklap'
            elif method_name.startswith('naive_'):
                prompt = method_name.replace('naive_', '')
                mtype = 'naive'
            elif method_name.startswith('masked_dphi_'):
                prompt = method_name.replace('masked_dphi_', '')
                mtype = 'masked_dphi'
            elif method_name.startswith('quicklap_language_only_'):
                prompt = method_name.replace('quicklap_language_only_', '')
                mtype = 'quicklap_language_only'
            else:
                continue
            
            if prompt not in prompt_results:
                prompt_results[prompt] = {mt: [] for mt in method_types}
            prompt_results[prompt][mtype] = np.mean(values)
        
        # Plot grouped bars
        prompts = list(prompt_results.keys())
        x = np.arange(len(prompts))
        width = 0.2
        
        colors_list = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
        
        for i, mtype in enumerate(method_types):
            values = [prompt_results[p].get(mtype, 0) for p in prompts]
            ax.bar(x + i*width, values, width, label=mtype.replace('_', ' '), 
                  color=colors_list[i], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Prompt', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{env.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([p.replace('_', ' ') for p in prompts], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{metric_name} - Prompt Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / f'{metric_name.lower().replace(" ", "_")}_prompt_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weights_comparison(learned_weights, oracle_weights, environments, output_dir):
    """Compare learned weights vs oracle weights."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, env in enumerate(environments):
        ax = axes[idx]
        oracle = oracle_weights[env]
        n_weights = len(oracle)
        
        # Get best performing methods from each category
        best_methods = {}
        for method_name, seed_dict in learned_weights[env].items():
            if 'quicklap_language_only' in method_name or 'adapt_gated_llm' in method_name or 'no_feature_context_language_only' in method_name:
                weights = list(seed_dict.values())[0]  # Get first seed
                # Map adapt_gated_llm to quicklap for display
                if 'adapt_gated_llm' in method_name:
                    method_type = 'quicklap'
                elif 'no_feature_context_language_only' in method_name:
                    method_type = 'no_feature_context'
                else:
                    method_type = 'quicklap_language_only'
                
                if method_type not in best_methods:
                    best_methods[method_type] = (method_name, weights)
        
        # Plot
        x = np.arange(n_weights)
        width = 0.2
        
        ax.bar(x - 1.5*width, oracle, width, label='Oracle (GT)', color='#2ecc71', alpha=0.8, edgecolor='black')
        
        colors_methods = ['#3498db', '#9b59b6', '#1abc9c']
        method_order = ['quicklap', 'quicklap_language_only', 'no_feature_context']
        for i, mtype in enumerate(method_order):
            if mtype in best_methods:
                mname, weights = best_methods[mtype]
                ax.bar(x + (i - 0.5)*width, weights, width, 
                      label=f'{mtype.replace("_", " ")}', 
                      color=colors_methods[i], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Weight Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{env.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'w{i}' for i in range(n_weights)])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Learned Weights vs Oracle Weights', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'weights_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(results, environments, output_dir):
    """Create a summary table with best results."""
    summary = []
    
    for env in environments:
        print(f"\n{'='*80}")
        print(f"Environment: {env.upper()}")
        print(f"{'='*80}")
        
        # Reward
        rewards = results['results_reward'][env]
        oracle_reward = rewards.get('oracle', [0])[0]
        
        print(f"\nOracle Reward: {oracle_reward:.2f}")
        print(f"\nTop 5 Methods by Reward:")
        sorted_rewards = sorted(
            [(k, np.mean(v)) for k, v in rewards.items() if k != 'oracle'],
            key=lambda x: x[1],
            reverse=True
        )
        for i, (method, reward) in enumerate(sorted_rewards[:5], 1):
            gap = oracle_reward - reward
            print(f"  {i}. {method}: {reward:.2f} (gap: {gap:.2f})")
        
        # Regret
        if 'results_regret' in results:
            regrets = results['results_regret'][env]
            print(f"\nTop 5 Methods by Regret (lower is better):")
            sorted_regrets = sorted(
                [(k, np.mean(v)) for k, v in regrets.items()],
                key=lambda x: x[1]
            )
            for i, (method, regret) in enumerate(sorted_regrets[:5], 1):
                print(f"  {i}. {method}: {regret:.2f}")
        
        # MSE
        if 'results_weights_mse' in results:
            mses = results['results_weights_mse'][env]
            print(f"\nTop 5 Methods by Weights MSE (lower is better):")
            sorted_mses = sorted(
                [(k, np.mean(v)) for k, v in mses.items()],
                key=lambda x: x[1]
            )
            for i, (method, mse) in enumerate(sorted_mses[:5], 1):
                print(f"  {i}. {method}: {mse:.4f}")


def main():
    """Main visualization function."""
    # Load results
    results_path = Path(__file__).parent / "logs/2026_01_08_20_36_53/results.json"
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    # Create output directory
    output_dir = results_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Get environments
    environments = list(results['results_reward'].keys())
    print(f"\nEnvironments: {environments}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    # 1. Reward comparison
    plot_metric_comparison(
        results['results_reward'],
        'Reward',
        environments,
        output_dir
    )
    
    # 2. Regret comparison
    if 'results_regret' in results:
        plot_metric_comparison(
            results['results_regret'],
            'Regret',
            environments,
            output_dir
        )
    
    # 3. Weights MSE comparison
    if 'results_weights_mse' in results:
        plot_metric_comparison(
            results['results_weights_mse'],
            'Weights MSE',
            environments,
            output_dir
        )
    
    # 4. Prompt sensitivity
    plot_prompt_sensitivity(
        results['results_reward'],
        'Reward',
        environments,
        output_dir
    )
    
    # 5. Weights comparison
    if 'learned_weights_raw' in results and 'oracle_weights_GT' in results:
        plot_weights_comparison(
            results['learned_weights_raw'],
            results['oracle_weights_GT'],
            environments,
            output_dir
        )
    
    # 6. Print summary table
    create_summary_table(results, environments, output_dir)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

