"""
Detailed Analysis Script

Performs in-depth analysis of attack impact:
- Per-user throughput distribution
- Resource allocation efficiency
- Temporal degradation patterns
- Attack success metrics
"""

import os
import yaml
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from environment.network_env import WirelessNetworkEnv
from stable_baselines3 import PPO


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detailed_episode_analysis(model, env, n_episodes: int = 20):
    """
    Collect detailed per-step, per-user metrics
    """
    # Storage
    all_user_throughputs = []
    all_allocations = []
    all_cqi_mismatches = []
    malicious_throughputs = []
    honest_throughputs = []
    allocation_to_malicious = []
    allocation_to_honest = []
    
    rewards_per_episode = []
    fairness_per_episode = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        malicious_users = set(info.get('malicious_users', []))
        honest_users = set(range(env.n_users)) - malicious_users
        
        episode_user_tput = []
        episode_allocations = []
        episode_cqi_mismatch = []
        episode_mal_tput = []
        episode_hon_tput = []
        episode_mal_alloc = []
        episode_hon_alloc = []
        
        episode_rewards = []
        episode_fairness = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Collect data
            user_tput = info['throughput']
            allocations = action / np.sum(action)  # Normalized allocations
            
            episode_user_tput.append(user_tput)
            episode_allocations.append(allocations)
            episode_rewards.append(reward)
            episode_fairness.append(info['fairness'])
            
            # CQI mismatch (if attack enabled)
            if 'true_cqi' in info and 'reported_cqi' in info:
                mismatch = np.abs(info['reported_cqi'] - info['true_cqi'])
                episode_cqi_mismatch.append(mismatch)
            
            # Separate malicious vs honest
            if len(malicious_users) > 0:
                mal_tput = [user_tput[i] for i in malicious_users]
                hon_tput = [user_tput[i] for i in honest_users]
                mal_alloc = [allocations[i] for i in malicious_users]
                hon_alloc = [allocations[i] for i in honest_users]
                
                episode_mal_tput.append(np.mean(mal_tput))
                episode_hon_tput.append(np.mean(hon_tput))
                episode_mal_alloc.append(np.mean(mal_alloc))
                episode_hon_alloc.append(np.mean(hon_alloc))
        
        # Aggregate episode
        all_user_throughputs.extend(episode_user_tput)
        all_allocations.extend(episode_allocations)
        if len(episode_cqi_mismatch) > 0:
            all_cqi_mismatches.extend(episode_cqi_mismatch)
        
        if len(episode_mal_tput) > 0:
            malicious_throughputs.append(np.mean(episode_mal_tput))
            honest_throughputs.append(np.mean(episode_hon_tput))
            allocation_to_malicious.append(np.mean(episode_mal_alloc))
            allocation_to_honest.append(np.mean(episode_hon_alloc))
        
        rewards_per_episode.append(np.mean(episode_rewards))
        fairness_per_episode.append(np.mean(episode_fairness))
    
    results = {
        'all_user_throughputs': all_user_throughputs,
        'all_allocations': all_allocations,
        'all_cqi_mismatches': all_cqi_mismatches,
        'malicious_avg_throughput': np.mean(malicious_throughputs) if malicious_throughputs else None,
        'honest_avg_throughput': np.mean(honest_throughputs) if honest_throughputs else None,
        'malicious_avg_allocation': np.mean(allocation_to_malicious) if allocation_to_malicious else None,
        'honest_avg_allocation': np.mean(allocation_to_honest) if allocation_to_honest else None,
        'mean_reward': np.mean(rewards_per_episode),
        'mean_fairness': np.mean(fairness_per_episode),
        'attack_success_rate': None,  # Will compute
    }
    
    # Compute attack success rate
    if len(allocation_to_malicious) > 0 and len(allocation_to_honest) > 0:
        fair_share = 1.0 / env.n_users
        # Success = malicious users getting MORE than fair share on average
        success_rate = np.mean([1 if a > fair_share else 0 for a in allocation_to_malicious])
        results['attack_success_rate'] = success_rate
    
    return results


def compare_scenarios_detailed(config: dict, model_dir: str, output_dir: str):
    """
    Detailed comparison with visualization
    """
    print("\n" + "="*70)
    print("DETAILED SCENARIO ANALYSIS")
    print("="*70)
    
    net_config = config['network']
    attack_config = config['attack']
    
    # Environments
    clean_env = WirelessNetworkEnv(
        n_users=net_config['n_users'],
        n_rbs=net_config['n_rbs'],
        max_steps=net_config['max_steps_per_episode'],
        attack_enabled=False
    )
    
    attack_env = WirelessNetworkEnv(
        n_users=net_config['n_users'],
        n_rbs=net_config['n_rbs'],
        max_steps=net_config['max_steps_per_episode'],
        attack_enabled=True,
        attack_probability=attack_config['attack_probability'],
        attack_magnitude=attack_config['noise_magnitude']
    )
    
    # Load models
    baseline_path = os.path.join(model_dir, 'ppo_baseline.zip')
    robust_path = os.path.join(model_dir, 'ppo_robust.zip')
    
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline model not found at {baseline_path}")
        return
    
    baseline_model = PPO.load(baseline_path)
    
    results = {}
    
    # 1. Baseline on clean
    print("\n--- Baseline on Clean Environment ---")
    results['baseline_clean'] = detailed_episode_analysis(baseline_model, clean_env, n_episodes=20)
    
    # 2. Baseline under attack
    print("\n--- Baseline Under Attack ---")
    results['baseline_attack'] = detailed_episode_analysis(baseline_model, attack_env, n_episodes=20)
    
    # 3. Robust under attack (if available)
    if os.path.exists(robust_path):
        robust_model = PPO.load(robust_path)
        print("\n--- Robust Model Under Attack ---")
        results['robust_attack'] = detailed_episode_analysis(robust_model, attack_env, n_episodes=20)
    
    # Print summary
    print("\n" + "="*70)
    print("DETAILED RESULTS SUMMARY")
    print("="*70)
    
    for scenario, metrics in results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        print(f"  Mean Fairness: {metrics['mean_fairness']:.3f}")
        
        if metrics['malicious_avg_throughput'] is not None:
            print(f"  Malicious Avg Throughput: {metrics['malicious_avg_throughput']:.2f} Mbps")
            print(f"  Honest Avg Throughput: {metrics['honest_avg_throughput']:.2f} Mbps")
            print(f"  Malicious Avg Allocation: {metrics['malicious_avg_allocation']:.2%}")
            print(f"  Honest Avg Allocation: {metrics['honest_avg_allocation']:.2%}")
            print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'detailed_analysis.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                key: (val.tolist() if isinstance(val, np.ndarray) else 
                     [x.tolist() if isinstance(x, np.ndarray) else x for x in val] if isinstance(val, list) else val)
                for key, val in v.items()
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {output_dir}/detailed_analysis.json")
    
    # Generate plots
    plot_detailed_comparison(results, output_dir)
    
    return results


def plot_detailed_comparison(results, output_dir):
    """
    Generate detailed comparison plots
    """
    print("\nGenerating detailed plots...")
    
    # Plot 1: Malicious vs Honest User Performance
    if 'baseline_attack' in results and results['baseline_attack']['malicious_avg_throughput'] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        scenarios = []
        mal_tput = []
        hon_tput = []
        mal_alloc = []
        hon_alloc = []
        
        for scenario in ['baseline_clean', 'baseline_attack', 'robust_attack']:
            if scenario in results and results[scenario]['malicious_avg_throughput'] is not None:
                scenarios.append(scenario.replace('_', '\n'))
                mal_tput.append(results[scenario]['malicious_avg_throughput'])
                hon_tput.append(results[scenario]['honest_avg_throughput'])
                mal_alloc.append(results[scenario]['malicious_avg_allocation'] * 100)
                hon_alloc.append(results[scenario]['honest_avg_allocation'] * 100)
        
        # Throughput comparison
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[0].bar(x - width/2, mal_tput, width, label='Malicious Users', color='#e74c3c', alpha=0.8)
        axes[0].bar(x + width/2, hon_tput, width, label='Honest Users', color='#2ecc71', alpha=0.8)
        axes[0].set_xlabel('Scenario')
        axes[0].set_ylabel('Average Throughput (Mbps)')
        axes[0].set_title('Throughput: Malicious vs Honest Users')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(scenarios)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Allocation comparison
        axes[1].bar(x - width/2, mal_alloc, width, label='Malicious Users', color='#e74c3c', alpha=0.8)
        axes[1].bar(x + width/2, hon_alloc, width, label='Honest Users', color='#2ecc71', alpha=0.8)
        axes[1].set_xlabel('Scenario')
        axes[1].set_ylabel('Average Resource Allocation (%)')
        axes[1].set_title('Resource Allocation: Malicious vs Honest Users')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(scenarios)
        axes[1].axhline(y=10, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Fair Share (10%)')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'malicious_vs_honest.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'malicious_vs_honest.pdf'), bbox_inches='tight')
        plt.close()
        print("✓ Saved malicious_vs_honest plots")
    
    # Plot 2: Attack Success Rate
    if 'baseline_attack' in results and results['baseline_attack']['attack_success_rate'] is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scenarios = []
        success_rates = []
        
        for scenario in ['baseline_attack', 'robust_attack']:
            if scenario in results and results[scenario]['attack_success_rate'] is not None:
                scenarios.append(scenario.replace('_', '\n'))
                success_rates.append(results[scenario]['attack_success_rate'] * 100)
        
        colors = ['#e74c3c', '#3498db']
        bars = ax.bar(scenarios, success_rates, color=colors[:len(scenarios)], alpha=0.8, width=0.5)
        
        ax.set_ylabel('Attack Success Rate (%)')
        ax.set_title('Attack Success Rate: Malicious Users Exceeding Fair Share')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attack_success_rate.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'attack_success_rate.pdf'), bbox_inches='tight')
        plt.close()
        print("✓ Saved attack_success_rate plot")


def main():
    parser = argparse.ArgumentParser(description='Detailed analysis of trained models')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/detailed_analysis')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    results = compare_scenarios_detailed(config, args.model_dir, args.output_dir)
    
    print("\n" + "="*70)
    print("DETAILED ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()