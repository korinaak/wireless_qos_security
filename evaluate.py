"""
Evaluation Script

Comprehensively evaluates trained models across different scenarios:
- Clean environment (no attack)
- Under attack environment
- Compares baseline vs robust performance

Generates detailed metrics for the paper
"""

import os
import yaml
import argparse
import numpy as np
import json
from tqdm import tqdm

from environment.network_env import WirelessNetworkEnv
from environment.qos_metrics import QoSMetrics
from agents.ppo_agent import PPOAgent
from agents.robust_ppo import RobustPPOAgent
from stable_baselines3 import PPO


def load_config(config_path: str) -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model, env, n_episodes: int = 50):
    """
    Detailed evaluation of a model
    
    Returns comprehensive metrics
    """
    qos_metrics = QoSMetrics()
    
    episode_rewards = []
    episode_throughputs = []
    episode_fairness = []
    episode_latencies = []
    episode_satisfaction = []
    episode_percentile5 = []
    
    all_throughputs = []
    all_latencies = []
    
    print(f"Evaluating for {n_episodes} episodes...")
    
    for ep in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        step_throughputs = []
        step_fairness = []
        step_latencies = []
        step_user_throughputs = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Collect metrics
            user_tput = info['throughput']
            total_tput = info['total_throughput']
            fairness = info['fairness']
            latency = info['mean_latency']
            
            step_throughputs.append(total_tput)
            step_fairness.append(fairness)
            step_latencies.append(latency)
            step_user_throughputs.append(user_tput)
        
        # Episode statistics
        episode_rewards.append(episode_reward)
        episode_throughputs.append(np.mean(step_throughputs))
        episode_fairness.append(np.mean(step_fairness))
        episode_latencies.append(np.mean(step_latencies))
        
        # User-level statistics
        avg_user_tput = np.mean(step_user_throughputs, axis=0)
        all_throughputs.extend(avg_user_tput)
        all_latencies.extend(step_latencies)
        
        # Satisfaction (users meeting QoS requirement)
        satisfaction = qos_metrics.calculate_user_satisfaction(
            avg_user_tput, 
            min_required_throughput=1.0
        )
        episode_satisfaction.append(satisfaction)
        
        # 5th percentile throughput
        percentile_5 = qos_metrics.calculate_5th_percentile_throughput(avg_user_tput)
        episode_percentile5.append(percentile_5)
    
    # Aggregate results
    results = {
        # Reward
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        
        # Throughput
        'mean_throughput': float(np.mean(episode_throughputs)),
        'std_throughput': float(np.std(episode_throughputs)),
        'min_throughput': float(np.min(episode_throughputs)),
        'max_throughput': float(np.max(episode_throughputs)),
        
        # Fairness
        'mean_fairness': float(np.mean(episode_fairness)),
        'std_fairness': float(np.std(episode_fairness)),
        'min_fairness': float(np.min(episode_fairness)),
        
        # Latency
        'mean_latency': float(np.mean(episode_latencies)),
        'std_latency': float(np.std(episode_latencies)),
        'max_latency': float(np.max(episode_latencies)),
        
        # User satisfaction
        'mean_satisfaction': float(np.mean(episode_satisfaction)),
        'std_satisfaction': float(np.std(episode_satisfaction)),
        
        # Cell-edge performance
        'mean_5th_percentile_throughput': float(np.mean(episode_percentile5)),
        'std_5th_percentile_throughput': float(np.std(episode_percentile5)),
    }
    
    return results


def compare_scenarios(config: dict, model_dir: str, n_episodes: int = 50):
    """
    Compare all three models across clean and attacked environments
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    net_config = config['network']
    attack_config = config['attack']
    
    # Create environments
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
    models = {}
    model_files = {
        'baseline': 'ppo_baseline.zip',
        'under_attack': 'ppo_under_attack.zip',
        'robust': 'ppo_robust.zip'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            print(f"Loading {name} model from {path}...")
            models[name] = PPO.load(path)
        else:
            print(f"Warning: {name} model not found at {path}")
    
    # Evaluation matrix
    results = {}
    
    # 1. Baseline on clean environment
    if 'baseline' in models:
        print("\n--- Evaluating Baseline on Clean Environment ---")
        results['baseline_clean'] = evaluate_model(models['baseline'], clean_env, n_episodes)
    
    # 2. Baseline on attacked environment
    if 'baseline' in models:
        print("\n--- Evaluating Baseline on Attacked Environment ---")
        results['baseline_attack'] = evaluate_model(models['baseline'], attack_env, n_episodes)
    
    # 3. Under-attack trained model on attacked environment
    if 'under_attack' in models:
        print("\n--- Evaluating Under-Attack Trained Model on Attacked Environment ---")
        results['trained_under_attack'] = evaluate_model(models['under_attack'], attack_env, n_episodes)
    
    # 4. Robust model on attacked environment
    if 'robust' in models:
        print("\n--- Evaluating Robust Model on Attacked Environment ---")
        results['robust_attack'] = evaluate_model(models['robust'], attack_env, n_episodes)
    
    # 5. Robust model on clean environment (sanity check)
    if 'robust' in models:
        print("\n--- Evaluating Robust Model on Clean Environment (Sanity Check) ---")
        results['robust_clean'] = evaluate_model(models['robust'], clean_env, n_episodes)
    
    return results


def calculate_degradation(baseline, under_attack):
    """Calculate performance degradation percentage"""
    return {
        'throughput_degradation_%': ((baseline['mean_throughput'] - under_attack['mean_throughput']) / baseline['mean_throughput']) * 100,
        'fairness_degradation_%': ((baseline['mean_fairness'] - under_attack['mean_fairness']) / baseline['mean_fairness']) * 100,
        'latency_increase_%': ((under_attack['mean_latency'] - baseline['mean_latency']) / baseline['mean_latency']) * 100,
    }


def calculate_recovery(under_attack, with_defense):
    """Calculate defense recovery percentage"""
    return {
        'throughput_recovery_%': ((with_defense['mean_throughput'] - under_attack['mean_throughput']) / under_attack['mean_throughput']) * 100,
        'fairness_recovery_%': ((with_defense['mean_fairness'] - under_attack['mean_fairness']) / under_attack['mean_fairness']) * 100,
        'latency_reduction_%': ((under_attack['mean_latency'] - with_defense['mean_latency']) / under_attack['mean_latency']) * 100,
    }


def print_summary(results):
    """Print formatted summary"""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    scenarios = {
        'baseline_clean': 'Baseline (Clean)',
        'baseline_attack': 'Baseline (Under Attack)',
        'trained_under_attack': 'Trained Under Attack',
        'robust_attack': 'Robust (Under Attack)',
        'robust_clean': 'Robust (Clean)'
    }
    
    for key, name in scenarios.items():
        if key in results:
            print(f"\n{name}:")
            r = results[key]
            print(f"  Throughput: {r['mean_throughput']:.2f} ± {r['std_throughput']:.2f} Mbps")
            print(f"  Fairness: {r['mean_fairness']:.3f} ± {r['std_fairness']:.3f}")
            print(f"  Latency: {r['mean_latency']:.2f} ± {r['std_latency']:.2f}")
            print(f"  Satisfaction: {r['mean_satisfaction']:.2%}")
            print(f"  5th Percentile Throughput: {r['mean_5th_percentile_throughput']:.2f} Mbps")
    
    # Calculate degradation and recovery
    if 'baseline_clean' in results and 'baseline_attack' in results:
        print("\n" + "-"*70)
        print("ATTACK IMPACT:")
        degradation = calculate_degradation(results['baseline_clean'], results['baseline_attack'])
        for metric, value in degradation.items():
            print(f"  {metric}: {value:.1f}%")
    
    if 'baseline_attack' in results and 'robust_attack' in results:
        print("\n" + "-"*70)
        print("DEFENSE EFFECTIVENESS:")
        recovery = calculate_recovery(results['baseline_attack'], results['robust_attack'])
        for metric, value in recovery.items():
            print(f"  {metric}: {value:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run evaluation
    results = compare_scenarios(config, args.model_dir, args.n_episodes)
    
    # Print summary
    print_summary(results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {args.output}")
    print("\nRun plot_results.py to generate visualizations for your paper!")


if __name__ == "__main__":
    main()