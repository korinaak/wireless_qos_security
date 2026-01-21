"""
Main Training Script

Trains three variants of PPO:
1. Baseline (no attack, no defense)
2. Standard PPO under attack
3. Robust PPO with defenses

Usage:
    python train.py --config configs/config.yaml
"""

import os
import yaml
import argparse
import numpy as np
from datetime import datetime

from environment.network_env import WirelessNetworkEnv
from agents.ppo_agent import PPOAgent
from agents.robust_ppo import RobustPPOAgent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, scenario: str):
    """
    Create environment based on scenario
    
    Args:
        config: Configuration dictionary
        scenario: 'baseline', 'attack', or 'defense'
    
    Returns:
        Environment instance
    """
    net_config = config['network']
    attack_config = config['attack']
    
    if scenario == 'baseline':
        # No attack
        env = WirelessNetworkEnv(
            n_users=net_config['n_users'],
            n_rbs=net_config['n_rbs'],
            max_steps=net_config['max_steps_per_episode'],
            attack_enabled=False
        )
    elif scenario == 'attack':
        # With attack, no defense
        env = WirelessNetworkEnv(
            n_users=net_config['n_users'],
            n_rbs=net_config['n_rbs'],
            max_steps=net_config['max_steps_per_episode'],
            attack_enabled=True,
            attack_probability=attack_config['attack_probability'],
            attack_magnitude=attack_config['noise_magnitude']
        )
    else:  # 'defense'
        # With attack (will be handled by Robust PPO)
        env = WirelessNetworkEnv(
            n_users=net_config['n_users'],
            n_rbs=net_config['n_rbs'],
            max_steps=net_config['max_steps_per_episode'],
            attack_enabled=True,
            attack_probability=attack_config['attack_probability'],
            attack_magnitude=attack_config['noise_magnitude']
        )
    
    return env


def train_baseline(config: dict, save_dir: str):
    """
    Train baseline PPO (no attack, no defense)
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE PPO (No Attack, No Defense)")
    print("="*70)
    
    env = create_env(config, 'baseline')
    
    ppo_config = config['ppo']
    training_config = config['training']
    
    agent = PPOAgent(
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        tensorboard_log=training_config['log_dir'],
        device='auto'
    )
    
    # Train
    agent.train(
        total_timesteps=training_config['total_timesteps'],
        log_interval=10
    )
    
    # Save
    model_path = os.path.join(save_dir, 'ppo_baseline.zip')
    agent.save(model_path)
    print(f"✓ Baseline model saved to {model_path}")
    
    # Evaluate
    results = agent.evaluate(n_episodes=config['training']['n_eval_episodes'])
    print(f"\nBaseline Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
    
    return agent, results


def train_under_attack(config: dict, save_dir: str):
    """
    Train standard PPO under attack (no defense)
    """
    print("\n" + "="*70)
    print("TRAINING STANDARD PPO UNDER ATTACK (No Defense)")
    print("="*70)
    
    env = create_env(config, 'attack')
    
    ppo_config = config['ppo']
    training_config = config['training']
    
    agent = PPOAgent(
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        tensorboard_log=training_config['log_dir'],
        device='auto'
    )
    
    # Train
    agent.train(
        total_timesteps=training_config['total_timesteps'],
        log_interval=10
    )
    
    # Save
    model_path = os.path.join(save_dir, 'ppo_under_attack.zip')
    agent.save(model_path)
    print(f"✓ Under-attack model saved to {model_path}")
    
    # Evaluate
    results = agent.evaluate(n_episodes=config['training']['n_eval_episodes'])
    print(f"\nUnder Attack Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
    
    return agent, results


def train_robust(config: dict, save_dir: str):
    """
    Train Robust PPO with defenses
    """
    print("\n" + "="*70)
    print("TRAINING ROBUST PPO (With Defenses)")
    print("="*70)
    
    env = create_env(config, 'defense')
    
    ppo_config = config['ppo']
    training_config = config['training']
    defense_config = config['defense']
    
    # Configure defenses
    adversarial_config = {
        'poison_probability': defense_config['adversarial_training']['poison_ratio'],
        'poison_magnitude': config['attack']['noise_magnitude'],
        'attack_type': 'cqi_overstatement'
    }
    
    anomaly_config = {
        'method': 'zscore',
        'threshold': defense_config['anomaly_detection']['threshold'],
        'window_size': defense_config['anomaly_detection']['window_size']
    }
    
    validation_config = {
        'cqi_bounds': tuple(defense_config['input_validation']['cqi_bounds']),
        'buffer_bounds': tuple(defense_config['input_validation']['buffer_bounds']),
        'max_change_rate': defense_config['input_validation']['rate_limit']
    }
    
    agent = RobustPPOAgent(
        env=env,
        defense_strategy='full',  # Use all defenses
        adversarial_config=adversarial_config,
        anomaly_config=anomaly_config,
        validation_config=validation_config,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        tensorboard_log=training_config['log_dir'],
        device='auto',
        verbose=1
    )
    
    # Train
    agent.train(
        total_timesteps=training_config['total_timesteps'],
        log_interval=10
    )
    
    # Save
    model_path = os.path.join(save_dir, 'ppo_robust.zip')
    agent.save(model_path)
    print(f"✓ Robust model saved to {model_path}")
    
    # Evaluate
    eval_env = create_env(config, 'attack')
    results = agent.evaluate(eval_env, n_episodes=config['training']['n_eval_episodes'])
    print(f"\nRobust PPO Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
    
    return agent, results


def main():
    parser = argparse.ArgumentParser(description='Train PPO agents for wireless resource allocation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--scenarios', nargs='+', 
                       default=['baseline', 'attack', 'robust'],
                       choices=['baseline', 'attack', 'robust', 'all'],
                       help='Which scenarios to train')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'training_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    os.makedirs(config['training']['model_save_path'], exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"WIRELESS NETWORK QOS SECURITY - TRAINING PIPELINE")
    print(f"{'='*70}")
    print(f"Output directory: {save_dir}")
    print(f"Configuration: {args.config}\n")
    
    results = {}
    
    # Determine which scenarios to run
    scenarios = args.scenarios
    if 'all' in scenarios:
        scenarios = ['baseline', 'attack', 'robust']
    
    # Train each scenario
    if 'baseline' in scenarios:
        _, baseline_results = train_baseline(config, save_dir)
        results['baseline'] = baseline_results
    
    if 'attack' in scenarios:
        _, attack_results = train_under_attack(config, save_dir)
        results['attack'] = attack_results
    
    if 'robust' in scenarios:
        _, robust_results = train_robust(config, save_dir)
        results['robust'] = robust_results
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    for scenario, metrics in results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Throughput: {metrics['mean_throughput']:.2f} ± {metrics.get('std_throughput', 0):.2f} Mbps")
        print(f"  Fairness: {metrics['mean_fairness']:.3f} ± {metrics.get('std_fairness', 0):.3f}")
        print(f"  Latency: {metrics['mean_latency']:.2f} ± {metrics.get('std_latency', 0):.2f}")
    
    # Save results summary
    import json
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*70)
    print("All training complete! Run evaluate.py for detailed comparison.")
    print("="*70)


if __name__ == "__main__":
    main()