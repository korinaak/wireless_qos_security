"""
Robust PPO Agent with Integrated Defenses

Combines PPO with defense mechanisms:
- Adversarial training
- Anomaly detection
- Input validation
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from typing import Optional, Dict
import sys
sys.path.append('..')

from defenses.adversarial_training import AdversarialTrainer
from defenses.anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector
from defenses.input_validator import InputValidator


class RobustEnvironmentWrapper(gym.Wrapper):
    """
    Environment wrapper that applies defense mechanisms
    """
    
    def __init__(self,
                 env: gym.Env,
                 use_anomaly_detection: bool = True,
                 use_input_validation: bool = True,
                 anomaly_config: Optional[Dict] = None,
                 validation_config: Optional[Dict] = None):
        """
        Args:
            env: Base environment
            use_anomaly_detection: Enable anomaly detection
            use_input_validation: Enable input validation
            anomaly_config: Config for anomaly detector
            validation_config: Config for input validator
        """
        super().__init__(env)
        
        n_users = env.n_users
        
        self.use_anomaly_detection = use_anomaly_detection
        self.use_input_validation = use_input_validation
        
        # Initialize defenses
        if use_anomaly_detection:
            if anomaly_config and anomaly_config.get('use_ensemble', False):
                self.anomaly_detector = EnsembleAnomalyDetector(
                    n_users=n_users,
                    voting_threshold=anomaly_config.get('voting_threshold', 2)
                )
            else:
                self.anomaly_detector = AnomalyDetector(
                    n_users=n_users,
                    method=anomaly_config.get('method', 'zscore') if anomaly_config else 'zscore',
                    threshold=anomaly_config.get('threshold', 2.5) if anomaly_config else 2.5,
                    window_size=anomaly_config.get('window_size', 10) if anomaly_config else 10
                )
        
        if use_input_validation:
            self.input_validator = InputValidator(
                n_users=n_users,
                cqi_bounds=validation_config.get('cqi_bounds', (0.0, 1.0)) if validation_config else (0.0, 1.0),
                buffer_bounds=validation_config.get('buffer_bounds', (0.0, 1.0)) if validation_config else (0.0, 1.0),
                max_change_rate=validation_config.get('max_change_rate', 0.1) if validation_config else 0.1
            )
        
        # Statistics
        self.total_anomalies_detected = 0
        self.total_validations = 0
    
    def _apply_defenses(self, state: np.ndarray) -> np.ndarray:
        """
        Apply defense mechanisms to state
        """
        defended_state = state.copy()
        n_users = self.env.n_users
        
        # 1. Anomaly detection
        if self.use_anomaly_detection:
            cqi_values = defended_state[:n_users]
            filtered_cqi, anomalies = self.anomaly_detector.detect_anomalies(cqi_values)
            
            if len(anomalies) > 0:
                self.total_anomalies_detected += len(anomalies)
            
            defended_state[:n_users] = filtered_cqi
        
        # 2. Input validation
        if self.use_input_validation:
            defended_state = self.input_validator.validate(defended_state)
            self.total_validations += 1
        
        return defended_state
    
    def reset(self, **kwargs):
        """Reset with defense reset"""
        state, info = self.env.reset(**kwargs)
        
        # Reset defenses
        if self.use_anomaly_detection:
            self.anomaly_detector.reset()
        if self.use_input_validation:
            self.input_validator.reset()
        
        # Apply defenses to initial state
        defended_state = self._apply_defenses(state)
        
        return defended_state, info
    
    def step(self, action):
        """Step with defense application"""
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply defenses
        defended_state = self._apply_defenses(state)
        
        return defended_state, reward, terminated, truncated, info
    
    def get_defense_statistics(self) -> Dict:
        """Get statistics from defense mechanisms"""
        stats = {
            'total_anomalies_detected': self.total_anomalies_detected,
            'total_validations': self.total_validations
        }
        
        if self.use_anomaly_detection:
            stats['anomaly_stats'] = self.anomaly_detector.get_statistics() if hasattr(self.anomaly_detector, 'get_statistics') else {}
        
        if self.use_input_validation:
            stats['validation_stats'] = self.input_validator.get_statistics()
        
        return stats


class RobustPPOAgent:
    """
    Robust PPO Agent with integrated defense mechanisms
    """
    
    def __init__(self,
                 env: gym.Env,
                 defense_strategy: str = 'full',  # 'full', 'adversarial_only', 'detection_only', 'none'
                 adversarial_config: Optional[Dict] = None,
                 anomaly_config: Optional[Dict] = None,
                 validation_config: Optional[Dict] = None,
                 **ppo_kwargs):
        """
        Initialize Robust PPO agent
        
        Args:
            env: Base environment
            defense_strategy: Defense configuration
            adversarial_config: Adversarial training config
            anomaly_config: Anomaly detection config
            validation_config: Input validation config
            **ppo_kwargs: PPO hyperparameters
        """
        self.base_env = env
        self.defense_strategy = defense_strategy
        
        # Configure defenses based on strategy
        if defense_strategy == 'full':
            use_adversarial = True
            use_anomaly = True
            use_validation = True
        elif defense_strategy == 'adversarial_only':
            use_adversarial = True
            use_anomaly = False
            use_validation = False
        elif defense_strategy == 'detection_only':
            use_adversarial = False
            use_anomaly = True
            use_validation = True
        else:  # 'none'
            use_adversarial = False
            use_anomaly = False
            use_validation = False
        
        # Wrap environment with defenses
        self.env = env
        
        # Apply adversarial training wrapper if enabled
        if use_adversarial:
            adversarial_config = adversarial_config or {}
            self.env = AdversarialTrainer(
                base_env=self.env,
                poison_probability=adversarial_config.get('poison_probability', 0.2),
                poison_magnitude=adversarial_config.get('poison_magnitude', 0.5),
                attack_type=adversarial_config.get('attack_type', 'cqi_overstatement')
            )
        
        # Apply detection/validation wrapper if enabled
        if use_anomaly or use_validation:
            self.env = RobustEnvironmentWrapper(
                env=self.env,
                use_anomaly_detection=use_anomaly,
                use_input_validation=use_validation,
                anomaly_config=anomaly_config,
                validation_config=validation_config
            )
        
        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            **ppo_kwargs
        )
    
    def train(self, total_timesteps: int, **kwargs):
        """Train the robust agent"""
        print(f"Training Robust PPO with strategy: {self.defense_strategy}")
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        print("Training completed!")
    
    def evaluate(self, eval_env: gym.Env, n_episodes: int = 20) -> Dict:
        """
        Evaluate on potentially attacked environment
        """
        episode_rewards = []
        episode_throughputs = []
        episode_fairness = []
        episode_latencies = []
        
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            
            throughputs = []
            fairness_values = []
            latencies = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                throughputs.append(info['total_throughput'])
                fairness_values.append(info['fairness'])
                latencies.append(info['mean_latency'])
            
            episode_rewards.append(episode_reward)
            episode_throughputs.append(np.mean(throughputs))
            episode_fairness.append(np.mean(fairness_values))
            episode_latencies.append(np.mean(latencies))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'mean_throughput': np.mean(episode_throughputs),
            'mean_fairness': np.mean(episode_fairness),
            'mean_latency': np.mean(episode_latencies),
        }
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model"""
        self.model = PPO.load(path, env=self.env)


# Test Robust PPO
if __name__ == "__main__":
    print("Testing Robust PPO Agent...")
    
    from environment.network_env import WirelessNetworkEnv
    
    # Create base environment
    env = WirelessNetworkEnv(
        n_users=5,
        n_rbs=10,
        max_steps=50,
        attack_enabled=True,
        attack_probability=0.3,
        attack_magnitude=0.5
    )
    
    # Test different defense strategies
    strategies = ['none', 'adversarial_only', 'detection_only', 'full']
    
    for strategy in strategies:
        print(f"\n=== Testing Strategy: {strategy} ===")
        
        agent = RobustPPOAgent(
            env=env,
            defense_strategy=strategy,
            learning_rate=3e-4,
            n_steps=512,
            verbose=0
        )
        
        # Short training
        agent.train(total_timesteps=2000, log_interval=100)
        
        # Evaluate
        results = agent.evaluate(env, n_episodes=5)
        print(f"Results: {results}")
    
    print("\n✓ Robust PPO Agent working correctly!")