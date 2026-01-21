"""
PPO Agent for Resource Allocation

Wrapper around Stable-Baselines3 PPO with custom callbacks
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Dict
import torch


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress
    """
    
    def __init__(self, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.qos_metrics = []
        
    def _on_step(self) -> bool:
        """Called at each step"""
        # Log episode metrics when done
        if self.locals.get('dones'):
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    
                    # Extract QoS metrics from info
                    if 'total_throughput' in info:
                        self.qos_metrics.append({
                            'throughput': info['total_throughput'],
                            'fairness': info.get('fairness', 0),
                            'latency': info.get('mean_latency', 0)
                        })
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout"""
        if len(self.qos_metrics) > 0:
            avg_throughput = np.mean([m['throughput'] for m in self.qos_metrics[-10:]])
            avg_fairness = np.mean([m['fairness'] for m in self.qos_metrics[-10:]])
            
            if self.verbose > 0:
                print(f"Recent Avg Throughput: {avg_throughput:.2f} Mbps, "
                      f"Fairness: {avg_fairness:.3f}")


class PPOAgent:
    """
    PPO Agent for wireless resource allocation
    """
    
    def __init__(self, 
                 env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 tensorboard_log: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize PPO agent
        
        Args:
            env: Gym environment
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: Clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            tensorboard_log: Tensorboard log directory
            device: Device (cpu/cuda)
        """
        self.env = env
        
        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=1
        )
        
        self.training_callback = TrainingCallback()
    
    def train(self, 
              total_timesteps: int,
              callback = None,
              log_interval: int = 10):
        """
        Train the agent
        
        Args:
            total_timesteps: Total training timesteps
            callback: Additional callback
            log_interval: Logging frequency
        """
        if callback is None:
            callback = self.training_callback
        
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        print("Training completed!")
    
    def evaluate(self, n_episodes: int = 20) -> Dict:
        """
        Evaluate trained agent
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_throughputs = []
        episode_fairness = []
        episode_latencies = []
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            
            throughputs = []
            fairness_values = []
            latencies = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                throughputs.append(info['total_throughput'])
                fairness_values.append(info['fairness'])
                latencies.append(info['mean_latency'])
            
            episode_rewards.append(episode_reward)
            episode_throughputs.append(np.mean(throughputs))
            episode_fairness.append(np.mean(fairness_values))
            episode_latencies.append(np.mean(latencies))
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_throughput': np.mean(episode_throughputs),
            'std_throughput': np.std(episode_throughputs),
            'mean_fairness': np.mean(episode_fairness),
            'std_fairness': np.std(episode_fairness),
            'mean_latency': np.mean(episode_latencies),
            'std_latency': np.std(episode_latencies),
        }
        
        return results
    
    def save(self, path: str):
        """Save model to disk"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action for given observation
        
        Args:
            observation: Current state
            deterministic: Use deterministic policy
        
        Returns:
            action, state (for recurrent policies)
        """
        return self.model.predict(observation, deterministic=deterministic)


# Test PPO Agent
if __name__ == "__main__":
    print("Testing PPO Agent...")
    
    # Import environment
    import sys
    sys.path.append('..')
    from environment.network_env import WirelessNetworkEnv
    
    # Create environment
    env = WirelessNetworkEnv(n_users=5, n_rbs=10, max_steps=50)
    
    # Create agent
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64
    )
    
    # Short training test
    print("\n--- Short Training Test (5000 steps) ---")
    agent.train(total_timesteps=5000, log_interval=1)
    
    # Evaluate
    print("\n--- Evaluation (5 episodes) ---")
    results = agent.evaluate(n_episodes=5)
    
    for key, value in results.items():
        print(f"{key}: {value:.3f}")
    
    print("\n✓ PPO Agent working correctly!")