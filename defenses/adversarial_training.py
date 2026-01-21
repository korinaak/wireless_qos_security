"""
Adversarial Training Defense

Trains DRL agent on both clean and poisoned samples
to build robustness against adversarial perturbations.

Key Idea: During training, inject poisoned states with probability p,
forcing the agent to learn a policy that works under both
clean and adversarial conditions.
"""

import numpy as np
from typing import Callable, Optional
import gymnasium as gym


class AdversarialTrainer(gym.Wrapper):
    """
    Adversarial Training wrapper for environment
    
    Augments training data with adversarial examples
    """
    
    def __init__(self,
                 base_env: gym.Env,
                 poison_probability: float = 0.2,
                 poison_magnitude: float = 0.5,
                 attack_type: str = 'cqi_overstatement'):
        """
        Initialize adversarial trainer
        
        Args:
            base_env: Base environment
            poison_probability: Probability of injecting poisoned state
            poison_magnitude: Perturbation strength
            attack_type: Type of attack to defend against
        """
        # Initialize the gym.Wrapper parent class
        super().__init__(base_env)

        self.poison_probability = poison_probability
        self.poison_magnitude = poison_magnitude
        self.attack_type = attack_type
        
        # Statistics
        self.n_clean_samples = 0
        self.n_poisoned_samples = 0
    
    # --- ADDED: Explicitly expose environment properties ---
    @property
    def n_users(self):
        """Forward n_users from the base environment"""
        return self.env.n_users

    @property
    def n_rbs(self):
        """Forward n_rbs from the base environment"""
        # Use getattr in case the base env doesn't have n_rbs (safer)
        return getattr(self.env, 'n_rbs', None)
    # -----------------------------------------------------

    def _generate_adversarial_state(self, clean_state: np.ndarray) -> np.ndarray:
        """
        Generate adversarial perturbation
        
        Args:
            clean_state: Clean observation
        
        Returns:
            Adversarially perturbed state
        """
        n_users = len(clean_state) // 2
        poisoned_state = clean_state.copy()
        
        # Perturb CQI values
        cqi_values = poisoned_state[:n_users]
        
        if self.attack_type == 'cqi_overstatement':
            # Randomly boost some CQIs
            n_perturb = max(1, int(n_users * 0.3))
            perturb_indices = np.random.choice(n_users, n_perturb, replace=False)
            
            for idx in perturb_indices:
                noise = np.random.uniform(0, self.poison_magnitude)
                cqi_values[idx] = np.clip(cqi_values[idx] + noise, 0, 1)
        
        elif self.attack_type == 'random_noise':
            # Add Gaussian noise
            noise = np.random.normal(0, self.poison_magnitude * 0.5, n_users)
            cqi_values = np.clip(cqi_values + noise, 0, 1)
        
        poisoned_state[:n_users] = cqi_values
        return poisoned_state
    
    def step(self, action):
        """
        Environment step with potential adversarial injection
        """
        # Execute action in base environment
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # With probability p, replace state with adversarial version
        if np.random.random() < self.poison_probability:
            state = self._generate_adversarial_state(state)
            self.n_poisoned_samples += 1
            info['adversarial_sample'] = True
        else:
            self.n_clean_samples += 1
            info['adversarial_sample'] = False
        
        return state, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)
    
    def get_statistics(self) -> dict:
        """Get training statistics"""
        total = self.n_clean_samples + self.n_poisoned_samples
        return {
            'clean_samples': self.n_clean_samples,
            'poisoned_samples': self.n_poisoned_samples,
            'poison_ratio': self.n_poisoned_samples / total if total > 0 else 0
        }


class RobustPPOTrainer:
    """
    PPO trainer with adversarial training
    
    Combines standard PPO with adversarial data augmentation
    """
    
    def __init__(self,
                 env: gym.Env,
                 ppo_model,
                 adversarial_config: dict):
        """
        Args:
            env: Base environment
            ppo_model: PPO model instance
            adversarial_config: Dict with poison_prob, magnitude, etc.
        """
        self.env = env
        self.ppo_model = ppo_model
        
        # Wrap environment with adversarial trainer
        self.adv_env = AdversarialTrainer(
            base_env=env,
            poison_probability=adversarial_config.get('poison_probability', 0.2),
            poison_magnitude=adversarial_config.get('poison_magnitude', 0.5),
            attack_type=adversarial_config.get('attack_type', 'cqi_overstatement')
        )
    
    def train(self, total_timesteps: int, **kwargs):
        """
        Train PPO with adversarial training
        """
        print(f"Starting Adversarial Training...")
        print(f"Poison probability: {self.adv_env.poison_probability}")
        
        # Train on adversarially augmented environment
        self.ppo_model.set_env(self.adv_env)
        self.ppo_model.learn(total_timesteps=total_timesteps, **kwargs)
        
        stats = self.adv_env.get_statistics()
        print(f"\nTraining Statistics:")
        print(f"  Clean samples: {stats['clean_samples']}")
        print(f"  Poisoned samples: {stats['poisoned_samples']}")
        print(f"  Poison ratio: {stats['poison_ratio']:.2%}")
    
    def get_robust_model(self):
        """Get the trained robust model"""
        return self.ppo_model


# Test Adversarial Training
if __name__ == "__main__":
    print("Testing Adversarial Training...")
    
    import sys
    sys.path.append('..')
    from environment.network_env import WirelessNetworkEnv
    
    # Create base environment
    base_env = WirelessNetworkEnv(
        n_users=5,
        n_rbs=10,
        max_steps=50,
        attack_enabled=False  # We'll inject attacks during training
    )
    
    # Wrap with adversarial trainer
    adv_env = AdversarialTrainer(
        base_env=base_env,
        poison_probability=0.3,
        poison_magnitude=0.5,
        attack_type='cqi_overstatement'
    )
    
    print("\n--- Testing Adversarial Environment ---")
    state, info = adv_env.reset()
    
    # Check if n_users is accessible
    print(f"Checking n_users attribute: {adv_env.n_users}")
    
    for step in range(10):
        action = adv_env.action_space.sample()
        state, reward, term, trunc, info = adv_env.step(action)
        
        if info.get('adversarial_sample'):
            print(f"Step {step+1}: ADVERSARIAL sample")
        else:
            print(f"Step {step+1}: Clean sample")
    
    stats = adv_env.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\n✓ Adversarial Training working correctly!")