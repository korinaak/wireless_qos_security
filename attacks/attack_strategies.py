"""
Various Attack Strategies

Provides different attack patterns:
- Random Noise: Gaussian/Uniform noise injection
- Targeted Attack: Focus on specific users
- Coordinated Attack: Multiple malicious users cooperate
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List


class AttackStrategy(ABC):
    """
    Base class for attack strategies
    """
    
    def __init__(self, n_users: int, magnitude: float = 0.5):
        self.n_users = n_users
        self.magnitude = magnitude
    
    @abstractmethod
    def apply_attack(self, clean_state: np.ndarray) -> np.ndarray:
        """Apply attack to clean state"""
        pass


class RandomNoiseAttack(AttackStrategy):
    """
    Random noise injection attack
    
    Adds Gaussian or uniform noise to state observations
    """
    
    def __init__(self, 
                 n_users: int,
                 magnitude: float = 0.3,
                 noise_type: str = 'gaussian',
                 attack_probability: float = 0.3):
        """
        Args:
            n_users: Number of users
            magnitude: Noise standard deviation or range
            noise_type: 'gaussian' or 'uniform'
            attack_probability: Fraction of users to attack
        """
        super().__init__(n_users, magnitude)
        self.noise_type = noise_type
        self.attack_probability = attack_probability
        
        # Select victims
        n_victims = int(n_users * attack_probability)
        self.victim_users = np.random.choice(n_users, size=n_victims, replace=False)
    
    def apply_attack(self, clean_state: np.ndarray) -> np.ndarray:
        """
        Add noise to CQI values
        """
        poisoned_state = clean_state.copy()
        cqi_values = poisoned_state[:self.n_users]
        
        for user_idx in self.victim_users:
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0, self.magnitude)
            elif self.noise_type == 'uniform':
                noise = np.random.uniform(-self.magnitude, self.magnitude)
            else:
                noise = 0
            
            cqi_values[user_idx] = np.clip(cqi_values[user_idx] + noise, 0, 1)
        
        poisoned_state[:self.n_users] = cqi_values
        return poisoned_state
    
    def get_victims(self) -> List[int]:
        return self.victim_users.tolist()


class TargetedAttack(AttackStrategy):
    """
    Targeted attack on specific high-priority users
    
    Aims to degrade QoS for specific users by manipulating their CQI
    """
    
    def __init__(self,
                 n_users: int,
                 target_users: List[int],
                 magnitude: float = 0.7,
                 attack_mode: str = 'suppress'):
        """
        Args:
            n_users: Total number of users
            target_users: List of user indices to target
            magnitude: Attack strength
            attack_mode: 'suppress' (reduce their CQI) or 'inflate' (steal their resources)
        """
        super().__init__(n_users, magnitude)
        self.target_users = target_users
        self.attack_mode = attack_mode
    
    def apply_attack(self, clean_state: np.ndarray) -> np.ndarray:
        """
        Apply targeted attack
        """
        poisoned_state = clean_state.copy()
        cqi_values = poisoned_state[:self.n_users]
        
        for user_idx in self.target_users:
            if self.attack_mode == 'suppress':
                # Make their channel appear worse
                cqi_values[user_idx] = np.clip(
                    cqi_values[user_idx] - self.magnitude, 0, 1
                )
            elif self.attack_mode == 'inflate':
                # Make attacker's channel appear better (if attacker controls target)
                cqi_values[user_idx] = np.clip(
                    cqi_values[user_idx] + self.magnitude, 0, 1
                )
        
        poisoned_state[:self.n_users] = cqi_values
        return poisoned_state
    
    def get_targets(self) -> List[int]:
        return self.target_users


class CoordinatedAttack(AttackStrategy):
    """
    Coordinated attack where multiple malicious users cooperate
    
    Malicious users coordinate to maximize their collective gain
    """
    
    def __init__(self,
                 n_users: int,
                 n_malicious: int = 3,
                 magnitude: float = 0.6,
                 strategy: str = 'resource_theft'):
        """
        Args:
            n_users: Total number of users
            n_malicious: Number of coordinating malicious users
            magnitude: Attack strength
            strategy: 'resource_theft' or 'denial_of_service'
        """
        super().__init__(n_users, magnitude)
        self.n_malicious = n_malicious
        self.strategy = strategy
        
        # Select malicious coalition
        self.malicious_coalition = np.random.choice(
            n_users, size=n_malicious, replace=False
        )
        
        # Select victims (remaining users)
        self.victims = [u for u in range(n_users) if u not in self.malicious_coalition]
    
    def apply_attack(self, clean_state: np.ndarray) -> np.ndarray:
        """
        Apply coordinated attack
        """
        poisoned_state = clean_state.copy()
        cqi_values = poisoned_state[:self.n_users]
        
        if self.strategy == 'resource_theft':
            # Malicious: inflate their CQI
            for m_user in self.malicious_coalition:
                boost = np.random.uniform(0.5, 1.0) * self.magnitude
                cqi_values[m_user] = np.clip(cqi_values[m_user] + boost, 0, 1)
            
            # Victims: suppress their CQI slightly
            for v_user in self.victims:
                suppress = np.random.uniform(0, 0.3) * self.magnitude
                cqi_values[v_user] = np.clip(cqi_values[v_user] - suppress, 0, 1)
        
        elif self.strategy == 'denial_of_service':
            # Maximize disruption: make all victims appear to have poor channels
            for v_user in self.victims:
                cqi_values[v_user] = np.clip(
                    cqi_values[v_user] * (1 - self.magnitude), 0, 1
                )
        
        poisoned_state[:self.n_users] = cqi_values
        return poisoned_state
    
    def get_coalition(self) -> List[int]:
        return self.malicious_coalition.tolist()
    
    def get_victims(self) -> List[int]:
        return self.victims


# Test Attack Strategies
if __name__ == "__main__":
    print("Testing Attack Strategies...")
    
    n_users = 10
    clean_cqi = np.random.uniform(0.4, 0.8, n_users)
    clean_buffers = np.random.uniform(0.2, 0.6, n_users)
    clean_state = np.concatenate([clean_cqi, clean_buffers])
    
    print(f"Clean CQI: {clean_cqi}\n")
    
    # Test 1: Random Noise Attack
    print("=== Random Noise Attack ===")
    random_attack = RandomNoiseAttack(
        n_users=n_users,
        magnitude=0.3,
        noise_type='gaussian',
        attack_probability=0.4
    )
    
    poisoned = random_attack.apply_attack(clean_state)
    print(f"Victims: {random_attack.get_victims()}")
    print(f"Poisoned CQI: {poisoned[:n_users]}\n")
    
    # Test 2: Targeted Attack
    print("=== Targeted Attack ===")
    targeted_attack = TargetedAttack(
        n_users=n_users,
        target_users=[0, 1, 2],
        magnitude=0.7,
        attack_mode='suppress'
    )
    
    poisoned = targeted_attack.apply_attack(clean_state)
    print(f"Targets: {targeted_attack.get_targets()}")
    print(f"Poisoned CQI: {poisoned[:n_users]}\n")
    
    # Test 3: Coordinated Attack
    print("=== Coordinated Attack ===")
    coordinated_attack = CoordinatedAttack(
        n_users=n_users,
        n_malicious=3,
        magnitude=0.6,
        strategy='resource_theft'
    )
    
    poisoned = coordinated_attack.apply_attack(clean_state)
    print(f"Malicious Coalition: {coordinated_attack.get_coalition()}")
    print(f"Victims: {coordinated_attack.get_victims()}")
    print(f"Poisoned CQI: {poisoned[:n_users]}\n")
    
    print("✓ All attack strategies working correctly!")