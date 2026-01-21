"""
CQI Falsification Attack

Malicious users report false Channel Quality Indicator (CQI) values
to manipulate resource allocation decisions.

Attack Types:
1. Overstatement: Report higher CQI to steal resources
2. Understatement: Report lower CQI to cause underutilization
3. Random: Random perturbations
"""

import numpy as np
from typing import List, Tuple


class CQIPoisoning:
    """
    CQI Falsification Attack Implementation
    """
    
    def __init__(self,
                 attack_type: str = 'overstatement',
                 magnitude: float = 0.5,
                 probability: float = 0.3,
                 n_users: int = 10):
        """
        Initialize CQI poisoning attack
        
        Args:
            attack_type: 'overstatement', 'understatement', 'random'
            magnitude: Perturbation strength [0, 1]
            probability: Fraction of malicious users
            n_users: Total number of users
        """
        self.attack_type = attack_type
        self.magnitude = magnitude
        self.probability = probability
        self.n_users = n_users
        
        # Select malicious users
        n_malicious = int(n_users * probability)
        self.malicious_users = np.random.choice(
            n_users, 
            size=n_malicious, 
            replace=False
        )
    
    def poison_state(self, clean_state: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Apply CQI poisoning to state observation
        
        Args:
            clean_state: Clean state [CQI_1, ..., CQI_N, Buffer_1, ..., Buffer_N]
        
        Returns:
            Tuple of (poisoned_state, malicious_user_ids)
        """
        poisoned_state = clean_state.copy()
        
        # Extract CQI portion (first n_users elements)
        cqi_values = poisoned_state[:self.n_users]
        
        # Apply attack based on type
        for user_idx in self.malicious_users:
            if self.attack_type == 'overstatement':
                # Report falsely high CQI
                noise = np.random.uniform(0, self.magnitude)
                cqi_values[user_idx] = np.clip(cqi_values[user_idx] + noise, 0, 1)
                
            elif self.attack_type == 'understatement':
                # Report falsely low CQI
                noise = np.random.uniform(0, self.magnitude)
                cqi_values[user_idx] = np.clip(cqi_values[user_idx] - noise, 0, 1)
                
            elif self.attack_type == 'random':
                # Random perturbation
                noise = np.random.uniform(-self.magnitude, self.magnitude)
                cqi_values[user_idx] = np.clip(cqi_values[user_idx] + noise, 0, 1)
        
        poisoned_state[:self.n_users] = cqi_values
        
        return poisoned_state, self.malicious_users.tolist()
    
    def update_malicious_users(self):
        """Re-select malicious users (for new episode)"""
        n_malicious = int(self.n_users * self.probability)
        self.malicious_users = np.random.choice(
            self.n_users,
            size=n_malicious,
            replace=False
        )
    
    def get_attack_info(self) -> dict:
        """Get attack configuration info"""
        return {
            'attack_type': self.attack_type,
            'magnitude': self.magnitude,
            'probability': self.probability,
            'malicious_users': self.malicious_users.tolist(),
            'n_malicious': len(self.malicious_users)
        }


class AdaptiveCQIPoisoning(CQIPoisoning):
    """
    Adaptive CQI poisoning that adjusts based on observed allocations
    
    More sophisticated: learns which perturbations are most effective
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Track allocation history
        self.allocation_history = []
        self.success_rate = np.zeros(self.n_users)
    
    def poison_state(self, clean_state: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Apply adaptive poisoning based on historical success
        """
        poisoned_state = clean_state.copy()
        cqi_values = poisoned_state[:self.n_users]
        
        for user_idx in self.malicious_users:
            # Adapt magnitude based on success rate
            adaptive_magnitude = self.magnitude * (1 + self.success_rate[user_idx])
            
            if self.attack_type == 'overstatement':
                noise = np.random.uniform(0, adaptive_magnitude)
                cqi_values[user_idx] = np.clip(cqi_values[user_idx] + noise, 0, 1)
        
        poisoned_state[:self.n_users] = cqi_values
        
        return poisoned_state, self.malicious_users.tolist()
    
    def update_success(self, allocations: np.ndarray):
        """
        Update success rate based on received allocations
        
        Args:
            allocations: Resource allocation proportions
        """
        for user_idx in self.malicious_users:
            # Success = receiving more than fair share
            fair_share = 1.0 / self.n_users
            if allocations[user_idx] > fair_share:
                self.success_rate[user_idx] = min(1.0, self.success_rate[user_idx] + 0.1)
            else:
                self.success_rate[user_idx] = max(0.0, self.success_rate[user_idx] - 0.05)


# Test CQI Poisoning
if __name__ == "__main__":
    print("Testing CQI Poisoning Attack...")
    
    # Create attack
    attack = CQIPoisoning(
        attack_type='overstatement',
        magnitude=0.5,
        probability=0.3,
        n_users=10
    )
    
    print(f"Attack Info: {attack.get_attack_info()}")
    
    # Simulate clean state
    clean_cqi = np.random.uniform(0.3, 0.8, 10)
    clean_buffers = np.random.uniform(0.2, 0.6, 10)
    clean_state = np.concatenate([clean_cqi, clean_buffers])
    
    print(f"\nClean CQI: {clean_cqi}")
    
    # Apply attack
    poisoned_state, malicious = attack.poison_state(clean_state)
    poisoned_cqi = poisoned_state[:10]
    
    print(f"Poisoned CQI: {poisoned_cqi}")
    print(f"Malicious users: {malicious}")
    
    # Show differences
    print("\n--- CQI Changes ---")
    for i in range(10):
        if i in malicious:
            print(f"User {i} (malicious): {clean_cqi[i]:.3f} → {poisoned_cqi[i]:.3f} "
                  f"(+{poisoned_cqi[i] - clean_cqi[i]:.3f})")
    
    # Test adaptive attack
    print("\n\n=== Testing Adaptive Attack ===")
    adaptive_attack = AdaptiveCQIPoisoning(
        attack_type='overstatement',
        magnitude=0.3,
        probability=0.2,
        n_users=10
    )
    
    # Simulate multiple rounds
    for round in range(3):
        poisoned, malicious = adaptive_attack.poison_state(clean_state)
        
        # Simulate allocations (malicious users get more)
        allocations = np.random.uniform(0.05, 0.15, 10)
        for m in malicious:
            allocations[m] *= 1.5  # Malicious users successful
        
        adaptive_attack.update_success(allocations)
        print(f"Round {round+1} - Success rates: {adaptive_attack.success_rate[malicious]}")
    
    print("\n✓ CQI Poisoning attack working correctly!")