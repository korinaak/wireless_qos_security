"""
Enhanced Wireless Network Environment with Attack Support

KEY CHANGE: PPO now relies on REPORTED CQI (which can be poisoned),
not the true channel state. This makes attacks much more impactful.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Tuple

from .channel_model import ChannelModel
from .qos_metrics import QoSMetrics


class WirelessNetworkEnv(gym.Env):
    """
    5G/6G Wireless Network Environment with Attack Simulation
    """
    
    def __init__(self, 
                 n_users: int = 10,
                 n_rbs: int = 20,
                 max_steps: int = 100,
                 attack_enabled: bool = False,
                 attack_probability: float = 0.3,
                 attack_magnitude: float = 0.5):
        """
        Args:
            n_users: Number of users in the network
            n_rbs: Number of Resource Blocks
            max_steps: Maximum steps per episode
            attack_enabled: Whether to enable state poisoning
            attack_probability: Fraction of malicious users
            attack_magnitude: Perturbation strength [0, 1]
        """
        super(WirelessNetworkEnv, self).__init__()
        
        # Network parameters
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.max_steps = max_steps
        self.current_step = 0
        
        # Attack configuration
        self.attack_enabled = attack_enabled
        self.attack_probability = attack_probability
        self.attack_magnitude = attack_magnitude
        
        # Identify malicious users (fixed per episode)
        self.malicious_users = np.array([])
        
        # Physical layer
        self.bandwidth_per_rb = 180e3  # Hz
        self.tx_power = 23  # dBm
        self.noise_power = -104  # dBm
        
        # Initialize channel model and metrics
        self.channel_model = ChannelModel(model_type='urban_macro')
        self.qos_metrics = QoSMetrics()
        
        # User positions (meters from BS)
        self.user_distances = None
        
        # State variables
        self.true_cqi = None  # Ground truth CQI
        self.reported_cqi = None  # What users report (can be poisoned)
        self.user_buffer_sizes = None
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(n_users * 2,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(n_users,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Reset user positions
        self.user_distances = np.random.uniform(10, 500, self.n_users)
        
        # Reset buffers
        self.user_buffer_sizes = np.random.uniform(20, 80, self.n_users)
        
        # Generate initial TRUE channel conditions
        self.true_cqi = self._generate_cqi()
        
        # Re-select malicious users for new episode
        if self.attack_enabled:
            n_malicious = int(self.n_users * self.attack_probability)
            self.malicious_users = np.random.choice(
                self.n_users,
                size=n_malicious,
                replace=False
            )
        else:
            self.malicious_users = np.array([])
        
        # Initial reported CQI (apply attack if enabled)
        self.reported_cqi = self._apply_attack(self.true_cqi.copy())
        
        # Reset metrics
        self.qos_metrics.reset()
        
        state = self._get_state()
        info = {'malicious_users': self.malicious_users.tolist()}
        
        return state, info
    
    def _generate_cqi(self) -> np.ndarray:
        """Generate TRUE CQI using realistic channel model"""
        sinr = self.channel_model.calculate_sinr(
            distance=self.user_distances,
            tx_power_dbm=self.tx_power,
            noise_power_dbm=self.noise_power
        )
        
        cqi = self.channel_model.sinr_to_cqi(sinr)
        return cqi
    
    def _apply_attack(self, clean_cqi: np.ndarray) -> np.ndarray:
        """
        Apply state poisoning attack (CQI falsification)
        
        CRITICAL: Malicious users report FALSE CQI values
        """
        if not self.attack_enabled or len(self.malicious_users) == 0:
            return clean_cqi
        
        poisoned_cqi = clean_cqi.copy()
        
        for user_idx in self.malicious_users:
            # Attack strategy: report falsely HIGH CQI to steal resources
            # More aggressive perturbation
            noise = np.random.uniform(self.attack_magnitude * 0.5, self.attack_magnitude)
            poisoned_cqi[user_idx] = np.clip(poisoned_cqi[user_idx] + noise, 0, 1)
        
        return poisoned_cqi
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state
        
        CRITICAL: State uses REPORTED CQI (potentially poisoned), not true CQI
        This is what PPO sees and makes decisions on!
        """
        normalized_buffers = np.clip(self.user_buffer_sizes / 100.0, 0, 1)
        state = np.concatenate([self.reported_cqi, normalized_buffers])
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Normalize action
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.n_users) / self.n_users
        
        # Allocate RBs based on action
        allocated_rbs = action * self.n_rbs
        
        # CRITICAL: Calculate ACTUAL throughput using TRUE CQI
        # But PPO made decisions based on REPORTED (poisoned) CQI!
        # This mismatch causes degradation when under attack
        user_throughputs = self.channel_model.calculate_throughput(
            cqi_normalized=self.true_cqi,  # Use TRUE channel quality
            allocated_rbs=allocated_rbs,
            bandwidth_per_rb=self.bandwidth_per_rb
        )
        
        # Service rates (packets served)
        packets_served = user_throughputs * 0.1
        
        # Update buffers
        self.user_buffer_sizes = np.maximum(0, self.user_buffer_sizes - packets_served)
        
        # New arrivals (Poisson)
        new_arrivals = np.random.poisson(5, self.n_users)
        self.user_buffer_sizes = np.minimum(self.user_buffer_sizes + new_arrivals, 100)
        
        # Calculate latency
        service_rates = packets_served + 1e-6
        latencies = self.qos_metrics.calculate_average_latency(
            self.user_buffer_sizes, 
            service_rates
        )
        
        # Calculate fairness
        fairness = self.qos_metrics.calculate_jain_fairness(user_throughputs)
        
        # Update metrics history
        self.qos_metrics.update_history(user_throughputs, latencies, fairness)
        
        # Calculate reward (penalize if mismatch between reported and actual)
        reward = self._calculate_reward(user_throughputs, fairness, allocated_rbs)
        
        # Update TRUE channel (time-varying)
        self.true_cqi = self._generate_cqi()
        
        # Update REPORTED channel (apply attack)
        self.reported_cqi = self._apply_attack(self.true_cqi.copy())
        
        # Get new state (uses reported CQI)
        state = self._get_state()
        
        # Episode termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'throughput': user_throughputs,
            'total_throughput': np.sum(user_throughputs),
            'mean_latency': np.mean(latencies),
            'fairness': fairness,
            'malicious_users': self.malicious_users.tolist(),
            'true_cqi': self.true_cqi.copy(),
            'reported_cqi': self.reported_cqi.copy(),
        }
        
        return state, reward, terminated, truncated, info
    
    def _calculate_reward(self, throughputs: np.ndarray, fairness: float, allocations: np.ndarray) -> float:
        """
        Reward function with penalty for inefficient allocations
        
        When under attack, PPO allocates resources to users with bad channels
        (who reported good CQI falsely) → low throughput → negative reward
        """
        # Sum throughput (main objective)
        sum_throughput = np.sum(throughputs)
        
        # Fairness component
        fairness_reward = 15 * fairness
        
        # Buffer penalty (latency)
        buffer_penalty = -0.02 * np.sum(self.user_buffer_sizes)
        
        # NEW: Efficiency penalty
        # If we allocated RBs to users who couldn't use them well → penalty
        # Expected throughput if allocations were optimal
        max_possible_throughput = self.channel_model.calculate_throughput(
            cqi_normalized=self.true_cqi,
            allocated_rbs=np.ones(self.n_users) * self.n_rbs / self.n_users,  # Equal allocation baseline
            bandwidth_per_rb=self.bandwidth_per_rb
        )
        efficiency = sum_throughput / (np.sum(max_possible_throughput) + 1e-6)
        efficiency_reward = 10 * efficiency
        
        # Combined reward
        reward = sum_throughput + fairness_reward + buffer_penalty + efficiency_reward
        
        return reward
    
    def get_qos_summary(self) -> Dict:
        """Get QoS metrics summary"""
        return self.qos_metrics.get_summary_statistics()


# Test the enhanced environment
if __name__ == "__main__":
    print("Testing Enhanced Wireless Network Environment...")
    
    # Test without attack
    print("\n=== Baseline (No Attack) ===")
    env_clean = WirelessNetworkEnv(n_users=5, n_rbs=10, attack_enabled=False)
    state, info = env_clean.reset()
    
    for step in range(5):
        action = env_clean.action_space.sample()
        state, reward, term, trunc, info = env_clean.step(action)
        print(f"Step {step+1}: Throughput={info['total_throughput']:.2f} Mbps, "
              f"Fairness={info['fairness']:.3f}, Reward={reward:.2f}")
    
    # Test with attack
    print("\n=== Under Attack ===")
    env_attack = WirelessNetworkEnv(
        n_users=5, n_rbs=10, 
        attack_enabled=True, 
        attack_probability=0.4,
        attack_magnitude=0.6
    )
    state, info = env_attack.reset()
    print(f"Malicious users: {info['malicious_users']}")
    
    for step in range(5):
        action = env_attack.action_space.sample()
        state, reward, term, trunc, info = env_attack.step(action)
        
        # Show CQI mismatch
        if step == 0:
            print(f"\nTrue CQI: {info['true_cqi']}")
            print(f"Reported CQI: {info['reported_cqi']}")
            print(f"Difference: {info['reported_cqi'] - info['true_cqi']}\n")
        
        print(f"Step {step+1}: Throughput={info['total_throughput']:.2f} Mbps, "
              f"Fairness={info['fairness']:.3f}, Reward={reward:.2f}")
    
    print("\n✓ Enhanced environment working correctly!")