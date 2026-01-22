"""
Enhanced Wireless Network Environment with Black Hole Attack Support

PhD Experiment Version (Balanced Rewards):
- Tuned for Stability: Reduced traffic load (Poisson 3) to allow empty buffers.
- Tuned for Learning: Reduced buffer penalty so Agent focuses on Throughput/Fairness.
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
        
        self.malicious_users = np.array([])
        
        # Physical layer
        self.bandwidth_per_rb = 180e3
        self.tx_power = 23
        self.noise_power = -104
        
        self.channel_model = ChannelModel(model_type='urban_macro')
        self.qos_metrics = QoSMetrics()
        
        self.user_distances = None
        self.true_cqi = None
        self.reported_cqi = None
        self.user_buffer_sizes = None
        
        # Gym spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_users * 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(n_users,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Random distances
        self.user_distances = np.random.uniform(10, 500, self.n_users)
        # Empty buffers at start
        self.user_buffer_sizes = np.zeros(self.n_users) 
        
        if self.attack_enabled:
            n_malicious = int(self.n_users * self.attack_probability)
            self.malicious_users = np.random.choice(self.n_users, size=n_malicious, replace=False)
        else:
            self.malicious_users = np.array([])
            
        self.true_cqi = self._generate_cqi()
        self.reported_cqi = self._apply_attack(self.true_cqi.copy())
        
        self.qos_metrics.reset()
        return self._get_state(), {'malicious_users': self.malicious_users.tolist()}
    
    def _generate_cqi(self) -> np.ndarray:
        sinr = self.channel_model.calculate_sinr(self.user_distances, self.tx_power, self.noise_power)
        cqi = self.channel_model.sinr_to_cqi(sinr)
        
        # Sabotage malicious users' true channel
        if self.attack_enabled and len(self.malicious_users) > 0:
            for user_idx in self.malicious_users:
                cqi[user_idx] = np.random.uniform(0.0, 0.1)
        return cqi
    
    def _apply_attack(self, clean_cqi: np.ndarray) -> np.ndarray:
        if not self.attack_enabled or len(self.malicious_users) == 0:
            return clean_cqi
        
        poisoned_cqi = clean_cqi.copy()
        for user_idx in self.malicious_users:
            poisoned_cqi[user_idx] = 1.0 # Lie: Perfect channel
        return poisoned_cqi
    
    def _get_state(self) -> np.ndarray:
        normalized_buffers = np.clip(self.user_buffer_sizes / 100.0, 0, 1)
        state = np.concatenate([self.reported_cqi, normalized_buffers])
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        # Normalize Action
        action = np.clip(action, 0, 1)
        if np.sum(action) > 0: action /= np.sum(action)
        else: action = np.ones(self.n_users) / self.n_users
        
        allocated_rbs = action * self.n_rbs
        
        # Physical Throughput
        user_throughputs = self.channel_model.calculate_throughput(
            cqi_normalized=self.true_cqi,
            allocated_rbs=allocated_rbs,
            bandwidth_per_rb=self.bandwidth_per_rb
        )
        
        # Traffic Dynamics
        packets_served = user_throughputs * 1.0 
        self.user_buffer_sizes = np.maximum(0, self.user_buffer_sizes - packets_served)
        
        # Arrivals: Poisson(3) -> Reduced from 4/5 to allow buffers to empty
        new_arrivals = np.random.poisson(3, self.n_users)
        self.user_buffer_sizes = np.minimum(self.user_buffer_sizes + new_arrivals, 100)
        
        # Metrics
        service_rates = packets_served # Passed to QoS metrics (logic handled there)
        latencies = self.qos_metrics.calculate_average_latency(self.user_buffer_sizes, service_rates)
        fairness = self.qos_metrics.calculate_jain_fairness(user_throughputs)
        
        self.qos_metrics.update_history(user_throughputs, latencies, fairness)
        
        # Reward
        reward = self._calculate_reward(user_throughputs, fairness, allocated_rbs)
        
        # Next State
        self.true_cqi = self._generate_cqi()
        self.reported_cqi = self._apply_attack(self.true_cqi.copy())
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            'throughput': user_throughputs,
            'total_throughput': np.sum(user_throughputs),
            'mean_latency': np.mean(latencies),
            'fairness': fairness,
            'malicious_users': self.malicious_users.tolist()
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def _calculate_reward(self, throughputs: np.ndarray, fairness: float, allocations: np.ndarray) -> float:
        sum_throughput = np.sum(throughputs) # Expect ~20-25
        
        # Increased Fairness weight to enforce fair play in Baseline
        fairness_reward = 10.0 * fairness 
        
        # Drastically reduced buffer penalty
        # Now it's a "tie-breaker" rather than the main driver
        buffer_penalty = -0.01 * np.sum(self.user_buffer_sizes)
        
        # Discrepancy Penalty (Attack Detection)
        expected_se = self.channel_model.cqi_to_spectral_efficiency(self.reported_cqi)
        expected_mbps = (expected_se * self.bandwidth_per_rb * allocations) / 1e6
        discrepancy = np.sum(expected_mbps) - sum_throughput
        
        attack_penalty = 0
        if discrepancy > 1.5:
            # Strong penalty for falling into the trap
            attack_penalty = -2.0 * discrepancy
            
        return sum_throughput + fairness_reward + buffer_penalty + attack_penalty

    def get_qos_summary(self) -> Dict:
        return self.qos_metrics.get_summary_statistics()

# Verification
if __name__ == "__main__":
    print("Testing Balanced Network Env...")
    env = WirelessNetworkEnv(n_users=5, n_rbs=10, attack_enabled=True)
    env.reset()
    act = np.ones(5)/5
    s, r, term, trunc, i = env.step(act)
    print(f"Reward: {r:.2f}, Throughput: {i['total_throughput']:.2f}")
    print("✓ Ready.")