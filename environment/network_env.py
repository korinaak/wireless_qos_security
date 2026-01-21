"""
Enhanced Wireless Network Environment with Attack Support

Integrates ChannelModel and QoSMetrics for realistic simulation
Supports poisoning attacks on state observations
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
        self.malicious_users = np.random.choice(
            n_users, 
            size=int(n_users * attack_probability),
            replace=False
        )
        
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
        self.current_cqi = None
        self.user_buffer_sizes = None
        
        # Gym spaces
        # State: [CQI_1, ..., CQI_N, Buffer_1, ..., Buffer_N]
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(n_users * 2,),
            dtype=np.float32
        )
        
        # Action: Resource allocation percentages
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
        
        # Generate initial channel conditions
        self.current_cqi = self._generate_cqi()
        
        # Re-select malicious users for new episode
        if self.attack_enabled:
            self.malicious_users = np.random.choice(
                self.n_users,
                size=int(self.n_users * self.attack_probability),
                replace=False
            )
        
        # Reset metrics
        self.qos_metrics.reset()
        
        state = self._get_state()
        info = {'malicious_users': self.malicious_users.tolist()}
        
        return state, info
    
    def _generate_cqi(self) -> np.ndarray:
        """Generate CQI using realistic channel model"""
        sinr = self.channel_model.calculate_sinr(
            distance=self.user_distances,
            tx_power_dbm=self.tx_power,
            noise_power_dbm=self.noise_power
        )
        
        cqi = self.channel_model.sinr_to_cqi(sinr)
        return cqi
    
    def _apply_attack(self, clean_state: np.ndarray) -> np.ndarray:
        """
        Apply state poisoning attack (CQI falsification)
        
        Malicious users report false CQI values
        """
        if not self.attack_enabled:
            return clean_state
        
        poisoned_state = clean_state.copy()
        
        # Extract CQI part (first n_users elements)
        cqi_part = poisoned_state[:self.n_users]
        
        for user_idx in self.malicious_users:
            # Attack strategy: report falsely high CQI to steal resources
            noise = np.random.uniform(0, self.attack_magnitude)
            cqi_part[user_idx] = np.clip(cqi_part[user_idx] + noise, 0, 1)
        
        poisoned_state[:self.n_users] = cqi_part
        
        return poisoned_state
    
    def _get_state(self) -> np.ndarray:
        """Get current state with potential poisoning"""
        # Clean state
        normalized_buffers = np.clip(self.user_buffer_sizes / 100.0, 0, 1)
        clean_state = np.concatenate([self.current_cqi, normalized_buffers])
        
        # Apply attack if enabled
        observed_state = self._apply_attack(clean_state)
        
        return observed_state.astype(np.float32)
    
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
        
        # Allocate RBs
        allocated_rbs = action * self.n_rbs
        
        # Calculate throughput using channel model
        user_throughputs = self.channel_model.calculate_throughput(
            cqi_normalized=self.current_cqi,
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
        
        # Calculate reward
        reward = self._calculate_reward(user_throughputs, fairness)
        
        # Update channel (time-varying)
        self.current_cqi = self._generate_cqi()
        
        # Get new state
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
        }
        
        return state, reward, terminated, truncated, info
    
    def _calculate_reward(self, throughputs: np.ndarray, fairness: float) -> float:
        """
        Reward function: balance throughput, fairness, and latency
        """
        # Sum throughput
        sum_throughput = np.sum(throughputs)
        
        # Fairness component
        fairness_reward = 10 * fairness
        
        # Buffer penalty (latency)
        buffer_penalty = -0.01 * np.sum(self.user_buffer_sizes)
        
        # Combined reward
        reward = sum_throughput + fairness_reward + buffer_penalty
        
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
              f"Fairness={info['fairness']:.3f}, Latency={info['mean_latency']:.2f}")
    
    # Test with attack
    print("\n=== Under Attack ===")
    env_attack = WirelessNetworkEnv(
        n_users=5, n_rbs=10, 
        attack_enabled=True, 
        attack_probability=0.4,
        attack_magnitude=0.5
    )
    state, info = env_attack.reset()
    print(f"Malicious users: {info['malicious_users']}")
    
    for step in range(5):
        action = env_attack.action_space.sample()
        state, reward, term, trunc, info = env_attack.step(action)
        print(f"Step {step+1}: Throughput={info['total_throughput']:.2f} Mbps, "
              f"Fairness={info['fairness']:.3f}")
    
    print("\n✓ Enhanced environment working correctly!")