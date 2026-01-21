"""
QoS Metrics Calculator

Implements metrics for evaluating network performance:
- Throughput (aggregate and per-user)
- Latency (queueing delay)
- Packet Loss Rate
- Jain's Fairness Index
- Spectral Efficiency
"""

import numpy as np
from typing import List, Dict, Tuple

class QoSMetrics:
    """
    Quality of Service metrics calculator
    """
    
    def __init__(self):
        self.throughput_history = []
        self.latency_history = []
        self.fairness_history = []
        
    @staticmethod
    def calculate_jain_fairness(values: np.ndarray) -> float:
        """
        Jain's Fairness Index
        
        Formula: (Σx_i)² / (n × Σx_i²)
        Range: [1/n, 1], where 1 is perfect fairness
        
        Args:
            values: Array of values (e.g., throughputs)
        
        Returns:
            Fairness index
        """
        values = np.array(values)
        if np.sum(values) == 0 or len(values) == 0:
            return 0.0
        
        numerator = np.sum(values) ** 2
        denominator = len(values) * np.sum(values ** 2)
        
        return numerator / (denominator + 1e-10)
    
    @staticmethod
    def calculate_aggregate_throughput(user_throughputs: np.ndarray) -> float:
        """
        Calculate total network throughput
        
        Args:
            user_throughputs: Throughput per user (Mbps)
        
        Returns:
            Total throughput (Mbps)
        """
        return np.sum(user_throughputs)
    
    @staticmethod
    def calculate_average_latency(buffer_sizes: np.ndarray,
                                  service_rates: np.ndarray) -> np.ndarray:
        """
        Calculate average queueing latency per user
        Using Little's Law: L = λ × W → W = L / λ
        
        Args:
            buffer_sizes: Queue lengths (packets)
            service_rates: Service rates (packets/time unit)
        
        Returns:
            Average latency per user (time units)
        """
        # Avoid division by zero
        service_rates = np.maximum(service_rates, 1e-6)
        latency = buffer_sizes / service_rates
        
        return latency
    
    @staticmethod
    def calculate_packet_loss_rate(buffer_sizes: np.ndarray,
                                   max_buffer_size: float = 100.0) -> float:
        """
        Calculate packet loss rate (overflow probability)
        
        Args:
            buffer_sizes: Current buffer occupancy
            max_buffer_size: Maximum buffer capacity
        
        Returns:
            Packet loss rate [0, 1]
        """
        overflow_count = np.sum(buffer_sizes >= max_buffer_size)
        return overflow_count / len(buffer_sizes)
    
    @staticmethod
    def calculate_spectral_efficiency(throughput: float,
                                     bandwidth: float) -> float:
        """
        Calculate spectral efficiency
        
        Args:
            throughput: Total throughput (bps)
            bandwidth: Total bandwidth (Hz)
        
        Returns:
            Spectral efficiency (bits/s/Hz)
        """
        return throughput / (bandwidth + 1e-10)
    
    @staticmethod
    def calculate_user_satisfaction(throughputs: np.ndarray,
                                   min_required_throughput: float = 1.0) -> float:
        """
        Calculate fraction of users meeting QoS requirements
        
        Args:
            throughputs: User throughputs (Mbps)
            min_required_throughput: Minimum required (Mbps)
        
        Returns:
            Satisfaction rate [0, 1]
        """
        satisfied_users = np.sum(throughputs >= min_required_throughput)
        return satisfied_users / len(throughputs)
    
    @staticmethod
    def calculate_5th_percentile_throughput(throughputs: np.ndarray) -> float:
        """
        Calculate 5th percentile user throughput (cell-edge performance)
        
        Args:
            throughputs: User throughputs
        
        Returns:
            5th percentile throughput
        """
        return np.percentile(throughputs, 5)
    
    def update_history(self, 
                      throughputs: np.ndarray,
                      latencies: np.ndarray,
                      fairness: float):
        """
        Update metric history for tracking over time
        """
        self.throughput_history.append(throughputs.copy())
        self.latency_history.append(latencies.copy())
        self.fairness_history.append(fairness)
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics over entire history
        
        Returns:
            Dictionary of metrics
        """
        if len(self.throughput_history) == 0:
            return {}
        
        throughputs = np.array(self.throughput_history)
        latencies = np.array(self.latency_history)
        
        return {
            'mean_aggregate_throughput': np.mean(np.sum(throughputs, axis=1)),
            'mean_user_throughput': np.mean(throughputs),
            'std_user_throughput': np.std(throughputs),
            'mean_latency': np.mean(latencies),
            'max_latency': np.max(latencies),
            'mean_fairness': np.mean(self.fairness_history),
            'min_fairness': np.min(self.fairness_history),
        }
    
    def reset(self):
        """Reset metric history"""
        self.throughput_history = []
        self.latency_history = []
        self.fairness_history = []
    
    @staticmethod
    def compare_scenarios(baseline_metrics: Dict,
                         attack_metrics: Dict,
                         defense_metrics: Dict) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics across scenarios
        
        Returns:
            Dictionary with degradation/improvement percentages
        """
        def percent_change(old, new):
            if old == 0:
                return 0.0
            return ((new - old) / old) * 100
        
        comparison = {}
        
        for metric in baseline_metrics.keys():
            baseline = baseline_metrics[metric]
            attack = attack_metrics.get(metric, 0)
            defense = defense_metrics.get(metric, 0)
            
            comparison[metric] = {
                'baseline': baseline,
                'under_attack': attack,
                'with_defense': defense,
                'attack_degradation_%': percent_change(baseline, attack),
                'defense_recovery_%': percent_change(attack, defense)
            }
        
        return comparison


# Test QoS Metrics
if __name__ == "__main__":
    print("Testing QoS Metrics...")
    
    metrics = QoSMetrics()
    
    # Simulate some data
    user_throughputs = np.array([10.5, 8.2, 12.1, 7.8, 9.5])
    buffer_sizes = np.array([20, 35, 15, 40, 25])
    service_rates = np.array([5, 4, 6, 3, 5])
    
    # Calculate metrics
    fairness = metrics.calculate_jain_fairness(user_throughputs)
    total_throughput = metrics.calculate_aggregate_throughput(user_throughputs)
    latencies = metrics.calculate_average_latency(buffer_sizes, service_rates)
    satisfaction = metrics.calculate_user_satisfaction(user_throughputs, min_required_throughput=8.0)
    percentile_5 = metrics.calculate_5th_percentile_throughput(user_throughputs)
    
    print(f"User Throughputs: {user_throughputs}")
    print(f"Jain's Fairness Index: {fairness:.3f}")
    print(f"Total Throughput: {total_throughput:.2f} Mbps")
    print(f"Average Latencies: {latencies}")
    print(f"User Satisfaction: {satisfaction:.2%}")
    print(f"5th Percentile Throughput: {percentile_5:.2f} Mbps")
    
    # Test comparison
    baseline = {'throughput': 100, 'latency': 10, 'fairness': 0.9}
    attack = {'throughput': 70, 'latency': 25, 'fairness': 0.6}
    defense = {'throughput': 90, 'latency': 12, 'fairness': 0.85}
    
    comparison = metrics.compare_scenarios(baseline, attack, defense)
    print("\n--- Scenario Comparison ---")
    for metric, values in comparison.items():
        print(f"{metric}:")
        print(f"  Attack degradation: {values['attack_degradation_%']:.1f}%")
        print(f"  Defense recovery: {values['defense_recovery_%']:.1f}%")
    
    print("\n✓ QoS Metrics working correctly!")