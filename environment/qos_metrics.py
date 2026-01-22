"""
QoS Metrics Calculator

Implements metrics for evaluating network performance:
- Throughput (aggregate and per-user)
- Latency (queueing delay) - FIXED for stability
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
        """
        values = np.array(values)
        if np.sum(values) == 0 or len(values) == 0:
            return 0.0
        
        numerator = np.sum(values) ** 2
        denominator = len(values) * np.sum(values ** 2)
        
        return numerator / (denominator + 1e-10)
    
    @staticmethod
    def calculate_aggregate_throughput(user_throughputs: np.ndarray) -> float:
        return np.sum(user_throughputs)
    
    @staticmethod
    def calculate_average_latency(buffer_sizes: np.ndarray,
                                  service_rates: np.ndarray) -> np.ndarray:
        """
        Calculate average queueing latency per user (Little's Law approximation)
        
        FIX: Handles near-zero service rates to avoid explosion.
        """
        # Set a minimum effective service rate to avoid division by zero (or near zero)
        # 0.1 implies that if service is 0, we estimate latency based on a slow drain rate
        # rather than infinity. This keeps metrics readable.
        safe_service_rates = np.maximum(service_rates, 0.1)
        
        latency = buffer_sizes / safe_service_rates
        
        # Optional: Cap latency reporting to avoid skewing averages with one stuck user
        # latency = np.minimum(latency, 1000.0) 
        
        return latency
    
    @staticmethod
    def calculate_packet_loss_rate(buffer_sizes: np.ndarray,
                                   max_buffer_size: float = 100.0) -> float:
        overflow_count = np.sum(buffer_sizes >= max_buffer_size)
        return overflow_count / len(buffer_sizes)
    
    @staticmethod
    def calculate_spectral_efficiency(throughput: float,
                                     bandwidth: float) -> float:
        return throughput / (bandwidth + 1e-10)
    
    @staticmethod
    def calculate_user_satisfaction(throughputs: np.ndarray,
                                   min_required_throughput: float = 1.0) -> float:
        satisfied_users = np.sum(throughputs >= min_required_throughput)
        return satisfied_users / len(throughputs)
    
    @staticmethod
    def calculate_5th_percentile_throughput(throughputs: np.ndarray) -> float:
        return np.percentile(throughputs, 5)
    
    def update_history(self, 
                      throughputs: np.ndarray,
                      latencies: np.ndarray,
                      fairness: float):
        self.throughput_history.append(throughputs.copy())
        self.latency_history.append(latencies.copy())
        self.fairness_history.append(fairness)
    
    def get_summary_statistics(self) -> Dict[str, float]:
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
        self.throughput_history = []
        self.latency_history = []
        self.fairness_history = []
    
    @staticmethod
    def compare_scenarios(baseline_metrics: Dict,
                         attack_metrics: Dict,
                         defense_metrics: Dict) -> Dict[str, Dict[str, float]]:
        def percent_change(old, new):
            if old == 0: return 0.0
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