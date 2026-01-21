"""
Anomaly Detection Defense

Detects and filters anomalous CQI reports using statistical methods.

Methods:
1. Z-score based detection (statistical outliers)
2. Moving average deviation
3. Interquartile Range (IQR) method
"""

import numpy as np
from collections import deque
from typing import Tuple, List


class AnomalyDetector:
    """
    Statistical anomaly detector for CQI reports
    """
    
    def __init__(self,
                 n_users: int,
                 method: str = 'zscore',
                 threshold: float = 2.5,
                 window_size: int = 10):
        """
        Initialize anomaly detector
        
        Args:
            n_users: Number of users
            method: 'zscore', 'moving_avg', or 'iqr'
            threshold: Detection threshold (std devs for zscore)
            window_size: Historical window size
        """
        self.n_users = n_users
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        
        # Historical CQI data for each user
        self.cqi_history = [deque(maxlen=window_size) for _ in range(n_users)]
        
        # Statistics
        self.n_detections = 0
        self.detected_users = []
    
    def update_history(self, cqi_values: np.ndarray):
        """
        Update CQI history
        
        Args:
            cqi_values: Current CQI observations
        """
        for i, cqi in enumerate(cqi_values):
            self.cqi_history[i].append(cqi)
    
    def detect_anomalies(self, cqi_values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Detect anomalous CQI reports
        
        Args:
            cqi_values: Current CQI values [0, 1]
        
        Returns:
            Tuple of (filtered_cqi, anomalous_user_indices)
        """
        anomalous_users = []
        filtered_cqi = cqi_values.copy()
        
        for i in range(self.n_users):
            if len(self.cqi_history[i]) < 3:
                # Not enough history yet
                continue
            
            current_cqi = cqi_values[i]
            history = np.array(self.cqi_history[i])
            
            is_anomalous = False
            
            if self.method == 'zscore':
                is_anomalous = self._zscore_detection(current_cqi, history)
            
            elif self.method == 'moving_avg':
                is_anomalous = self._moving_avg_detection(current_cqi, history)
            
            elif self.method == 'iqr':
                is_anomalous = self._iqr_detection(current_cqi, history)
            
            if is_anomalous:
                anomalous_users.append(i)
                # Replace with median of history
                filtered_cqi[i] = np.median(history)
                self.n_detections += 1
        
        self.detected_users = anomalous_users
        
        # Update history with filtered values
        self.update_history(filtered_cqi)
        
        return filtered_cqi, anomalous_users
    
    def _zscore_detection(self, value: float, history: np.ndarray) -> bool:
        """
        Z-score based anomaly detection
        
        Anomaly if |value - mean| > threshold * std
        """
        mean = np.mean(history)
        std = np.std(history)
        
        if std < 1e-6:  # Avoid division by zero
            return False
        
        z_score = abs(value - mean) / std
        return z_score > self.threshold
    
    def _moving_avg_detection(self, value: float, history: np.ndarray) -> bool:
        """
        Moving average deviation detection
        """
        moving_avg = np.mean(history)
        deviation = abs(value - moving_avg)
        
        # Anomaly if deviation > threshold * historical std
        historical_std = np.std(history)
        return deviation > self.threshold * historical_std
    
    def _iqr_detection(self, value: float, history: np.ndarray) -> bool:
        """
        Interquartile Range (IQR) method
        
        Anomaly if value is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        """
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        return value < lower_bound or value > upper_bound
    
    def get_statistics(self) -> dict:
        """Get detection statistics"""
        return {
            'total_detections': self.n_detections,
            'recent_anomalous_users': self.detected_users,
            'detection_method': self.method,
            'threshold': self.threshold
        }
    
    def reset(self):
        """Reset detector state"""
        self.cqi_history = [deque(maxlen=self.window_size) for _ in range(self.n_users)]
        self.n_detections = 0
        self.detected_users = []


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detectors
    
    Combines multiple detection methods for higher accuracy
    """
    
    def __init__(self, n_users: int, voting_threshold: int = 2):
        """
        Args:
            n_users: Number of users
            voting_threshold: Minimum votes to flag as anomaly
        """
        self.n_users = n_users
        self.voting_threshold = voting_threshold
        
        # Create ensemble of detectors
        self.detectors = [
            AnomalyDetector(n_users, method='zscore', threshold=2.5),
            AnomalyDetector(n_users, method='moving_avg', threshold=2.0),
            AnomalyDetector(n_users, method='iqr', threshold=1.5)
        ]
    
    def detect_anomalies(self, cqi_values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Ensemble detection with voting
        """
        votes = np.zeros(self.n_users)
        
        # Each detector votes
        for detector in self.detectors:
            _, anomalous = detector.detect_anomalies(cqi_values)
            for user_idx in anomalous:
                votes[user_idx] += 1
        
        # Flag users with enough votes
        anomalous_users = np.where(votes >= self.voting_threshold)[0].tolist()
        
        # Filter anomalous values
        filtered_cqi = cqi_values.copy()
        for user_idx in anomalous_users:
            # Replace with median from history
            histories = [np.array(d.cqi_history[user_idx]) for d in self.detectors]
            all_history = np.concatenate([h for h in histories if len(h) > 0])
            if len(all_history) > 0:
                filtered_cqi[user_idx] = np.median(all_history)
        
        return filtered_cqi, anomalous_users
    
    def reset(self):
        """Reset all detectors"""
        for detector in self.detectors:
            detector.reset()


# Test Anomaly Detector
if __name__ == "__main__":
    print("Testing Anomaly Detector...")
    
    n_users = 5
    detector = AnomalyDetector(
        n_users=n_users,
        method='zscore',
        threshold=2.5,
        window_size=10
    )
    
    print("\n--- Simulating Normal + Anomalous CQI Reports ---")
    
    # Simulate normal CQI values
    for step in range(15):
        if step < 10:
            # Normal values
            cqi = np.random.uniform(0.5, 0.7, n_users)
            print(f"Step {step+1}: Normal CQI = {cqi}")
        else:
            # Inject anomaly in user 2
            cqi = np.random.uniform(0.5, 0.7, n_users)
            cqi[2] = 0.95  # Anomalous spike
            print(f"Step {step+1}: CQI with ANOMALY = {cqi}")
        
        filtered, anomalous = detector.detect_anomalies(cqi)
        
        if len(anomalous) > 0:
            print(f"  → DETECTED anomalies in users: {anomalous}")
            print(f"  → Filtered CQI = {filtered}")
    
    stats = detector.get_statistics()
    print(f"\nDetection Statistics: {stats}")
    
    # Test ensemble
    print("\n\n=== Testing Ensemble Detector ===")
    ensemble = EnsembleAnomalyDetector(n_users=n_users, voting_threshold=2)
    
    # Build history
    for _ in range(10):
        cqi = np.random.uniform(0.5, 0.7, n_users)
        ensemble.detect_anomalies(cqi)
    
    # Inject strong anomaly
    cqi_anomalous = np.array([0.6, 0.65, 0.98, 0.55, 0.62])  # User 2 anomalous
    filtered, detected = ensemble.detect_anomalies(cqi_anomalous)
    
    print(f"Original CQI: {cqi_anomalous}")
    print(f"Detected anomalies: {detected}")
    print(f"Filtered CQI: {filtered}")
    
    print("\n✓ Anomaly Detector working correctly!")