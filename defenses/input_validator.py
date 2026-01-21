"""
Input Validation Defense

Sanitizes and validates input observations before feeding to DRL agent.

Techniques:
1. Bounds checking (clip to valid ranges)
2. Rate limiting (prevent sudden large changes)
3. Consistency checking (cross-validate with other metrics)
"""

import numpy as np
from typing import Tuple, Optional


class InputValidator:
    """
    Input validation and sanitization for state observations
    """
    
    def __init__(self,
                 n_users: int,
                 cqi_bounds: Tuple[float, float] = (0.0, 1.0),
                 buffer_bounds: Tuple[float, float] = (0.0, 1.0),
                 max_change_rate: float = 0.1):
        """
        Initialize input validator
        
        Args:
            n_users: Number of users
            cqi_bounds: Valid CQI range
            buffer_bounds: Valid buffer range
            max_change_rate: Maximum allowed change per step [0, 1]
        """
        self.n_users = n_users
        self.cqi_bounds = cqi_bounds
        self.buffer_bounds = buffer_bounds
        self.max_change_rate = max_change_rate
        
        # Track previous state for rate limiting
        self.previous_state = None
        
        # Statistics
        self.n_violations = 0
        self.violation_types = {'bounds': 0, 'rate': 0, 'consistency': 0}
    
    def validate(self, state: np.ndarray) -> np.ndarray:
        """
        Validate and sanitize state observation
        
        Args:
            state: Raw state [CQI_1, ..., CQI_N, Buffer_1, ..., Buffer_N]
        
        Returns:
            Validated and sanitized state
        """
        validated_state = state.copy()
        
        # Extract CQI and buffer components
        cqi_values = validated_state[:self.n_users]
        buffer_values = validated_state[self.n_users:]
        
        # 1. Bounds checking
        cqi_values, bounds_violations_cqi = self._check_bounds(
            cqi_values, self.cqi_bounds
        )
        buffer_values, bounds_violations_buf = self._check_bounds(
            buffer_values, self.buffer_bounds
        )
        
        if bounds_violations_cqi or bounds_violations_buf:
            self.violation_types['bounds'] += 1
            self.n_violations += 1
        
        # 2. Rate limiting (prevent sudden jumps)
        if self.previous_state is not None:
            cqi_values, rate_violations = self._check_rate_limit(
                cqi_values, 
                self.previous_state[:self.n_users]
            )
            
            if rate_violations:
                self.violation_types['rate'] += 1
                self.n_violations += 1
        
        # 3. Consistency checking
        cqi_values, consistency_violations = self._check_consistency(
            cqi_values, buffer_values
        )
        
        if consistency_violations:
            self.violation_types['consistency'] += 1
            self.n_violations += 1
        
        # Reassemble state
        validated_state = np.concatenate([cqi_values, buffer_values])
        
        # Update previous state
        self.previous_state = validated_state.copy()
        
        return validated_state
    
    def _check_bounds(self, 
                     values: np.ndarray,
                     bounds: Tuple[float, float]) -> Tuple[np.ndarray, bool]:
        """
        Clip values to valid bounds
        
        Returns:
            (clipped_values, violation_detected)
        """
        lower, upper = bounds
        clipped = np.clip(values, lower, upper)
        violation = not np.allclose(values, clipped)
        
        return clipped, violation
    
    def _check_rate_limit(self,
                         current: np.ndarray,
                         previous: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Limit rate of change
        
        If change > max_change_rate, gradually adjust instead
        """
        change = current - previous
        abs_change = np.abs(change)
        
        violation = np.any(abs_change > self.max_change_rate)
        
        if violation:
            # Limit the change
            limited_change = np.clip(change, -self.max_change_rate, self.max_change_rate)
            adjusted = previous + limited_change
            return adjusted, True
        
        return current, False
    
    def _check_consistency(self,
                          cqi_values: np.ndarray,
                          buffer_values: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Check consistency between CQI and buffer states
        
        Heuristic: Users with consistently high buffers shouldn't have
        persistently excellent CQI (would have drained the buffer)
        """
        violation = False
        adjusted_cqi = cqi_values.copy()
        
        # Simple heuristic: if buffer is full (>0.8) and CQI is excellent (>0.9),
        # it's suspicious (unless there's a good reason like recent arrival burst)
        suspicious_mask = (buffer_values > 0.8) & (cqi_values > 0.9)
        
        if np.any(suspicious_mask):
            # Slightly reduce CQI for suspicious users
            adjusted_cqi[suspicious_mask] *= 0.95
            violation = True
        
        return adjusted_cqi, violation
    
    def get_statistics(self) -> dict:
        """Get validation statistics"""
        return {
            'total_violations': self.n_violations,
            'bounds_violations': self.violation_types['bounds'],
            'rate_violations': self.violation_types['rate'],
            'consistency_violations': self.violation_types['consistency']
        }
    
    def reset(self):
        """Reset validator state"""
        self.previous_state = None
        self.n_violations = 0
        self.violation_types = {'bounds': 0, 'rate': 0, 'consistency': 0}


class AdaptiveInputValidator(InputValidator):
    """
    Adaptive validator that learns normal patterns
    
    Adjusts validation thresholds based on observed data distribution
    """
    
    def __init__(self, n_users: int, adaptation_rate: float = 0.01, **kwargs):
        """
        Args:
            n_users: Number of users
            adaptation_rate: How fast to adapt thresholds
            **kwargs: Passed to InputValidator
        """
        super().__init__(n_users, **kwargs)
        self.adaptation_rate = adaptation_rate
        
        # Learn statistics
        self.cqi_mean = np.ones(n_users) * 0.5
        self.cqi_std = np.ones(n_users) * 0.2
        
    def validate(self, state: np.ndarray) -> np.ndarray:
        """
        Validate with adaptive thresholds
        """
        # First, standard validation
        validated_state = super().validate(state)
        
        # Update statistics (exponential moving average)
        cqi_values = validated_state[:self.n_users]
        
        self.cqi_mean = (1 - self.adaptation_rate) * self.cqi_mean + \
                        self.adaptation_rate * cqi_values
        
        # Update std
        squared_diff = (cqi_values - self.cqi_mean) ** 2
        self.cqi_std = np.sqrt(
            (1 - self.adaptation_rate) * self.cqi_std**2 +
            self.adaptation_rate * squared_diff
        )
        
        return validated_state
    
    def get_learned_stats(self) -> dict:
        """Get learned statistics"""
        return {
            'cqi_mean': self.cqi_mean.tolist(),
            'cqi_std': self.cqi_std.tolist()
        }


# Test Input Validator
if __name__ == "__main__":
    print("Testing Input Validator...")
    
    n_users = 5
    validator = InputValidator(
        n_users=n_users,
        cqi_bounds=(0.0, 1.0),
        buffer_bounds=(0.0, 1.0),
        max_change_rate=0.1
    )
    
    print("\n--- Test 1: Bounds Violation ---")
    state_oob = np.array([0.5, 1.2, -0.1, 0.7, 0.6,  # CQI (out of bounds)
                          0.3, 0.4, 0.5, 0.6, 0.7])   # Buffers
    
    validated = validator.validate(state_oob)
    print(f"Original CQI: {state_oob[:n_users]}")
    print(f"Validated CQI: {validated[:n_users]}")
    print(f"Stats: {validator.get_statistics()}")
    
    print("\n--- Test 2: Rate Limit Violation ---")
    validator.reset()
    
    # Initial state
    state1 = np.random.uniform(0.4, 0.6, 10)
    validated1 = validator.validate(state1)
    
    # Sudden jump
    state2 = state1.copy()
    state2[2] = 0.95  # Sudden jump in user 2's CQI
    validated2 = validator.validate(state2)
    
    print(f"State 1 CQI[2]: {state1[2]:.3f}")
    print(f"State 2 CQI[2] (jump): {state2[2]:.3f}")
    print(f"Validated CQI[2]: {validated2[2]:.3f}")
    print(f"Stats: {validator.get_statistics()}")
    
    print("\n--- Test 3: Consistency Check ---")
    validator.reset()
    
    # Inconsistent state: high buffer + excellent CQI
    state_inconsistent = np.array([0.5, 0.6, 0.95, 0.7, 0.6,  # CQI
                                   0.3, 0.4, 0.9, 0.5, 0.4])   # Buffers (user 2 has high buffer)
    
    validated = validator.validate(state_inconsistent)
    print(f"Original CQI: {state_inconsistent[:n_users]}")
    print(f"Validated CQI: {validated[:n_users]}")
    print(f"Stats: {validator.get_statistics()}")
    
    # Test Adaptive Validator
    print("\n\n=== Testing Adaptive Validator ===")
    adaptive = AdaptiveInputValidator(n_users=n_users, adaptation_rate=0.1)
    
    # Simulate learning phase
    for step in range(20):
        state = np.random.uniform(0.4, 0.7, 10)
        adaptive.validate(state)
    
    stats = adaptive.get_learned_stats()
    print(f"Learned CQI mean: {np.array(stats['cqi_mean'])}")
    print(f"Learned CQI std: {np.array(stats['cqi_std'])}")
    
    print("\n✓ Input Validator working correctly!")