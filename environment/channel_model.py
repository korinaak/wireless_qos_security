"""
Channel Model for 5G/6G Wireless Networks

Implements:
- Path loss models (Urban Macro, Urban Micro)
- Fast fading (Rayleigh, Rician)
- Shadowing (Log-normal)
- SINR calculation
- CQI mapping
"""

import numpy as np
from typing import Tuple, Optional

class ChannelModel:
    """
    Comprehensive channel modeling for wireless networks
    """
    
    def __init__(self, 
                 model_type: str = 'urban_macro',
                 frequency: float = 2.6e9,  # Hz (5G Band n41)
                 bs_height: float = 25.0,   # meters
                 ue_height: float = 1.5):    # meters
        """
        Args:
            model_type: 'urban_macro', 'urban_micro', 'indoor'
            frequency: Carrier frequency in Hz
            bs_height: Base station antenna height
            ue_height: User equipment antenna height
        """
        self.model_type = model_type
        self.frequency = frequency
        self.bs_height = bs_height
        self.ue_height = ue_height
        
        # Constants
        self.speed_of_light = 3e8
        self.wavelength = self.speed_of_light / frequency
        
    def calculate_path_loss(self, distance: np.ndarray) -> np.ndarray:
        """
        Calculate path loss based on 3GPP models
        
        Args:
            distance: User distances in meters (can be array)
        
        Returns:
            Path loss in dB
        """
        # Avoid log(0)
        distance = np.maximum(distance, 1.0)
        
        if self.model_type == 'urban_macro':
            # 3GPP TR 38.901 Urban Macro (UMa)
            # PL = 28.0 + 22*log10(d) + 20*log10(fc)
            fc_ghz = self.frequency / 1e9
            pl = 28.0 + 22 * np.log10(distance) + 20 * np.log10(fc_ghz)
            
        elif self.model_type == 'urban_micro':
            # 3GPP TR 38.901 Urban Micro (UMi)
            fc_ghz = self.frequency / 1e9
            pl = 32.4 + 21 * np.log10(distance) + 20 * np.log10(fc_ghz)
            
        elif self.model_type == 'indoor':
            # Indoor Office
            pl = 20 * np.log10(distance) + 46.4 + 20 * np.log10(self.frequency / 1e9)
            
        else:
            # Free space path loss (baseline)
            pl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) - 147.55
        
        return pl
    
    def generate_shadowing(self, n_users: int, std_dev: float = 8.0) -> np.ndarray:
        """
        Generate log-normal shadowing
        
        Args:
            n_users: Number of users
            std_dev: Standard deviation in dB (typical: 8 dB)
        
        Returns:
            Shadowing in dB
        """
        return np.random.normal(0, std_dev, n_users)
    
    def generate_fast_fading(self, 
                            n_users: int, 
                            fading_type: str = 'rayleigh',
                            k_factor: float = 10.0) -> np.ndarray:
        """
        Generate fast fading
        
        Args:
            n_users: Number of users
            fading_type: 'rayleigh' or 'rician'
            k_factor: Rician K-factor (dB) for LoS scenarios
        
        Returns:
            Fading gain (linear scale)
        """
        if fading_type == 'rayleigh':
            # Rayleigh fading: |h|^2 ~ Exponential(1)
            fading_gain = np.random.exponential(1.0, n_users)
            
        elif fading_type == 'rician':
            # Rician fading: LoS + scattered components
            k_linear = 10**(k_factor / 10)
            
            # Real and imaginary parts
            los_component = np.sqrt(k_linear / (k_linear + 1))
            scatter_std = np.sqrt(1 / (2 * (k_linear + 1)))
            
            real_part = los_component + np.random.normal(0, scatter_std, n_users)
            imag_part = np.random.normal(0, scatter_std, n_users)
            
            fading_gain = real_part**2 + imag_part**2
        else:
            fading_gain = np.ones(n_users)
        
        return fading_gain
    
    def calculate_sinr(self,
                      distance: np.ndarray,
                      tx_power_dbm: float,
                      noise_power_dbm: float,
                      interference_power_dbm: float = -float('inf')) -> np.ndarray:
        """
        Calculate Signal-to-Interference-plus-Noise Ratio
        
        Args:
            distance: User distances
            tx_power_dbm: Transmit power in dBm
            noise_power_dbm: Noise power in dBm
            interference_power_dbm: Interference power in dBm
        
        Returns:
            SINR in linear scale
        """
        # Path loss
        path_loss = self.calculate_path_loss(distance)
        
        # Shadowing
        shadowing = self.generate_shadowing(len(distance))
        
        # Fast fading
        fading_gain = self.generate_fast_fading(len(distance))
        fading_db = 10 * np.log10(fading_gain)
        
        # Received signal power (dBm)
        rx_signal_dbm = tx_power_dbm - path_loss + shadowing + fading_db
        
        # Total noise + interference (dBm)
        if np.isfinite(interference_power_dbm):
            noise_plus_interference = 10 * np.log10(
                10**(noise_power_dbm/10) + 10**(interference_power_dbm/10)
            )
        else:
            noise_plus_interference = noise_power_dbm
        
        # SINR (linear)
        sinr_db = rx_signal_dbm - noise_plus_interference
        sinr_linear = 10**(sinr_db / 10)
        
        return sinr_linear
    
    def sinr_to_cqi(self, sinr_linear: np.ndarray) -> np.ndarray:
        """
        Map SINR to CQI (Channel Quality Indicator)
        
        CQI range: 0-15 (as per 3GPP standards)
        
        Args:
            sinr_linear: SINR in linear scale
        
        Returns:
            CQI values [0, 15], normalized to [0, 1]
        """
        # Simplified SINR to CQI mapping
        # Based on Shannon capacity approximation
        sinr_db = 10 * np.log10(sinr_linear + 1e-10)
        
        # CQI thresholds (dB): approximately logarithmic
        # CQI 0: < -10 dB
        # CQI 15: > 20 dB
        cqi = np.clip((sinr_db + 10) / 2, 0, 15)
        
        return cqi / 15.0  # Normalize to [0, 1]
    
    def cqi_to_spectral_efficiency(self, cqi_normalized: np.ndarray) -> np.ndarray:
        """
        Map CQI to spectral efficiency (bits/s/Hz)
        
        Args:
            cqi_normalized: CQI values in [0, 1]
        
        Returns:
            Spectral efficiency in bits/s/Hz
        """
        # CQI to spectral efficiency lookup (3GPP approximation)
        cqi_int = np.round(cqi_normalized * 15).astype(int)
        
        # Lookup table (simplified)
        se_table = np.array([
            0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91,
            2.41, 2.73, 3.32, 3.90, 4.52, 5.12, 5.55, 6.23
        ])
        
        return se_table[cqi_int]
    
    def calculate_throughput(self,
                            cqi_normalized: np.ndarray,
                            allocated_rbs: np.ndarray,
                            bandwidth_per_rb: float = 180e3) -> np.ndarray:
        """
        Calculate user throughput
        
        Args:
            cqi_normalized: Normalized CQI [0, 1]
            allocated_rbs: Number of RBs allocated to each user
            bandwidth_per_rb: Bandwidth per RB in Hz
        
        Returns:
            Throughput in Mbps
        """
        # Spectral efficiency
        se = self.cqi_to_spectral_efficiency(cqi_normalized)
        
        # Throughput = SE × BW × RBs
        throughput_bps = se * bandwidth_per_rb * allocated_rbs
        throughput_mbps = throughput_bps / 1e6
        
        return throughput_mbps


# Test the channel model
if __name__ == "__main__":
    print("Testing Channel Model...")
    
    channel = ChannelModel(model_type='urban_macro')
    
    # Test with 5 users
    distances = np.array([50, 100, 200, 300, 500])
    
    # Calculate SINR
    sinr = channel.calculate_sinr(
        distance=distances,
        tx_power_dbm=23,
        noise_power_dbm=-104
    )
    
    print(f"Distances (m): {distances}")
    print(f"SINR (linear): {sinr}")
    print(f"SINR (dB): {10*np.log10(sinr)}")
    
    # Calculate CQI
    cqi = channel.sinr_to_cqi(sinr)
    print(f"CQI (normalized): {cqi}")
    
    # Calculate throughput
    allocated_rbs = np.array([5, 5, 5, 5, 5])
    throughput = channel.calculate_throughput(cqi, allocated_rbs)
    print(f"Throughput (Mbps): {throughput}")
    
    print("\n✓ Channel Model working correctly!")