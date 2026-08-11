"""
Noise Utilities

Provides functions to add various types of noise to DTMF signals
for robustness testing.
"""

import numpy as np

def add_white_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to a signal at a specified SNR."""
    # Compute signal power
    signal_power = np.mean(signal ** 2)
    # Compute required noise power from SNR
    # SNR_dB = 10 * log10(P_signal / P_noise)
    noise_power = signal_power / (10 ** (snr_db / 10))
    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
