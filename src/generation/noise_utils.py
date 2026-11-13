"""
Noise Utilities (Backward Compatibility Wrapper)
=================================================
Delegates to the standard `src.noise.NoiseGenerator`.
"""

import numpy as np
from src.noise.noise_generator import NoiseGenerator

_default_generator = NoiseGenerator()


def add_white_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to a signal at a specified SNR."""
    return _default_generator.add_white_noise(signal, snr_db)
