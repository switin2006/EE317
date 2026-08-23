"""
Goertzel Algorithm Implementation
=================================
Computes the DFT at a single target frequency using a two-stage IIR + FIR structure.
"""

import numpy as np


def goertzel_power(signal: np.ndarray, target_freq: float,
                   sample_rate: int) -> float:
    """Compute |X[k0]|^2 at a single target frequency using Goertzel.
    
    Uses the optimised form requiring only the IIR stage.
    """
    N = len(signal)
    k0 = (target_freq / sample_rate) * N
    omega = 2.0 * np.pi * k0 / N
    coeff = 2.0 * np.cos(omega)

    v1 = 0.0
    v2 = 0.0

    for n in range(N):
        v = coeff * v1 - v2 + signal[n]
        v2 = v1
        v1 = v

    return v1 * v1 + v2 * v2 - coeff * v1 * v2


def goertzel_power_all_dtmf(signal: np.ndarray, sample_rate: int,
                            freqs: list[float]) -> dict[float, float]:
    """Run the Goertzel algorithm at every frequency in `freqs`."""
    return {f: goertzel_power(signal, f, sample_rate) for f in freqs}
