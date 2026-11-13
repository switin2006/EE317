"""
Noise Generator
================
Provides a class-based interface for adding various types of noise
to DTMF signals for robustness testing.

Consolidates the two previous implementations:
  - src/generation/noise_utils.py  (add_white_noise)
  - Wavelet/wavelet.py             (add_awgn_noise)

into a single, extensible class with consistent SNR handling.
"""

import numpy as np


class NoiseGenerator:
    """Generates and applies noise to signals at a specified SNR.

    Attributes:
        seed: Optional RNG seed for reproducible noise generation.
    """

    def __init__(self, seed: int | None = None):
        """Initialise the noise generator.

        Args:
            seed: Random seed for reproducibility.  Pass None for
                  non-deterministic behaviour.
        """
        self._rng = np.random.default_rng(seed)

    # ── Public API ────────────────────────────────────────────────

    def add_white_noise(self, signal: np.ndarray,
                        snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise (AWGN) at a target SNR.

        Args:
            signal:  1-D numpy array (the clean signal).
            snr_db:  Desired Signal-to-Noise Ratio in decibels.

        Returns:
            A new array containing the noisy signal.
        """
        noise = self._generate_white_noise(signal, snr_db)
        return signal + noise

    def add_pink_noise(self, signal: np.ndarray,
                       snr_db: float) -> np.ndarray:
        """Add pink (1/f) noise at a target SNR.

        Pink noise has equal energy per octave, making it more
        realistic for telephone-channel modelling.

        Args:
            signal:  1-D numpy array (the clean signal).
            snr_db:  Desired Signal-to-Noise Ratio in decibels.

        Returns:
            A new array containing the noisy signal.
        """
        N = len(signal)
        # Generate white noise in the frequency domain
        white = self._rng.standard_normal(N)
        fft_white = np.fft.rfft(white)

        # Build 1/sqrt(f) filter (skip DC to avoid division by zero)
        freqs = np.fft.rfftfreq(N)
        filt = np.ones_like(freqs)
        filt[1:] = 1.0 / np.sqrt(freqs[1:])

        pink = np.fft.irfft(fft_white * filt, n=N)

        # Scale to desired SNR
        pink = self._scale_to_snr(signal, pink, snr_db)
        return signal + pink

    def add_impulse_noise(self, signal: np.ndarray,
                          snr_db: float,
                          density: float = 0.01) -> np.ndarray:
        """Add impulse (click/pop) noise at a target SNR.

        Models short transient disturbances common in telephony.

        Args:
            signal:   1-D numpy array (the clean signal).
            snr_db:   Desired Signal-to-Noise Ratio in decibels.
            density:  Fraction of samples affected (0.0 – 1.0).

        Returns:
            A new array containing the noisy signal.
        """
        N = len(signal)
        impulse = np.zeros(N)

        # Randomly place impulses
        num_impulses = max(1, int(N * density))
        positions = self._rng.choice(N, size=num_impulses, replace=False)
        impulse[positions] = self._rng.choice([-1.0, 1.0], size=num_impulses)

        # Scale impulses to desired SNR
        impulse = self._scale_to_snr(signal, impulse, snr_db)
        return signal + impulse

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    def _signal_power(signal: np.ndarray) -> float:
        """Compute the average power of a signal."""
        return float(np.mean(signal ** 2))

    def _generate_white_noise(self, signal: np.ndarray,
                              snr_db: float) -> np.ndarray:
        """Generate a white Gaussian noise vector scaled to a target SNR."""
        sig_power = self._signal_power(signal)
        if sig_power == 0:
            return np.zeros_like(signal)

        noise_power = sig_power / (10 ** (snr_db / 10))
        return self._rng.normal(0, np.sqrt(noise_power), len(signal))

    @staticmethod
    def _scale_to_snr(signal: np.ndarray, noise: np.ndarray,
                      snr_db: float) -> np.ndarray:
        """Scale an arbitrary noise vector so signal+noise has the desired SNR."""
        sig_power = float(np.mean(signal ** 2))
        if sig_power == 0:
            return np.zeros_like(noise)

        noise_power_current = float(np.mean(noise ** 2))
        if noise_power_current == 0:
            return noise

        target_noise_power = sig_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power_current)
        return noise * scale

