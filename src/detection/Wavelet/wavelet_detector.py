"""
Wavelet-Based DTMF Detector
=============================
Inherits from the shared DTMFDetector base class and uses the
Wavelet Packet Transform (WPT) to detect DTMF tones.
"""

import numpy as np

from src.detection.detector_base import (
    DTMFDetector,
    DTMF_LOW_FREQS,
    DTMF_HIGH_FREQS,
)
from src.detection.Wavelet.wavelet_core import wpt_band_energies

SEARCH_WINDOW_HZ = 40  # ±40 Hz tolerance around each DTMF frequency


class WaveletDetector(DTMFDetector):
    """DTMF detector using the Wavelet Packet Transform (WPT)."""

    def __init__(self, wavelet: str = 'db4', level: int = 7):
        """Initialise the Wavelet detector.

        Args:
            wavelet: Wavelet family (e.g. 'db4', 'sym6').
            level:   WPT decomposition depth.
                     Level 7 → 128 bands of 31.25 Hz at 8 kHz.
        """
        self._wavelet = wavelet
        self._level = level

    @property
    def name(self) -> str:
        return "Wavelet Packet Transform (WPT)"

    def detect_single(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a single DTMF key from one tone segment."""
        if len(signal) == 0:
            return '?'

        energies, centers = wpt_band_energies(
            signal, fs=sample_rate,
            wavelet=self._wavelet, level=self._level
        )

        low_freq = self._pick_dominant(energies, centers, DTMF_LOW_FREQS)
        high_freq = self._pick_dominant(energies, centers, DTMF_HIGH_FREQS)

        return self._map_freqs_to_key(low_freq, high_freq)

    def detect_sequence(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a sequence of DTMF keys from a longer signal."""
        segments = self._segment_signal(signal, sample_rate)
        decoded = []

        for seg in segments:
            key = self.detect_single(seg, sample_rate)
            decoded.append(key)

        return ''.join(decoded)

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    def _pick_dominant(energies: np.ndarray, centers: np.ndarray,
                       freq_group: list[int]) -> int | None:
        """Find the DTMF frequency in `freq_group` with the highest
        WPT band energy within the search window."""
        best_energy = 0
        best_dtmf = None

        for target_f in freq_group:
            for i, center_f in enumerate(centers):
                if abs(center_f - target_f) < SEARCH_WINDOW_HZ:
                    if energies[i] > best_energy:
                        best_energy = energies[i]
                        best_dtmf = target_f

        return best_dtmf
