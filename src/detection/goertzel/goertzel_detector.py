"""
Goertzel-Based DTMF Detector
==============================
Inherits from the shared DTMFDetector base class and uses the Goertzel
algorithm to detect DTMF tones.
"""

import numpy as np

from src.detection.detector_base import (
    DTMFDetector,
    DTMF_LOW_FREQS,
    DTMF_HIGH_FREQS,
)
from src.detection.goertzel.goertzel_core import goertzel_power

RELATIVE_POWER_THRESHOLD = 0.10
INTER_TONE_GAP_FACTOR    = 3.0


class GoertzelDetector(DTMFDetector):
    """DTMF detector using the Goertzel algorithm."""

    @property
    def name(self) -> str:
        return "Goertzel Algorithm"

    def detect_single(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a single DTMF key from one tone segment."""
        all_freqs = DTMF_LOW_FREQS + DTMF_HIGH_FREQS
        powers = {f: goertzel_power(signal, f, sample_rate)
                  for f in all_freqs}

        max_power = max(powers.values())

        if max_power < 1e-10:
            return '?'

        low_freq  = self._pick_dominant(powers, DTMF_LOW_FREQS,  max_power)
        high_freq = self._pick_dominant(powers, DTMF_HIGH_FREQS, max_power)

        return self._map_freqs_to_key(low_freq, high_freq)

    def detect_sequence(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a sequence of DTMF keys from a longer signal."""
        segments = self._segment_signal(signal, sample_rate)
        decoded  = []

        for seg in segments:
            key = self.detect_single(seg, sample_rate)
            decoded.append(key)

        return ''.join(decoded)

    @staticmethod
    def _pick_dominant(powers: dict[float, float],
                       group: list[int],
                       max_power: float) -> int | None:
        """Pick the strongest frequency in `group`, enforcing thresholds."""
        sorted_freqs = sorted(group, key=lambda f: powers[f], reverse=True)
        best_freq   = sorted_freqs[0]
        best_power  = powers[best_freq]

        if best_power < RELATIVE_POWER_THRESHOLD * max_power:
            return None

        if len(sorted_freqs) > 1:
            runner_up_power = powers[sorted_freqs[1]]
            if runner_up_power > 0 and best_power < INTER_TONE_GAP_FACTOR * runner_up_power:
                return None

        return best_freq
