"""
Base class for all DTMF detection algorithms.

All detectors (FFT, Goertzel, Wavelet) must inherit from DTMFDetector
and implement the abstract methods. This ensures a uniform interface
for benchmarking and comparison.
"""

import numpy as np
from abc import ABC, abstractmethod

# Standard DTMF frequency definitions
DTMF_LOW_FREQS = [697, 770, 852, 941]
DTMF_HIGH_FREQS = [1209, 1336, 1477, 1633]

# DTMF keypad mapping: (low_freq, high_freq) -> key
DTMF_TABLE = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D',
}

class DTMFDetector(ABC):
    """Abstract base class for DTMF detection algorithms."""

    @abstractmethod
    def detect_single(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a single DTMF key from a signal segment.
        
        Must return a valid key (e.g., '5') or '?' if detection fails.
        """
        pass

    @abstractmethod
    def detect_sequence(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a sequence of DTMF keys from a longer signal."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this detection algorithm."""
        pass

    # --- Shared Helper Functions ---

    def _find_closest_freq(self, detected_freq: float, freq_list: list,
                           tolerance_pct: float = 3.5) -> int | None:
        """Find the closest standard DTMF frequency within a tolerance."""
        best_match = None
        best_error = float('inf')

        for freq in freq_list:
            error_pct = abs(detected_freq - freq) / freq * 100
            if error_pct < best_error and error_pct <= tolerance_pct:
                best_error = error_pct
                best_match = freq

        return best_match

    def _map_freqs_to_key(self, low_freq: int | None,
                          high_freq: int | None) -> str:
        """Map a (low_freq, high_freq) pair to a DTMF key."""
        if low_freq is None or high_freq is None:
            return '?'
        return DTMF_TABLE.get((low_freq, high_freq), '?')

    def _segment_signal(self, signal: np.ndarray, sample_rate: int,
                        frame_duration_ms: float = 40.0,
                        energy_threshold: float = 0.01) -> list[np.ndarray]:
        """Segment a multi-digit signal into individual tone segments based on energy."""
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        num_frames = len(signal) // frame_size

        is_active = []
        for i in range(num_frames):
            frame = signal[i * frame_size: (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            is_active.append(rms > energy_threshold)

        segments = []
        in_segment = False
        start = 0

        for i, active in enumerate(is_active):
            if active and not in_segment:
                start = i
                in_segment = True
            elif not active and in_segment:
                end = i
                seg = signal[start * frame_size: end * frame_size]
                segments.append(seg)
                in_segment = False

        if in_segment:
            seg = signal[start * frame_size: num_frames * frame_size]
            segments.append(seg)

        return segments
