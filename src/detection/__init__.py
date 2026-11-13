"""
DTMF Detection Algorithms
==========================
Unified interface for all DTMF detector implementations.
"""

from .detector_base import DTMFDetector, DTMF_TABLE, DTMF_LOW_FREQS, DTMF_HIGH_FREQS
from .goertzel import GoertzelDetector
from .FFT import FFTDetector
from .Wavelet import WaveletDetector

__all__ = [
    'DTMFDetector',
    'DTMF_TABLE',
    'DTMF_LOW_FREQS',
    'DTMF_HIGH_FREQS',
    'GoertzelDetector',
    'FFTDetector',
    'WaveletDetector',
]
