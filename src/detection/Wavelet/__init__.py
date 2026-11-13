"""Wavelet-based DTMF detection module."""
from .wavelet_detector import WaveletDetector
from .wavelet_core import (
    wpt_band_energies,
    analyze_dwt,
    detect_dtmf_wpt,
    plot_analysis,
    get_freq_ordered_nodes,
    node_to_freq_range,
    calculate_energy,
)

__all__ = [
    'WaveletDetector',
    'wpt_band_energies',
    'analyze_dwt',
    'detect_dtmf_wpt',
    'plot_analysis',
    'get_freq_ordered_nodes',
    'node_to_freq_range',
    'calculate_energy',
]
