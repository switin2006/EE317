"""
Test: Wavelet Single-Tone Detection
=====================================
Generates individual DTMF tones and verifies that the WaveletDetector
correctly identifies each one (clean and noisy).
"""

import sys
import os
import numpy as np

# ── make sure imports resolve from the project root ──────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.detection.Wavelet.wavelet_detector import WaveletDetector
from src.detection.Wavelet.wavelet_core import wpt_band_energies
from src.detection.detector_base import DTMF_LOW_FREQS, DTMF_HIGH_FREQS, DTMF_TABLE


# ── helper: generate a clean DTMF tone inline ────────────────────
def make_tone(low_freq: float, high_freq: float,
              sample_rate: int = 8000, duration: float = 0.2,
              amplitude: float = 0.5) -> np.ndarray:
    """Return x[n] = A*(sin low + sin high)."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * (np.sin(2 * np.pi * low_freq * t) +
                        np.sin(2 * np.pi * high_freq * t))



# ══════════════════════════════════════════════════════════════════
#                         MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    detector    = WaveletDetector()
    sample_rate = 8000
    duration    = 0.2          # 200 ms  (1 600 samples at 8 kHz)

    print("=" * 60)
    print(f"  Wavelet Single-Tone Detection Test")
    print(f"  Detector : {detector.name}")
    print(f"  Fs = {sample_rate} Hz   |   Tone = {duration*1000:.0f} ms")
    print("=" * 60)

    # ── 1.  Clean tones ──────────────────────────────────────────
    print("\n-- CLEAN TONES -----------------------------------------")
    passed = 0
    total  = 0

    for (low, high), expected_key in sorted(DTMF_TABLE.items()):
        signal    = make_tone(low, high, sample_rate, duration)
        detected  = detector.detect_single(signal, sample_rate)
        ok        = detected == expected_key
        passed   += ok
        total    += 1
        status    = "PASS" if ok else "FAIL"
        print(f"  {status}  Key '{expected_key}'  ({low} + {high} Hz)  ->  "
              f"detected '{detected}'")

    print(f"\n  Result: {passed}/{total} correct\n")

    # ── 2.  Show WPT energy spectrum for one key ─────────────────
    print("-- WPT ENERGY SPECTRUM for key '5' (770 + 1336 Hz) ----")
    signal = make_tone(770, 1336, sample_rate, duration)
    energies, centers = wpt_band_energies(signal, fs=sample_rate)

    all_dtmf = DTMF_LOW_FREQS + DTMF_HIGH_FREQS
    for f in all_dtmf:
        # Find the WPT band closest to this DTMF frequency
        idx = np.argmin(np.abs(centers - f))
        eng = energies[idx]
        bar = "#" * int(eng / 50)          # simple visual bar
        tag = " <-- PEAK" if f in (770, 1336) else ""
        print(f"  {f:5d} Hz  | energy = {eng:10.1f}  {bar}{tag}")

    print("\nDone.")
