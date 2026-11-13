"""
Test: Noise Generation
=======================
Tests that NoiseGenerator correctly adds white, pink, and impulse
noise at various SNR levels to DTMF signals.
"""

import sys
import os
import numpy as np

# ── make sure imports resolve from the project root ──────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.noise.noise_generator import NoiseGenerator
from src.generation.dtmf_generator import generate_dtmf_tone


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the actual SNR (dB) between a clean and noisy signal."""
    noise = noisy - clean
    sig_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(sig_power / noise_power)


# ==============================================================
#                         MAIN
# ==============================================================
if __name__ == "__main__":
    sample_rate = 8000
    duration    = 0.2

    print("=" * 60)
    print("  Noise Generator Tests")
    print("=" * 60)

    # Generate a clean reference tone (key '5': 770 + 1336 Hz)
    clean_signal = generate_dtmf_tone('5', sample_rate=sample_rate,
                                       duration=duration)

    # ── 1.  White Noise SNR Accuracy ─────────────────────────────
    print("\n-- WHITE NOISE (AWGN) ----------------------------------")
    ng = NoiseGenerator(seed=42)
    target_snrs = [20, 10, 5, 0, -5]
    all_pass = True

    for target in target_snrs:
        noisy = ng.add_white_noise(clean_signal, snr_db=target)
        actual = measure_snr(clean_signal, noisy)
        error  = abs(actual - target)
        ok     = error < 1.5   # allow ±1.5 dB tolerance
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  Target: {target:4d} dB  |  Actual: {actual:6.2f} dB  "
              f"|  Error: {error:.2f} dB")

    # ── 2.  Pink Noise ───────────────────────────────────────────
    print("\n-- PINK NOISE (1/f) ------------------------------------")
    for target in [10, 5, 0]:
        noisy = ng.add_pink_noise(clean_signal, snr_db=target)
        actual = measure_snr(clean_signal, noisy)
        error  = abs(actual - target)
        ok     = error < 2.0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  Target: {target:4d} dB  |  Actual: {actual:6.2f} dB  "
              f"|  Error: {error:.2f} dB")

    # ── 3.  Impulse Noise ────────────────────────────────────────
    print("\n-- IMPULSE NOISE ---------------------------------------")
    for target in [10, 5, 0]:
        noisy = ng.add_impulse_noise(clean_signal, snr_db=target, density=0.02)
        actual = measure_snr(clean_signal, noisy)
        error  = abs(actual - target)
        ok     = error < 2.0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  Target: {target:4d} dB  |  Actual: {actual:6.2f} dB  "
              f"|  Error: {error:.2f} dB")

    # ── 4.  Reproducibility (seeded) ─────────────────────────────
    print("\n-- REPRODUCIBILITY (seed=42) ----------------------------")
    ng_a = NoiseGenerator(seed=42)
    ng_b = NoiseGenerator(seed=42)
    noisy_a = ng_a.add_white_noise(clean_signal, snr_db=10)
    noisy_b = ng_b.add_white_noise(clean_signal, snr_db=10)
    ok = np.allclose(noisy_a, noisy_b)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  {status}  Same seed produces identical noise")

    # ── 5.  Zero-power guard ─────────────────────────────────────
    print("\n-- ZERO-POWER GUARD ------------------------------------")
    silence = np.zeros(1600)
    noisy_silence = ng.add_white_noise(silence, snr_db=10)
    ok = np.allclose(noisy_silence, silence)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  {status}  Silent signal remains silent after noise addition")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)

