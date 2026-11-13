"""
Test: Noise Removal
====================
Tests that NoiseRemover can clean noisy DTMF signals well enough
for the detectors to still recognise the correct key.
"""

import sys
import os
import numpy as np

# ── make sure imports resolve from the project root ──────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.noise.noise_generator import NoiseGenerator
from src.noise.noise_remover import NoiseRemover
from src.generation.dtmf_generator import generate_dtmf_tone
from src.detection.goertzel.goertzel_detector import GoertzelDetector
from src.detection.FFT.fft_detector import FFTDetector
from src.detection.detector_base import DTMF_TABLE


def measure_snr(clean: np.ndarray, processed: np.ndarray) -> float:
    """Measure SNR improvement after denoising."""
    noise = processed - clean
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
    snr_db      = 5  # fairly noisy

    print("=" * 60)
    print("  Noise Removal Tests")
    print(f"  Input SNR = {snr_db} dB")
    print("=" * 60)

    # Setup
    ng       = NoiseGenerator(seed=42)
    remover  = NoiseRemover(sample_rate=sample_rate)
    detector = GoertzelDetector()

    # ── 1.  Spectral Gating ──────────────────────────────────────
    print("\n-- SPECTRAL GATING -------------------------------------")
    passed = 0
    total  = 0
    for (low, high), expected_key in sorted(DTMF_TABLE.items()):
        clean = generate_dtmf_tone(expected_key, sample_rate=sample_rate,
                                    duration=duration)
        noisy = ng.add_white_noise(clean, snr_db=snr_db)
        cleaned = remover.spectral_gate(noisy, threshold_factor=0.08)
        detected = detector.detect_single(cleaned, sample_rate)
        ok = detected == expected_key
        passed += ok
        total  += 1

    print(f"  Detection after gating: {passed}/{total} keys correct")

    # ── 2.  Wavelet Thresholding ─────────────────────────────────
    print("\n-- WAVELET THRESHOLDING (db4, soft) ---------------------")
    passed = 0
    total  = 0
    for (low, high), expected_key in sorted(DTMF_TABLE.items()):
        clean = generate_dtmf_tone(expected_key, sample_rate=sample_rate,
                                    duration=duration)
        noisy = ng.add_white_noise(clean, snr_db=snr_db)
        cleaned = remover.wavelet_denoise(noisy, wavelet='db4', level=4,
                                           mode='soft')
        detected = detector.detect_single(cleaned, sample_rate)
        ok = detected == expected_key
        passed += ok
        total  += 1

    print(f"  Detection after wavelet denoise: {passed}/{total} keys correct")

    # ── 3.  Bandpass Filter ──────────────────────────────────────
    print("\n-- BANDPASS FILTER (650-1700 Hz) ------------------------")
    passed = 0
    total  = 0
    for (low, high), expected_key in sorted(DTMF_TABLE.items()):
        clean = generate_dtmf_tone(expected_key, sample_rate=sample_rate,
                                    duration=duration)
        noisy = ng.add_white_noise(clean, snr_db=snr_db)
        cleaned = remover.bandpass_filter(noisy)
        detected = detector.detect_single(cleaned, sample_rate)
        ok = detected == expected_key
        passed += ok
        total  += 1

    print(f"  Detection after bandpass: {passed}/{total} keys correct")

    # ── 4.  SNR Improvement Comparison ───────────────────────────
    print("\n-- SNR IMPROVEMENT COMPARISON ---------------------------")
    clean = generate_dtmf_tone('5', sample_rate=sample_rate, duration=duration)
    noisy = ng.add_white_noise(clean, snr_db=snr_db)

    methods = {
        'Spectral Gating':     remover.spectral_gate(noisy),
        'Wavelet (soft, db4)': remover.wavelet_denoise(noisy),
        'Bandpass Filter':     remover.bandpass_filter(noisy),
        'Moving Average (5)':  remover.moving_average(noisy, window_size=5),
    }

    input_snr = measure_snr(clean, noisy)
    print(f"  Input SNR:          {input_snr:6.2f} dB")

    for name, cleaned in methods.items():
        output_snr   = measure_snr(clean, cleaned)
        improvement  = output_snr - input_snr
        print(f"  {name:25s} -> {output_snr:6.2f} dB  "
              f"(+{improvement:.2f} dB improvement)")

    # ── 5.  Stress Test: Detection at very low SNR ───────────────
    print("\n-- STRESS TEST: Detection at 0 dB -----------------------")
    stress_snr = 0
    detector_fft = FFTDetector()

    for method_name, denoise_fn in [
        ("No denoising",    lambda s: s),
        ("Spectral Gate",   lambda s: remover.spectral_gate(s)),
        ("Wavelet Denoise", lambda s: remover.wavelet_denoise(s)),
        ("Bandpass Filter", lambda s: remover.bandpass_filter(s)),
    ]:
        goertzel_pass = 0
        fft_pass      = 0
        total_keys    = 0

        for (low, high), expected_key in sorted(DTMF_TABLE.items()):
            clean = generate_dtmf_tone(expected_key, sample_rate=sample_rate,
                                        duration=duration)
            noisy = ng.add_white_noise(clean, snr_db=stress_snr)
            cleaned = denoise_fn(noisy)

            if detector.detect_single(cleaned, sample_rate) == expected_key:
                goertzel_pass += 1
            if detector_fft.detect_single(cleaned, sample_rate) == expected_key:
                fft_pass += 1
            total_keys += 1

        print(f"  {method_name:20s} | Goertzel: {goertzel_pass:2d}/{total_keys}  "
              f"| FFT: {fft_pass:2d}/{total_keys}")

    print("\nDone.")

