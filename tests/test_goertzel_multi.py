"""
Test: Goertzel Multi-Tone (Sequence) Detection
================================================
Generates multi-digit DTMF sequences (with silence gaps) and verifies
that GoertzelDetector.detect_sequence() decodes the full string.
"""

import sys
import os
import numpy as np

# ── make sure imports resolve from the project root ──────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.goertzel.goertzel_detector import GoertzelDetector
from src.detection.detector_base import DTMF_TABLE


# ── helpers ──────────────────────────────────────────────────────
# We reverse the DTMF_TABLE so we can look up  key → (low, high)
KEY_TO_FREQS = {v: k for k, v in DTMF_TABLE.items()}


def make_tone(low_freq, high_freq, sample_rate, duration, amplitude=0.5):
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * (np.sin(2 * np.pi * low_freq * t) +
                        np.sin(2 * np.pi * high_freq * t))


def make_sequence(keys: str, sample_rate: int = 8000,
                  tone_dur: float = 0.2, silence_dur: float = 0.1,
                  amplitude: float = 0.5) -> np.ndarray:
    """Build a multi-tone signal:  tone – silence – tone – silence – ..."""
    parts = []
    for i, key in enumerate(keys.upper()):
        low, high = KEY_TO_FREQS[key]
        parts.append(make_tone(low, high, sample_rate, tone_dur, amplitude))
        if i < len(keys) - 1:
            parts.append(np.zeros(int(silence_dur * sample_rate)))
    return np.concatenate(parts)



# ==============================================================
#                         MAIN
# ==============================================================
if __name__ == "__main__":
    detector    = GoertzelDetector()
    sample_rate = 8000

    print("=" * 60)
    print(f"  Goertzel Multi-Tone (Sequence) Detection Test")
    print(f"  Detector : {detector.name}")
    print(f"  Fs = {sample_rate} Hz")
    print("=" * 60)

    # -- Test sequences -----------------------------------------------
    test_sequences = [
        "5551234",      # a phone number
        "0123456789",   # all digits
        "1A2B3C4D",     # digits + letters
        "*#0",          # special characters
        "911",          # short sequence
    ]

    # -- 1.  Clean sequences ------------------------------------------
    print("\n-- CLEAN SEQUENCES -------------------------------------")
    all_passed = 0
    all_total  = 0

    for seq in test_sequences:
        signal   = make_sequence(seq, sample_rate)
        detected = detector.detect_sequence(signal, sample_rate)
        ok       = detected == seq
        all_passed += ok
        all_total  += 1
        status   = "PASS" if ok else "FAIL"
        print(f"  {status}  Sent: '{seq}'  ->  Detected: '{detected}'")

    print(f"\n  Result: {all_passed}/{all_total} sequences correct\n")

    # -- 3.  Edge cases -----------------------------------------------
    print("-- EDGE CASES ----------------------------------------")

    # 3a. Very short tone (50 ms)
    short_seq    = make_sequence("42", sample_rate, tone_dur=0.05, silence_dur=0.05)
    short_result = detector.detect_sequence(short_seq, sample_rate)
    print(f"  Short tones  (50 ms):  Sent '42'  ->  Detected '{short_result}'")

    # 3b. Long tone (500 ms)
    long_seq    = make_sequence("8", sample_rate, tone_dur=0.5)
    long_result = detector.detect_sequence(long_seq, sample_rate)
    print(f"  Long tone   (500 ms):  Sent '8'   ->  Detected '{long_result}'")

    # 3c. Single digit
    single_seq    = make_sequence("0", sample_rate)
    single_result = detector.detect_sequence(single_seq, sample_rate)
    print(f"  Single digit        :  Sent '0'   ->  Detected '{single_result}'")

    print("\nDone.")

