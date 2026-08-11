"""
DTMF Signal Generator — Member 1

Generates synthetic DTMF tones for all 16 keys, multi-digit sequences,
and provides audio playback utilities.
"""

import numpy as np
from src.detector_base import KEY_TO_FREQS, DTMF_TABLE


# Default parameters
DEFAULT_SAMPLE_RATE = 8000       # 8 kHz (telephony standard)
DEFAULT_TONE_DURATION = 0.2      # 200 ms per tone
DEFAULT_SILENCE_DURATION = 0.1   # 100 ms silence between tones
DEFAULT_AMPLITUDE = 0.5          # Signal amplitude (0 to 1)


def generate_dtmf_tone(key: str,
                        sample_rate: int = DEFAULT_SAMPLE_RATE,
                        duration: float = DEFAULT_TONE_DURATION,
                        amplitude: float = DEFAULT_AMPLITUDE) -> np.ndarray:
    """Generate a single DTMF tone for a given key.

    Args:
        key: DTMF key character ('0'-'9', '*', '#', 'A'-'D').
        sample_rate: Sampling frequency in Hz.
        duration: Tone duration in seconds.
        amplitude: Peak amplitude of each sinusoid (total signal is 2x).

    Returns:
        1D numpy array containing the DTMF tone.

    Raises:
        ValueError: If the key is not a valid DTMF character.
    """
    key = key.upper()
    if key not in KEY_TO_FREQS:
        raise ValueError(
            f"Invalid DTMF key: '{key}'. "
            f"Valid keys: {list(KEY_TO_FREQS.keys())}"
        )

    low_freq, high_freq = KEY_TO_FREQS[key]
    t = np.arange(0, duration, 1.0 / sample_rate)

    # DTMF = sum of two sinusoids
    signal = amplitude * (np.sin(2 * np.pi * low_freq * t) +
                          np.sin(2 * np.pi * high_freq * t))

    return signal


def generate_silence(duration: float = DEFAULT_SILENCE_DURATION,
                     sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Generate a silence segment.

    Args:
        duration: Silence duration in seconds.
        sample_rate: Sampling frequency in Hz.

    Returns:
        1D numpy array of zeros.
    """
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples)


def generate_dtmf_sequence(keys: str,
                            sample_rate: int = DEFAULT_SAMPLE_RATE,
                            tone_duration: float = DEFAULT_TONE_DURATION,
                            silence_duration: float = DEFAULT_SILENCE_DURATION,
                            amplitude: float = DEFAULT_AMPLITUDE) -> np.ndarray:
    """Generate a sequence of DTMF tones (e.g., a phone number).

    Each tone is followed by a silence gap, except the last tone.

    Args:
        keys: String of DTMF keys (e.g., '5551234').
        sample_rate: Sampling frequency in Hz.
        tone_duration: Duration of each tone in seconds.
        silence_duration: Duration of silence between tones in seconds.
        amplitude: Peak amplitude of each sinusoid.

    Returns:
        1D numpy array containing the complete DTMF sequence.
    """
    segments = []

    for i, key in enumerate(keys):
        tone = generate_dtmf_tone(key, sample_rate, tone_duration, amplitude)
        segments.append(tone)

        # Add silence between tones (not after the last one)
        if i < len(keys) - 1:
            silence = generate_silence(silence_duration, sample_rate)
            segments.append(silence)

    return np.concatenate(segments)


def generate_all_tones(sample_rate: int = DEFAULT_SAMPLE_RATE,
                       duration: float = DEFAULT_TONE_DURATION,
                       amplitude: float = DEFAULT_AMPLITUDE) -> dict[str, np.ndarray]:
    """Generate DTMF tones for all 16 keys.

    Args:
        sample_rate: Sampling frequency in Hz.
        duration: Tone duration in seconds.
        amplitude: Peak amplitude.

    Returns:
        Dictionary mapping key characters to their signal arrays.
    """
    all_keys = list(KEY_TO_FREQS.keys())
    tones = {}
    for key in all_keys:
        tones[key] = generate_dtmf_tone(key, sample_rate, duration, amplitude)
    return tones


def get_signal_info(signal: np.ndarray, sample_rate: int) -> dict:
    """Get basic information about a signal.

    Args:
        signal: 1D numpy array.
        sample_rate: Sampling frequency in Hz.

    Returns:
        Dictionary with signal properties.
    """
    return {
        'num_samples': len(signal),
        'duration_s': len(signal) / sample_rate,
        'sample_rate': sample_rate,
        'peak_amplitude': np.max(np.abs(signal)),
        'rms_amplitude': np.sqrt(np.mean(signal ** 2)),
    }
