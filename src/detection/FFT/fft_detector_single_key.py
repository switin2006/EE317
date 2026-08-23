import numpy as np
from scipy.io import wavfile

# Import the lookup table your team created
from src.detection.detector_base import DTMF_TABLE

def decode_single_tone(filepath: str) -> str:
    """Reads a .wav file, computes the FFT, and detects the DTMF key."""
    # Read the audio file
    sample_rate, signal = wavfile.read(filepath)
    
    # Normalize the 16-bit PCM signal back to a -1 to 1 range
    if signal.dtype == np.int16:
        signal = signal / 32767.0
        
    N = len(signal)
    
    # Apply a Hamming window to reduce spectral leakage
    windowed_signal = signal * np.hamming(N)
    
    # Compute the FFT and get the magnitude spectrum
    fft_result = np.fft.fft(windowed_signal)
    frequencies = np.fft.fftfreq(N, 1/sample_rate)
    
    # We only care about the positive frequencies
    half_N = N // 2
    fft_mag = np.abs(fft_result[:half_N])
    freqs = frequencies[:half_N]
    
    # Isolate the standard DTMF frequency bands
    low_band_mask = (freqs >= 600) & (freqs <= 1000)
    high_band_mask = (freqs >= 1100) & (freqs <= 1700)
    
    # Find the peak frequency in each band
    peak_low = freqs[low_band_mask][np.argmax(fft_mag[low_band_mask])]
    peak_high = freqs[high_band_mask][np.argmax(fft_mag[high_band_mask])]
    
    # Standard DTMF frequencies for error correction
    standard_lows = [697, 770, 852, 941]
    standard_highs = [1209, 1336, 1477, 1633]
    
    # Snap the detected peaks to the closest standard frequencies
    matched_low = min(standard_lows, key=lambda f: abs(f - peak_low))
    matched_high = min(standard_highs, key=lambda f: abs(f - peak_high))
    
    # Lookup the key in your team's dictionary
    detected_key = DTMF_TABLE.get((matched_low, matched_high), "Unknown")
    
    print(f"--- Decoding: {filepath} ---")
    print(f"Detected Peaks: Low={peak_low:.1f} Hz, High={peak_high:.1f} Hz")
    print(f"Matched Freqs : Low={matched_low} Hz, High={matched_high} Hz")
    print(f"Decoded Key   : {detected_key}\n")
    
    return detected_key

if __name__ == "__main__":
    # Point the script to the clean '5' audio file you just generated
    decode_single_tone("demo_samples/clean_5.wav")