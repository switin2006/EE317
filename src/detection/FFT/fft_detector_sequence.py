import numpy as np
from scipy.io import wavfile
from src.detection.detector_base import DTMF_TABLE

def decode_sequence(filepath: str, frame_duration=0.04, energy_threshold=0.05):
    """Reads a sequence .wav file and decodes multiple DTMF keys."""
    sample_rate, signal = wavfile.read(filepath)
    
    if signal.dtype == np.int16:
        signal = signal / 32767.0
        
    frame_length = int(sample_rate * frame_duration)
    standard_lows = [697, 770, 852, 941]
    standard_highs = [1209, 1336, 1477, 1633]
    
    decoded_sequence = []
    last_detected = None
    
    print(f"--- Decoding Sequence: {filepath} ---")
    
    # Slide a window across the signal
    for i in range(0, len(signal) - frame_length, frame_length):
        frame = signal[i:i + frame_length]
        
        # 1. Energy Detection: Check if the frame is silence
        rms_energy = np.sqrt(np.mean(frame**2))
        if rms_energy < energy_threshold:
            last_detected = None # Reset when silence is found
            continue
            
        # 2. FFT Analysis (same as your single-tone logic)
        windowed = frame * np.hamming(len(frame))
        fft_res = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(len(frame), 1/sample_rate)
        
        half_N = len(frame) // 2
        fft_mag = np.abs(fft_res[:half_N])
        freqs = freqs[:half_N]
        
        low_mask = (freqs >= 600) & (freqs <= 1000)
        high_mask = (freqs >= 1100) & (freqs <= 1700)
        
        if not np.any(low_mask) or not np.any(high_mask):
            continue
            
        peak_low = freqs[low_mask][np.argmax(fft_mag[low_mask])]
        peak_high = freqs[high_mask][np.argmax(fft_mag[high_mask])]
        
        matched_low = min(standard_lows, key=lambda f: abs(f - peak_low))
        matched_high = min(standard_highs, key=lambda f: abs(f - peak_high))
        
        detected_key = DTMF_TABLE.get((matched_low, matched_high), None)
        
        # 3. Debouncing: Only add if it's a new key press
        if detected_key and detected_key != last_detected:
            decoded_sequence.append(detected_key)
            last_detected = detected_key
            
    final_output = "".join(decoded_sequence)
    print(f"Final Decoded Sequence: {final_output}\n")
    return final_output

if __name__ == "__main__":
    decode_sequence("demo_samples/clean_sequence.wav")