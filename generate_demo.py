import numpy as np
from scipy.io import wavfile
import os

from src.generation.dtmf_generator import generate_dtmf_sequence, generate_dtmf_tone
from src.generation.noise_utils import add_white_noise

def save_wav(filename, signal, sample_rate=8000):
    # Normalize to 16-bit integer PCM
    normalized = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(filename, sample_rate, normalized)
    print(f"Saved {filename}")

if __name__ == "__main__":
    sample_rate = 8000
    
    # Ensure a directory exists for output
    os.makedirs("demo_samples", exist_ok=True)
    
    # 1. Generate a clean single tone '5'
    print("Generating single tone '5'...")
    tone_5 = generate_dtmf_tone('5', sample_rate=sample_rate, duration=1.0)
    save_wav("demo_samples/clean_5.wav", tone_5, sample_rate)
    
    # 2. Generate a clean sequence (Phone Number)
    print("Generating sequence '5551234'...")
    sequence = generate_dtmf_sequence('5551234', sample_rate=sample_rate)
    save_wav("demo_samples/clean_sequence.wav", sequence, sample_rate)
    
    # 3. Generate a noisy version of the sequence (SNR 5 dB - very noisy)
    print("Generating noisy sequence (SNR 5 dB)...")
    noisy_sequence = add_white_noise(sequence, snr_db=5.0)
    save_wav("demo_samples/noisy_sequence_5db.wav", noisy_sequence, sample_rate)
    
    print("\nDone! Check the 'demo_samples' folder for your generated .wav files.")
