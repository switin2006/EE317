import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generation.dtmf_generator import generate_dtmf_tone
from src.detection.detector_base import DTMF_LOW_FREQS, DTMF_HIGH_FREQS
from src.detection.goertzel.goertzel_core import goertzel_power_all_dtmf

def main():
    # 1. Generate a sample tone for '5'
    sample_rate = 8000
    target_key = '5'
    print(f"Generating DTMF tone for '{target_key}'...")
    signal = generate_dtmf_tone(target_key, sample_rate=sample_rate)

    # 2. Define standard DTMF frequencies
    standard_freqs = DTMF_LOW_FREQS + DTMF_HIGH_FREQS
    
    # 3. Calculate Goertzel power at these frequencies
    print("Running Goertzel algorithm...")
    powers = goertzel_power_all_dtmf(signal, sample_rate, standard_freqs)
    
    # 4. Plot the results
    print("Generating bar chart...")
    freqs = list(powers.keys())
    power_vals = list(powers.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(f) for f in freqs], power_vals, color='skyblue', edgecolor='black')
    
    # Highlight the expected frequencies for '5' (770 Hz and 1336 Hz)
    expected_freqs = [770, 1336]
    for i, f in enumerate(freqs):
        if f in expected_freqs:
            bars[i].set_color('orange')
            bars[i].set_edgecolor('black')
            
    plt.title(f"Goertzel Algorithm Power Spectrum (Digit '{target_key}')", fontsize=14)
    plt.xlabel("Target Frequency (Hz)", fontsize=12)
    plt.ylabel("Computed Power", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goertzel_spectrum.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
