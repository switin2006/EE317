import numpy as np
import pywt
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal

# ==========================================
# PART 1: DTMF CONSTANTS AND CONFIGURATION
# ==========================================

DTMF_FREQUENCIES = {
    'L': [697, 770, 852, 941],
    'H': [1209, 1336, 1477, 1633]
}

DTMF_TABLE = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D'
}

# Reverse lookup
DIGIT_TO_FREQ = {v: k for k, v in DTMF_TABLE.items()}

# ==========================================
# PART 2: SIGNAL GENERATION AND NOISE
# ==========================================

def generate_dtmf_signal(digit, fs=8000, duration=0.2, amplitude=1.0):
    """Generates a DTMF signal for a given digit."""
    digit = str(digit).upper()
    if digit not in DIGIT_TO_FREQ:
        raise ValueError(f"Invalid DTMF digit: {digit}")
        
    f_low, f_high = DIGIT_TO_FREQ[digit]
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Generate the two sinusoids
    signal_low = amplitude * np.sin(2 * np.pi * f_low * t)
    signal_high = amplitude * np.sin(2 * np.pi * f_high * t)
    dtmf_signal = signal_low + signal_high
    
    return t, dtmf_signal, (f_low, f_high)

def add_awgn_noise(sig, snr_db):
    """Adds AWGN (Additive White Gaussian Noise) to the signal given a target SNR in dB."""
    # Calculate signal power and convert to dB
    sig_power = np.mean(sig ** 2)
    if sig_power == 0:
        return sig
    sig_power_db = 10 * np.log10(sig_power)
    
    # Calculate noise power in dB
    noise_power_db = sig_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)
    
    # Generate noise with calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), len(sig))
    return sig + noise

# ==========================================
# PART 3: WAVELET UTILITIES (WPT FREQ ORDERING)
# ==========================================

def get_freq_ordered_nodes(wp, level):
    """
    Returns the WPT nodes ordered strictly by frequency.
    In Wavelet Packets, high-pass and low-pass filtering can cause frequency band swapping.
    This function uses pywt's built-in order="freq" to sort the natural node paths into sequential frequency bands.
    """
    nodes = wp.get_level(level, order="freq")
    return nodes

def node_to_freq_range(node_index, level, fs):
    """Calculates the frequency range (in Hz) for a specific frequency-ordered node."""
    num_nodes = 2 ** level
    nyquist = fs / 2
    band_width = nyquist / num_nodes
    
    f_start = node_index * band_width
    f_end = (node_index + 1) * band_width
    f_center = (f_start + f_end) / 2
    
    return f_start, f_end, f_center

def calculate_energy(coeffs):
    """Calculates the energy of an array of coefficients."""
    return np.sum(np.abs(coeffs) ** 2)

# ==========================================
# PART 4: WAVELET-BASED DETECTION ALGORITHMS
# ==========================================

def analyze_dwt(sig, fs=8000, wavelet='db4', level=4):
    """Performs Standard DWT and returns level energies."""
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # coeffs layout: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    
    energies = []
    labels = []
    
    nyquist = fs / 2
    
    # Details from fine to coarse
    for i in range(level, 0, -1):
        idx = level - i + 1
        energy = calculate_energy(coeffs[idx])
        energies.append(energy)
        
        f_low = nyquist / (2**i)
        f_high = nyquist / (2**(i-1))
        labels.append(f"D{i} ({f_low:.0f}-{f_high:.0f} Hz)")
        
    # Approximation
    energy = calculate_energy(coeffs[0])
    energies.append(energy)
    f_high = nyquist / (2**level)
    labels.append(f"A{level} (0-{f_high:.0f} Hz)")
    
    # Reverse so A is first, D follows
    return energies[::-1], labels[::-1], coeffs

def detect_dtmf_wpt(sig, fs=8000, wavelet='db4', level=7):
    """
    Uses Wavelet Packet Transform (WPT) to detect DTMF frequencies.
    Level 7 yields 128 nodes. 
    Nyquist = 4000 Hz.
    Bandwidth per node = 4000 / 128 = 31.25 Hz.
    This provides enough resolution to distinguish DTMF frequencies.
    """
    wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, mode='symmetric')
    
    # Get nodes at specified level, ordered by frequency
    nodes = get_freq_ordered_nodes(wp, level)
    
    energies = np.zeros(len(nodes))
    node_centers = np.zeros(len(nodes))
    
    for i, node in enumerate(nodes):
        energies[i] = calculate_energy(node.data)
        _, _, f_center = node_to_freq_range(i, level, fs)
        node_centers[i] = f_center
        
    # We define tolerance bands for our search based on DTMF frequencies
    low_freqs = DTMF_FREQUENCIES['L']
    high_freqs = DTMF_FREQUENCIES['H']
    
    # Find max energy near expected low frequencies
    max_low_energy = 0
    detected_low_f = 0
    best_low_dtmf = None
    
    for target_f in low_freqs:
        # Check nodes near this target frequency
        for i, center_f in enumerate(node_centers):
            if abs(center_f - target_f) < 40:  # Search window (+- 40Hz)
                if energies[i] > max_low_energy:
                    max_low_energy = energies[i]
                    detected_low_f = center_f
                    best_low_dtmf = target_f

    # Find max energy near expected high frequencies
    max_high_energy = 0
    detected_high_f = 0
    best_high_dtmf = None
    
    for target_f in high_freqs:
        # Check nodes near this target frequency
        for i, center_f in enumerate(node_centers):
            if abs(center_f - target_f) < 40:  # Search window (+- 40Hz)
                if energies[i] > max_high_energy:
                    max_high_energy = energies[i]
                    detected_high_f = center_f
                    best_high_dtmf = target_f

    detected_digit = None
    if best_low_dtmf and best_high_dtmf:
        detected_digit = DTMF_TABLE.get((best_low_dtmf, best_high_dtmf), "Unknown")
        
    results = {
        'digit': detected_digit,
        'low_f_dtmf': best_low_dtmf,
        'high_f_dtmf': best_high_dtmf,
        'low_f_node_center': detected_low_f,
        'high_f_node_center': detected_high_f,
        'node_energies': energies,
        'node_centers': node_centers
    }
    
    return results

# ==========================================
# PART 5: VISUALIZATION
# ==========================================

def plot_analysis(t, sig_clean, sig_noisy, dwt_energies, dwt_labels, wpt_results, fs):
    """Creates a comprehensive visualization of the Wavelet analysis."""
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Clean Time Domain Signal
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t[:200], sig_clean[:200]) # Plot a small segment
    ax1.set_title("Clean DTMF Signal (Zoomed)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    
    # 2. Noisy Time Domain Signal
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(t[:200], sig_noisy[:200], color='orange')
    ax2.set_title("Noisy DTMF Signal (Zoomed)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    
    # 3. DWT Energy Distribution
    ax3 = plt.subplot(3, 2, 3)
    x_pos = np.arange(len(dwt_labels))
    ax3.bar(x_pos, dwt_energies, align='center', alpha=0.7, color='green')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dwt_labels, rotation=45, ha="right")
    ax3.set_title("Standard DWT Energy per Level (db4)")
    ax3.set_ylabel("Energy")
    ax3.grid(True, axis='y')
    
    # 4. WPT Energy Distribution
    ax4 = plt.subplot(3, 2, 4)
    energies = wpt_results['node_energies']
    centers = wpt_results['node_centers']
    ax4.plot(centers, energies, color='purple', label="WPT Band Energy")
    ax4.fill_between(centers, energies, color='purple', alpha=0.3)
    ax4.axvline(wpt_results['low_f_dtmf'], color='red', linestyle='--', label=f"Detected Low: {wpt_results['low_f_dtmf']} Hz")
    ax4.axvline(wpt_results['high_f_dtmf'], color='blue', linestyle='--', label=f"Detected High: {wpt_results['high_f_dtmf']} Hz")
    ax4.set_title("Wavelet Packet Transform Energy Spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Energy")
    ax4.legend()
    ax4.grid(True)
    
    # 5. CWT Scalogram (Optional Bonus Visualization)
    ax5 = plt.subplot(3, 1, 3)
    scales = np.arange(1, 128)
    # Using 'cmor' (Complex Morlet) which is excellent for frequency extraction
    coeffs, freqs = pywt.cwt(sig_noisy[:800], scales, 'cmor1.5-1.0', sampling_period=1/fs)
    img = ax5.imshow(np.abs(coeffs), extent=[0, t[800], freqs[-1], freqs[0]], aspect='auto', cmap='jet')
    ax5.set_title("Continuous Wavelet Transform (CWT) Scalogram")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax5, label="Magnitude")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# PART 6: MAIN WORKFLOW AND VALIDATION
# ==========================================

def run_experiment(digit='5', snr_db=10, fs=8000):
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: Detecting DTMF '{digit}' at SNR {snr_db} dB")
    print(f"{'='*50}")
    
    # 1. Generate Signal
    t, sig_clean, expected_freqs = generate_dtmf_signal(digit, fs=fs)
    print(f"Expected Digit: {digit}")
    print(f"Expected Frequencies: {expected_freqs[0]} Hz, {expected_freqs[1]} Hz")
    
    # 2. Add Noise
    sig_noisy = add_awgn_noise(sig_clean, snr_db)
    
    # 3. DWT Analysis
    dwt_energies, dwt_labels, _ = analyze_dwt(sig_noisy, fs=fs, level=4)
    
    # 4. WPT Analysis & Detection
    wpt_results = detect_dtmf_wpt(sig_noisy, fs=fs, wavelet='db4', level=7)
    
    # 5. Output Results
    print(f"\nDetected Low Frequency: {wpt_results['low_f_dtmf']} Hz (Node Center: {wpt_results['low_f_node_center']:.1f} Hz)")
    print(f"Detected High Frequency: {wpt_results['high_f_dtmf']} Hz (Node Center: {wpt_results['high_f_node_center']:.1f} Hz)")
    print(f"\nDetected Digit: {wpt_results['digit']}")
    
    status = "SUCCESS" if wpt_results['digit'] == digit else "FAILURE"
    print(f"Status: {status}")
    print(f"{'-'*50}\n")
    
    # 6. Visualization
    plot_analysis(t, sig_clean, sig_noisy, dwt_energies, dwt_labels, wpt_results, fs)

def analyze_audio_file(filepath):
    """Processes an external WAV file containing a DTMF tone."""
    print(f"\nAnalyzing Audio File: {filepath}")
    sig, fs = sf.read(filepath)
    
    # Convert stereo to mono if needed
    if len(sig.shape) > 1:
        sig = np.mean(sig, axis=1)
        
    # Resample to 8000 Hz if necessary
    target_fs = 8000
    if fs != target_fs:
        num_samples = int(len(sig) * float(target_fs) / fs)
        sig = signal.resample(sig, num_samples)
        fs = target_fs
        
    # Normalize signal
    sig = sig / np.max(np.abs(sig))
    
    # Detect using WPT
    wpt_results = detect_dtmf_wpt(sig, fs=fs, wavelet='db4', level=7)
    
    print(f"Detected Low Frequency: {wpt_results['low_f_dtmf']} Hz")
    print(f"Detected High Frequency: {wpt_results['high_f_dtmf']} Hz")
    print(f"Detected Digit: {wpt_results['digit']}")

if __name__ == "__main__":
    run_experiment(digit='5', snr_db=10)
    run_experiment(digit='A', snr_db=5)
    run_experiment(digit='#', snr_db=0)