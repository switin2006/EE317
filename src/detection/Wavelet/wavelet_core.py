"""
Wavelet Core Utilities
=======================
Low-level wavelet functions used by the WaveletDetector:
WPT frequency ordering, node-to-Hz mapping, and energy computation.
"""

import numpy as np
import pywt


def get_freq_ordered_nodes(wp, level):
    """Returns the WPT nodes ordered strictly by frequency.

    In Wavelet Packets, high-pass and low-pass filtering can cause
    frequency band swapping.  This function uses pywt's built-in
    order="freq" to sort the natural node paths into sequential
    frequency bands.
    """
    return wp.get_level(level, order="freq")


def node_to_freq_range(node_index: int, level: int, fs: int):
    """Calculate the frequency range (Hz) for a frequency-ordered node.

    Returns:
        (f_start, f_end, f_center) in Hz.
    """
    num_nodes = 2 ** level
    nyquist = fs / 2
    band_width = nyquist / num_nodes

    f_start = node_index * band_width
    f_end = (node_index + 1) * band_width
    f_center = (f_start + f_end) / 2

    return f_start, f_end, f_center


def calculate_energy(coeffs: np.ndarray) -> float:
    """Calculate the energy of an array of wavelet coefficients."""
    return float(np.sum(np.abs(coeffs) ** 2))


def analyze_dwt(sig: np.ndarray, fs: int = 8000, wavelet: str = 'db4',
                level: int = 4) -> tuple[list[float], list[str], list]:
    """Perform Standard DWT and return level energies and band labels.

    Args:
        sig:     1-D input signal.
        fs:      Sample rate in Hz.
        wavelet: Wavelet family (default: 'db4').
        level:   Decomposition level.

    Returns:
        (energies, labels, coeffs)
    """
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    energies = []
    labels = []
    nyquist = fs / 2

    # Details from fine to coarse
    for i in range(level, 0, -1):
        idx = level - i + 1
        energy = calculate_energy(coeffs[idx])
        energies.append(energy)

        f_low = nyquist / (2 ** i)
        f_high = nyquist / (2 ** (i - 1))
        labels.append(f"D{i} ({f_low:.0f}-{f_high:.0f} Hz)")

    # Approximation
    energy = calculate_energy(coeffs[0])
    energies.append(energy)
    f_high = nyquist / (2 ** level)
    labels.append(f"A{level} (0-{f_high:.0f} Hz)")

    # Reverse so Approximation is first, followed by Details
    return energies[::-1], labels[::-1], coeffs


def wpt_band_energies(signal: np.ndarray, fs: int = 8000,
                      wavelet: str = 'db4',
                      level: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Compute the energy in each WPT frequency band.

    Args:
        signal:  1-D input signal.
        fs:      Sample rate in Hz.
        wavelet: Wavelet family.
        level:   Decomposition depth (level 7 → 128 bands of 31.25 Hz).

    Returns:
        (energies, node_centers) — both 1-D arrays of length 2^level.
    """
    w = pywt.Wavelet(wavelet)
    min_len = (w.dec_len - 1) * (2 ** level)
    if len(signal) < min_len:
        signal = np.pad(signal, (0, min_len - len(signal)), mode='constant')

    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric')
    nodes = get_freq_ordered_nodes(wp, level)

    energies = np.zeros(len(nodes))
    node_centers = np.zeros(len(nodes))

    for i, node in enumerate(nodes):
        energies[i] = calculate_energy(node.data)
        _, _, f_center = node_to_freq_range(i, level, fs)
        node_centers[i] = f_center

    return energies, node_centers


def detect_dtmf_wpt(sig: np.ndarray, fs: int = 8000, wavelet: str = 'db4',
                    level: int = 7) -> dict:
    """Detailed DTMF detection via Wavelet Packet Transform.

    Returns the diagnostic dictionary containing the detected digit,
    identified frequencies, node centers, and the energy spectrum.
    """
    from src.detection.detector_base import DTMF_LOW_FREQS, DTMF_HIGH_FREQS, DTMF_TABLE

    energies, node_centers = wpt_band_energies(sig, fs=fs, wavelet=wavelet, level=level)

    max_low_energy = 0
    detected_low_f = 0
    best_low_dtmf = None

    for target_f in DTMF_LOW_FREQS:
        for i, center_f in enumerate(node_centers):
            if abs(center_f - target_f) < 40:
                if energies[i] > max_low_energy:
                    max_low_energy = energies[i]
                    detected_low_f = center_f
                    best_low_dtmf = target_f

    max_high_energy = 0
    detected_high_f = 0
    best_high_dtmf = None

    for target_f in DTMF_HIGH_FREQS:
        for i, center_f in enumerate(node_centers):
            if abs(center_f - target_f) < 40:
                if energies[i] > max_high_energy:
                    max_high_energy = energies[i]
                    detected_high_f = center_f
                    best_high_dtmf = target_f

    detected_digit = None
    if best_low_dtmf and best_high_dtmf:
        detected_digit = DTMF_TABLE.get((best_low_dtmf, best_high_dtmf), None)

    return {
        'digit': detected_digit,
        'low_f_dtmf': best_low_dtmf,
        'high_f_dtmf': best_high_dtmf,
        'low_f_node_center': detected_low_f,
        'high_f_node_center': detected_high_f,
        'node_energies': energies,
        'node_centers': node_centers,
    }


def plot_analysis(t: np.ndarray, sig_clean: np.ndarray, sig_noisy: np.ndarray,
                  dwt_energies: list, dwt_labels: list, wpt_results: dict,
                  fs: int = 8000, save_path: str | None = None):
    """Visualization of Wavelet analysis (time, DWT bars, WPT spectrum, CWT scalogram)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 12))

    # 1. Clean Time Domain
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t[:200], sig_clean[:200])
    ax1.set_title("Clean DTMF Signal (Zoomed)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # 2. Noisy Time Domain
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(t[:200], sig_noisy[:200], color='orange')
    ax2.set_title("Noisy DTMF Signal (Zoomed)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    # 3. DWT Energy
    ax3 = plt.subplot(3, 2, 3)
    x_pos = np.arange(len(dwt_labels))
    ax3.bar(x_pos, dwt_energies, align='center', alpha=0.7, color='green')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dwt_labels, rotation=45, ha="right")
    ax3.set_title("Standard DWT Energy per Level (db4)")
    ax3.set_ylabel("Energy")
    ax3.grid(True, axis='y')

    # 4. WPT Spectrum
    ax4 = plt.subplot(3, 2, 4)
    energies = wpt_results['node_energies']
    centers = wpt_results['node_centers']
    ax4.plot(centers, energies, color='purple', label="WPT Band Energy")
    ax4.fill_between(centers, energies, color='purple', alpha=0.3)
    if wpt_results['low_f_dtmf']:
        ax4.axvline(wpt_results['low_f_dtmf'], color='red', linestyle='--',
                    label=f"Detected Low: {wpt_results['low_f_dtmf']} Hz")
    if wpt_results['high_f_dtmf']:
        ax4.axvline(wpt_results['high_f_dtmf'], color='blue', linestyle='--',
                    label=f"Detected High: {wpt_results['high_f_dtmf']} Hz")
    ax4.set_title("Wavelet Packet Transform Energy Spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Energy")
    ax4.legend()
    ax4.grid(True)

    # 5. CWT Scalogram
    ax5 = plt.subplot(3, 1, 3)
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(sig_noisy[:min(800, len(sig_noisy))], scales, 'cmor1.5-1.0',
                             sampling_period=1 / fs)
    img = ax5.imshow(np.abs(coeffs),
                     extent=[0, t[min(800, len(t)) - 1], freqs[-1], freqs[0]],
                     aspect='auto', cmap='jet')
    ax5.set_title("Continuous Wavelet Transform (CWT) Scalogram")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax5, label="Magnitude")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
