"""
Noise Remover
==============
Provides a class-based interface for removing noise from DTMF signals
using multiple denoising strategies.

Methods implemented:
  1. Spectral Gating           — zero out FFT bins below an energy threshold
  2. Wavelet Thresholding      — soft-threshold DWT detail coefficients
  3. Bandpass Filtering         — keep only the DTMF frequency bands
  4. Moving Average (Smoothing) — simple time-domain low-pass approximation
"""

import numpy as np
from scipy.signal import butter, sosfilt


class NoiseRemover:
    """Remove noise from DTMF signals using various algorithms.

    Attributes:
        sample_rate: Sampling frequency of the signals (Hz).
    """

    def __init__(self, sample_rate: int = 8000):
        """Initialise the noise remover.

        Args:
            sample_rate: Sampling frequency in Hz (default 8000 for telephony).
        """
        self.sample_rate = sample_rate

    # ── 1.  Spectral Gating ──────────────────────────────────────

    def spectral_gate(self, signal: np.ndarray,
                      threshold_factor: float = 0.1) -> np.ndarray:
        """Remove noise by zeroing weak FFT bins (spectral gating).

        Keeps only frequency bins whose magnitude exceeds a fraction
        of the peak magnitude.

        Args:
            signal:            Noisy 1-D signal.
            threshold_factor:  Fraction of peak magnitude to use as the
                               gate threshold (0.0 – 1.0).

        Returns:
            Denoised signal (real-valued, same length as input).
        """
        spectrum = np.fft.fft(signal)
        magnitudes = np.abs(spectrum)
        threshold = threshold_factor * np.max(magnitudes)

        # Zero out bins below the threshold
        spectrum[magnitudes < threshold] = 0

        return np.real(np.fft.ifft(spectrum))

    # ── 2.  Wavelet Thresholding ─────────────────────────────────

    def wavelet_denoise(self, signal: np.ndarray,
                        wavelet: str = 'db4',
                        level: int = 4,
                        mode: str = 'soft') -> np.ndarray:
        """Remove noise using wavelet (DWT) coefficient thresholding.

        Uses the universal threshold (VisuShrink):
            λ = σ * √(2 ln N)
        where σ is estimated from the finest detail coefficients
        using the MAD estimator.

        Args:
            signal:  Noisy 1-D signal.
            wavelet: Wavelet family (e.g. 'db4', 'sym6', 'coif3').
            level:   Decomposition depth.
            mode:    'soft' or 'hard' thresholding.

        Returns:
            Denoised signal.
        """
        import pywt

        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Estimate noise σ from finest detail coefficients (MAD estimator)
        detail_1 = coeffs[-1]
        sigma = np.median(np.abs(detail_1)) / 0.6745

        # Universal threshold (VisuShrink)
        N = len(signal)
        threshold = sigma * np.sqrt(2 * np.log(N))

        # Threshold all detail coefficients (keep the approximation intact)
        denoised_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            if mode == 'soft':
                denoised = pywt.threshold(detail, threshold, mode='soft')
            else:
                denoised = pywt.threshold(detail, threshold, mode='hard')
            denoised_coeffs.append(denoised)

        return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]

    # ── 3.  Bandpass Filtering ───────────────────────────────────

    def bandpass_filter(self, signal: np.ndarray,
                        low_cut: float = 650.0,
                        high_cut: float = 1700.0,
                        order: int = 5) -> np.ndarray:
        """Remove noise using a Butterworth bandpass filter.

        The default band (650–1700 Hz) covers all eight standard DTMF
        frequencies: 697, 770, 852, 941 Hz (low group) and
        1209, 1336, 1477, 1633 Hz (high group).

        Args:
            signal:    Noisy 1-D signal.
            low_cut:   Lower cutoff frequency (Hz).
            high_cut:  Upper cutoff frequency (Hz).
            order:     Filter order.

        Returns:
            Bandpass-filtered signal.
        """
        nyquist = self.sample_rate / 2.0
        low = low_cut / nyquist
        high = high_cut / nyquist

        sos = butter(order, [low, high], btype='band', output='sos')
        return sosfilt(sos, signal)

    # ── 4.  Moving Average (Smoothing) ───────────────────────────

    def moving_average(self, signal: np.ndarray,
                       window_size: int = 5) -> np.ndarray:
        """Smooth a signal using a simple moving average filter.

        This is a basic time-domain low-pass filter.  Effective for
        suppressing high-frequency noise but may distort DTMF tones
        if the window is too large.

        Args:
            signal:       Noisy 1-D signal.
            window_size:  Number of samples in the averaging window.

        Returns:
            Smoothed signal (same length, edges padded with originals).
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')

