import numpy as np
from src.detection.detector_base import (
    DTMFDetector, 
    DTMF_TABLE,
    DTMF_LOW_FREQS,
    DTMF_HIGH_FREQS
)

class FFTDetector(DTMFDetector):
    """DTMF detector using the Fast Fourier Transform (FFT)."""

    @property
    def name(self) -> str:
        return "FFT Algorithm"

    def detect_single(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a single DTMF key from a signal segment."""
        N = len(signal)
        if N == 0:
            return '?'
            
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
        
        if not np.any(low_band_mask) or not np.any(high_band_mask):
            return '?'
            
        # Find the peak frequency in each band
        peak_low = freqs[low_band_mask][np.argmax(fft_mag[low_band_mask])]
        peak_high = freqs[high_band_mask][np.argmax(fft_mag[high_band_mask])]
        
        # Snap the detected peaks to the closest standard frequencies
        matched_low = min(DTMF_LOW_FREQS, key=lambda f: abs(f - peak_low))
        matched_high = min(DTMF_HIGH_FREQS, key=lambda f: abs(f - peak_high))
        
        # Lookup the key in your team's dictionary
        detected_key = DTMF_TABLE.get((matched_low, matched_high), '?')
        
        return detected_key

    def detect_sequence(self, signal: np.ndarray, sample_rate: int) -> str:
        """Detect a sequence of DTMF keys from a longer signal."""
        frame_duration = 0.04
        energy_threshold = 0.05
        frame_length = int(sample_rate * frame_duration)
        
        decoded_sequence = []
        last_detected = None
        
        # Slide a window across the signal
        for i in range(0, len(signal) - frame_length, frame_length):
            frame = signal[i:i + frame_length]
            
            # 1. Energy Detection: Check if the frame is silence
            rms_energy = np.sqrt(np.mean(frame**2))
            if rms_energy < energy_threshold:
                last_detected = None # Reset when silence is found
                continue
                
            # 2. FFT Analysis
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
            
            matched_low = min(DTMF_LOW_FREQS, key=lambda f: abs(f - peak_low))
            matched_high = min(DTMF_HIGH_FREQS, key=lambda f: abs(f - peak_high))
            
            detected_key = DTMF_TABLE.get((matched_low, matched_high), None)
            
            # 3. Debouncing: Only add if it's a new key press
            if detected_key and detected_key != last_detected:
                decoded_sequence.append(detected_key)
                last_detected = detected_key
                
        return "".join(decoded_sequence)
