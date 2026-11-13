# DTMF Signal Detection & Decoding
**DSP Course Project (EE317)**

This repository contains the codebase for our DSP project on detecting and decoding Dual-Tone Multi-Frequency (DTMF) signals using various algorithms (FFT, Goertzel, and Wavelet).

---

## 🚀 Getting Started for Team Members

The repository is structured to keep our generation logic separate from our detection logic so we can work in parallel without merge conflicts.

### 1. Repository Structure
- `src/generation/` ➔ Tools for generating clean DTMF tones and multi-key sequences.
- `src/noise/` ➔ Class-based noise generation (`NoiseGenerator`) and removal (`NoiseRemover`).
- `src/detection/` ➔ Standardized DTMF detection algorithms inheriting from `DTMFDetector`:
  - `goertzel/` ➔ Goertzel IIR filter detection.
  - `FFT/` ➔ Fast Fourier Transform peak detection.
  - `Wavelet/` ➔ Wavelet Packet Transform (WPT) multi-band energy detection.
- `tests/` ➔ Standardized test suites (single tone, sequence, noise).
- `demo_samples/` ➔ Output directory for generated `.wav` audio files.
- `generate_demo.py` ➔ Script to generate and save sample `.wav` files.
- `generate_plot.py` ➔ Script to compute and save Goertzel power spectrum plots.

### 2. How to Generate Signals & Noise

```python
from src.generation.dtmf_generator import generate_dtmf_tone, generate_dtmf_sequence
from src.noise import NoiseGenerator, NoiseRemover

sample_rate = 8000

# 1. Generate clean single tone or sequence
clean_5 = generate_dtmf_tone('5', sample_rate=sample_rate)
sequence = generate_dtmf_sequence('5551234', sample_rate=sample_rate)

# 2. Add noise (White Gaussian, Pink, or Impulse)
ng = NoiseGenerator(seed=42)
noisy_sequence = ng.add_white_noise(sequence, snr_db=5.0)

# 3. Clean noise using one of the denoising methods
remover = NoiseRemover(sample_rate=sample_rate)
cleaned_sequence = remover.spectral_gate(noisy_sequence)
# or: remover.bandpass_filter(noisy_sequence)
# or: remover.wavelet_denoise(noisy_sequence)
```

---

## 🧪 Running the Standardized Test Suite

Each algorithm and the noise engine have isolated, synchronized test suites inside `tests/`:

```bash
# Goertzel Tests
python -m tests.Goertzel.test_goertzel_single
python -m tests.Goertzel.test_goertzel_multi

# FFT Tests
python -m tests.FFT.test_fft_single
python -m tests.FFT.test_fft_multi

# Wavelet Tests
python -m tests.Wavelet.test_wavelet_single
python -m tests.Wavelet.test_wavelet_multi

# Noise Generation & Removal Benchmarks
python -m tests.Noise.test_noise_generation
python -m tests.Noise.test_noise_removal
```

### 3. Running the Generator Demo

If you want to hear what the signals sound like, run the demo script from the root folder:

```bash
python generate_demo.py
```

This will create `clean_5.wav`, `clean_sequence.wav`, and `noisy_sequence_5db.wav` in the `demo_samples/` folder for you to listen to.

---

## 🛠 How to Build Your Detection Algorithm

We have defined a standard interface that all algorithms must follow. This guarantees they will plug into our final benchmark perfectly.

1. **Create your folder:** Inside `src/detection/`, create a folder for your specific algorithm (e.g., `src/detection/fft/`, `src/detection/goertzel/`).
2. **Inherit from the Base Class:** In your python file, import `DTMFDetector` from `detector_base.py` and implement the required methods.

### Example Skeleton for Your Algorithm:

```python
import numpy as np
from src.detection.detector_base import DTMFDetector

class MyAwesomeDetector(DTMFDetector):
    
    @property
    def name(self) -> str:
        return "My Awesome Algorithm"

    def detect_single(self, signal: np.ndarray, sample_rate: int) -> str:
        # 1. Do your math/DSP here
        # 2. Use self._find_closest_freq() if needed
        # 3. Return the key (e.g., '5')
        pass

    def detect_sequence(self, signal: np.ndarray, sample_rate: int) -> str:
        # You can use self._segment_signal(signal, sample_rate) 
        # to split the sequence into chunks automatically!
        pass
```
