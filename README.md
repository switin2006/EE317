# DTMF Signal Detection & Decoding
**DSP Course Project (EE317)**

This repository contains the codebase for our DSP project on detecting and decoding Dual-Tone Multi-Frequency (DTMF) signals using various algorithms (FFT, Goertzel, and Wavelet).

---

## 🚀 Getting Started for Team Members

The repository is structured to keep our generation logic separate from our detection logic so we can work in parallel without merge conflicts.

### 1. Repository Structure
- `src/generation/` ➔ Contains the tools for generating clean and noisy DTMF tones. **(Done!)**
- `src/detection/` ➔ This is where we will build out our 3 detection algorithms.
- `demo_samples/` ➔ Output directory for generated `.wav` files (ignored in git).
- `generate_demo.py` ➔ A quick script to test signal generation.

### 2. How to Generate Signals for Your Algorithm

Before you write your detection algorithm, you need audio to test it on. The `dtmf_generator.py` file has been written to make this extremely easy.

In your detection scripts, simply import the generator like this:

```python
from src.generation.dtmf_generator import generate_dtmf_tone, generate_dtmf_sequence
from src.generation.noise_utils import add_white_noise

sample_rate = 8000

# 1. Generate a single 200ms tone for the number '5'
clean_5 = generate_dtmf_tone('5', sample_rate=sample_rate)

# 2. Generate a full phone number sequence with silences
sequence = generate_dtmf_sequence('5551234', sample_rate=sample_rate)

# 3. Add heavy background noise to test your algorithm's robustness
noisy_sequence = add_white_noise(sequence, snr_db=10.0)
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
