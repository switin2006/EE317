def run_batch_test(snr_levels=[20, 10, 0, -5], fs=8000):
    """
    Tests the WPT DTMF detection algorithm against all 16 digits
    across multiple Signal-to-Noise Ratios (SNR).
    """
    print("\n" + "="*50)
    print("STARTING BATCH STRESS TEST")
    print("="*50)
    
    digits = list(DIGIT_TO_FREQ.keys())
    
    for snr in snr_levels:
        correct_count = 0
        total = len(digits)
        
        for digit in digits:
            # 1. Generate clean signal
            t, sig_clean, _ = generate_dtmf_signal(digit, fs=fs, duration=0.2)
            
            # 2. Add noise
            sig_noisy = add_awgn_noise(sig_clean, snr)
            
            # 3. Detect
            wpt_results = detect_dtmf_wpt(sig_noisy, fs=fs, wavelet='db4', level=7)
            
            # 4. Tally results
            if wpt_results['digit'] == digit:
                correct_count += 1
                
        accuracy = (correct_count / total) * 100
        print(f"SNR: {snr:3d} dB | Accuracy: {accuracy:6.2f}% ({correct_count}/{total} digits detected)")

if __name__ == "__main__":
    # Run the visual experiment for one digit first
    run_experiment(digit='5', snr_db=5)
    
    # Run the automated stress test for all digits
    run_batch_test()