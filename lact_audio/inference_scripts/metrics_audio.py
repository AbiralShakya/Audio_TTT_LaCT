import torchaudio
import numpy as np
from scipy.signal import resample
from scipy.io import wavfile

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: pesq package not found. PESQ metric will not be computed.")

# FAD requires tensorflow and a pretrained VGGish model; placeholder for now
# from tensorflow_audio_fad import score as fad_score

def compute_snr(clean, test):
    noise = clean - test
    return 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))

def main():
    # Paths to generated files
    baseline_path = "musicgen_baseline_output.wav"
    lact_path = "musicgen_lact_output.wav"
    # Load audio
    sr1, baseline = wavfile.read(baseline_path)
    sr2, lact = wavfile.read(lact_path)
    assert sr1 == sr2, "Sampling rates must match"
    # Normalize
    baseline = baseline.astype(np.float32) / np.max(np.abs(baseline))
    lact = lact.astype(np.float32) / np.max(np.abs(lact))
    # For demo, use baseline as 'clean' reference
    snr = compute_snr(baseline, lact)
    print(f"SNR (baseline vs LaCT): {snr:.2f} dB")
    # PESQ (narrowband mode, 16kHz)
    if PESQ_AVAILABLE:
        if sr1 != 16000:
            baseline = resample(baseline, int(len(baseline) * 16000 / sr1))
            lact = resample(lact, int(len(lact) * 16000 / sr1))
            sr1 = 16000
        pesq_score = pesq(sr1, baseline, lact, 'nb')
        print(f"PESQ: {pesq_score:.3f}")
    else:
        print("PESQ: Not computed (pesq package not installed)")
    # FAD: Placeholder (requires VGGish and TensorFlow)
    print("FAD: Not computed (requires TensorFlow and VGGish)")

if __name__ == "__main__":
    main() 