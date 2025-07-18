import torch
import torchaudio
import numpy as np

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer package not found. WER/CER metrics will not be computed.")

# Feature extraction: log-mel spectrogram
class LogMelFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, normalized=True
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    def __call__(self, waveform):
        mel = self.mel(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel.transpose(0, 1)  # (time, mel)

# Metrics

def compute_wer(ref, hyp):
    if JIWER_AVAILABLE:
        return wer(ref, hyp)
    else:
        print("WER: Not computed (jiwer package not installed)")
        return None

def compute_cer(ref, hyp):
    if JIWER_AVAILABLE:
        return cer(ref, hyp)
    else:
        print("CER: Not computed (jiwer package not installed)")
        return None

# Hardware metrics (simple GPU memory/throughput logger)
def log_hardware_metrics():
    import torch
    if torch.cuda.is_available():
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available.") 