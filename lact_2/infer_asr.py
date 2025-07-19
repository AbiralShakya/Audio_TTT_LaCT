# file: infer_asr.py
import torch
from models.asr_model import ASRModel
from utils.text import sequence_to_text
import torchaudio

# Load trained ASR model (assuming we have a saved checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASRModel(input_dim=80, d_model=256, num_layers=6, num_heads=8, window_size=8, chunk_size=512, vocab_size=33)
model.load_state_dict(torch.load("asr_model.pt", map_location=device))
model.eval().to(device)

# Audio file for inference
audio_path = "example.wav"
waveform, sr = torchaudio.load(audio_path)
# Resample if needed
if sr != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    sr = 16000
# Compute log-mel spectrogram as in training
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
if waveform.dim() > 1:
    waveform = waveform[0:1, :]
mel_spec = mel_transform(waveform)
mel_spec = amp_to_db(mel_spec)
mel_spec = mel_spec.transpose(0, 1)  # (time, n_mels)
# Prepare input for model
mel_length = torch.tensor([mel_spec.shape[0]], dtype=torch.long)
mel_spec = mel_spec.unsqueeze(0)  # shape (1, T, n_mels)
mel_spec = mel_spec.to(device)
mel_length = mel_length.to(device)
# Forward pass to get log-probabilities
with torch.no_grad():
    log_probs = model(mel_spec, mel_length)  # shape (T, 1, vocab)
# Greedy CTC decoding
pred_indices = log_probs.argmax(dim=-1).squeeze(1).tolist()  # list of length T
# Collapse repeats and remove blanks (index 0 is blank)
decoded_indices = []
prev_idx = None
for idx in pred_indices:
    if idx != 0 and idx != prev_idx:
        decoded_indices.append(idx)
    prev_idx = idx
# Convert index sequence to text
transcription = sequence_to_text(decoded_indices)
print("Transcription:", transcription)
