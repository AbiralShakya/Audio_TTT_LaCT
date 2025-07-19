# file: synthesize_tts.py
import torch
from models.tts_model import TTSModel
from utils.text import text_to_sequence
import torchaudio

# Load trained TTS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TTSModel(vocab_size=33, d_model=256, num_encoder_layers=3, num_decoder_layers=6, num_heads=8, window_size=8, chunk_size=512, n_mels=80)
model.load_state_dict(torch.load("tts_model.pt", map_location=device))
model.eval().to(device)

# Input text to synthesize
input_text = "Hello world. This is a test of the LaCT TTS model."
# Convert text to sequence of token indices
text_seq = torch.LongTensor(text_to_sequence(input_text.upper())).unsqueeze(0).to(device)
text_len = torch.LongTensor([text_seq.size(1)]).to(device)
# Run inference
with torch.no_grad():
    generated_mel = model(text_seq, text_len, mel=None)  # auto-regressive generation
# generated_mel: shape (1, T_gen, n_mels)
generated_mel = generated_mel.squeeze(0).cpu()  # (T_gen, n_mels)
# Invert mel-spectrogram to waveform using Griffin-Lim (as no neural vocoder is provided)
n_fft = 1024
n_mels = generated_mel.size(1)
# Convert from dB to linear
mel_linear = torch.pow(10.0, generated_mel / 10.0)
# Create mel filter bank matrix for inversion
mel_fb = torchaudio.functional.create_fb_matrix(n_fft//2 + 1, f_min=0.0, f_max=8000.0 if hasattr(model, 'sample_rate') else 11025.0, 
                                               n_mels=n_mels, sample_rate=22050)
# Pseudo-inverse of mel filter bank
mel_fb_pinv = torch.pinverse(mel_fb)
# Estimate linear spectrogram (magnitude)
spec_est = torch.clamp(mel_linear @ mel_fb_pinv.T, min=1e-10)  # (T_gen, n_fft//2+1)
spec_est = spec_est.transpose(0, 1)  # shape (freq_bins, time)
# Griffin-Lim inversion to waveform
waveform = torchaudio.functional.griffinlim(spec_est, n_fft=n_fft, hop_length=256, win_length=1024, power=1.0, n_iter=32)
# Save or play the waveform
torchaudio.save("output.wav", waveform.unsqueeze(0), 22050)
print("Synthesized speech saved to output.wav")
