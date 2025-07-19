# file: evaluate_models.py
from utils.metrics import word_error_rate, mel_cepstral_distance
from models.asr_model import ASRModel
from models.tts_model import TTSModel
import torch

# Load trained models (paths assumed)
asr_model = ASRModel(...); asr_model.load_state_dict(torch.load("asr_model.pt")); asr_model.eval()
tts_model = TTSModel(...); tts_model.load_state_dict(torch.load("tts_model.pt")); tts_model.eval()

# Example ASR evaluation on a validation sample
val_sample_audio = "dev_sample.wav"
# (Load audio and get reference transcript from dataset or ground truth)
# ... (use similar steps as infer_asr.py to get predicted text)
pred_text = "PREDICTED TRANSCRIPT"
ref_text = "REFERENCE TRANSCRIPT"
wer = word_error_rate(ref_text, pred_text)
print(f"WER: {wer*100:.2f}%")

# Example TTS evaluation on a validation sentence
val_text = "This is a validation sentence."
ref_mel = ...  # ground truth mel spectrogram for this sentence from dataset
# Synthesize mel with model
text_seq = torch.LongTensor(text_to_sequence(val_text.upper())).unsqueeze(0)
text_len = torch.LongTensor([text_seq.size(1)])
with torch.no_grad():
    pred_mel = tts_model(text_seq, text_len, mel=None)
mcd = mel_cepstral_distance(ref_mel, pred_mel.squeeze(0))
print(f"MCD: {mcd:.2f} dB")
