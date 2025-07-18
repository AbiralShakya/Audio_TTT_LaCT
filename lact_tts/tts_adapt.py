import torch
from tts_preprocess import LJSpeechDataset, tts_collate_fn, tts_char_to_idx
from tts_model import TTSModel

# Load pre-trained TTS model
model = TTSModel(vocab_size=len(tts_char_to_idx)+1)
model.load_state_dict(torch.load("tts_model_lact.pth", map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare for adaptation: freeze text encoder (text invariances) and audio encoder, adapt decoder only
model.train()
for param in model.embedding.parameters():
    param.requires_grad = False
for param in model.text_encoder.parameters():
    param.requires_grad = False
for param in model.audio_encoder.parameters():
    param.requires_grad = False

# Get an adaptation audio sample (e.g., from a different speaker or environment).
# Here we use a random sample from LJSpeech for demonstration (in practice, use a new speaker sample).
dataset = LJSpeechDataset(root_dir="path/to/LJSpeech-1.1")
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=tts_collate_fn)
text_batch, text_lengths, mel_batch, mel_lengths = next(iter(loader))
mel_batch = mel_batch.to(device)
mel_lengths = mel_lengths.to(device)
# (We ignore text_batch because we assume no transcript for adaptation audio.)

# Optimizer for adaptation (only decoder parameters are trainable)
adapt_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(adapt_params, lr=1e-4)

# Perform adaptation steps using self-supervised reconstruction
adapt_steps = 10
for step in range(adapt_steps):
    # Create a noisy version of the mel (mask 20% of frames) for denoising training
    mel_noisy = mel_batch.clone()
    L = mel_lengths.item()
    num_mask = max(1, int(0.2 * L))
    mask_idx = torch.randperm(L)[:num_mask]
    mel_noisy[0, mask_idx, :] = 0.0
    # Encode and decode with teacher forcing
    enc_out, enc_mask = model.forward_audio_encoder(mel_noisy, mel_lengths)
    recon_pred, recon_stop_pred = model.decode(enc_out, enc_mask, target_mel=mel_batch, teacher_forcing=True)
    # Compute reconstruction loss (L1 on masked frames + stop token loss)
    mask_bool = (mel_noisy == 0)
    frame_mask = mask_bool[0].any(dim=-1)
    if frame_mask.any():
        pred_masked = recon_pred[0, frame_mask, :]
        tgt_masked = mel_batch[0, frame_mask, :]
        mel_loss = torch.nn.functional.l1_loss(pred_masked, tgt_masked)
    else:
        mel_loss = torch.tensor(0.0, device=device)
    stop_target = torch.zeros_like(recon_stop_pred)
    stop_target[0, mel_lengths.item()-1] = 1.0
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(recon_stop_pred, stop_target)
    loss = mel_loss + stop_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Synthesize speech from text using the adapted model
model.eval()
test_text = "This is a test of adaptation."
text_idx = [tts_char_to_idx[c] for c in test_text if c in tts_char_to_idx]
text_tensor = torch.tensor(text_idx, dtype=torch.long).unsqueeze(0).to(device)
with torch.no_grad():
    mel_out = model.infer(text_tensor)
print("Adaptation complete. Generated mel spectrogram shape:", mel_out.shape)
# (Use a vocoder on mel_out to generate waveform and listen to evaluate the voice adaptation.)
