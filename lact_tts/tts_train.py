import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tts_preprocess import LJSpeechDataset, tts_collate_fn, tts_char_to_idx
from tts_model import TTSModel

# Hyperparameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 5

# Load LJSpeech dataset and DataLoader
train_set = LJSpeechDataset(root_dir="path/to/LJSpeech-1.1")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=tts_collate_fn)

# Initialize model and optimizer
model = TTSModel(vocab_size=len(tts_char_to_idx)+1)  # +1 for padding index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss functions
mel_loss_fn = nn.L1Loss()
stop_loss_fn = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(1, num_epochs+1):
    total_mel_loss = 0.0
    total_stop_loss = 0.0
    for text_batch, text_lengths, mel_batch, mel_lengths in train_loader:
        text_batch = text_batch.to(device)
        mel_batch = mel_batch.to(device)
        text_lengths = text_lengths.to(device)
        mel_lengths = mel_lengths.to(device)
        # Forward text encoder and decode with teacher forcing (TTS task)
        text_enc_out, text_enc_mask = model.forward_text_encoder(text_batch, text_lengths)
        mel_pred, stop_pred = model.decode(text_enc_out, text_enc_mask, target_mel=mel_batch, teacher_forcing=True)
        # Compute supervised TTS loss
        max_T = mel_pred.size(1)
        out_mask = (torch.arange(max_T, device=device)[None, :] < mel_lengths[:, None])  # mask for valid mel outputs
        # Mel L1 loss on valid output frames
        mel_pred_masked = mel_pred[out_mask.unsqueeze(-1)].reshape(-1, mel_pred.size(-1))
        mel_tgt_masked = mel_batch[out_mask.unsqueeze(-1)].reshape(-1, mel_batch.size(-1))
        mel_loss = mel_loss_fn(mel_pred_masked, mel_tgt_masked)
        # Stop token loss (target = 1 at end-of-sequence frame, 0 otherwise)
        stop_target = torch.zeros_like(stop_pred)
        for i, L in enumerate(mel_lengths):
            if L > 0:
                stop_target[i, L-1] = 1.0
        stop_loss = stop_loss_fn(stop_pred, stop_target)
        # Forward audio encoder and decode (Reconstruction task with masked input)
        # Create noisy mel input by masking 20% of frames
        mel_noisy = mel_batch.clone()
        for i in range(mel_noisy.size(0)):
            L = mel_lengths[i].item()
            num_mask = max(1, int(0.2 * L))
            mask_idx = torch.randperm(L)[:num_mask]
            mel_noisy[i, mask_idx, :] = 0.0
        aud_enc_out, aud_enc_mask = model.forward_audio_encoder(mel_noisy, mel_lengths)
        recon_pred, recon_stop_pred = model.decode(aud_enc_out, aud_enc_mask, target_mel=mel_batch, teacher_forcing=True)
        # Compute reconstruction loss (focus on masked frames)
        recon_mask = (mel_noisy == 0)  # boolean mask of where frames were zeroed
        frame_mask = recon_mask.any(dim=-1)   # (batch, T) True for frames that were masked
        if frame_mask.any():
            recon_pred_masked = recon_pred[frame_mask.unsqueeze(-1)].reshape(-1, recon_pred.size(-1))
            recon_tgt_masked = mel_batch[frame_mask.unsqueeze(-1)].reshape(-1, mel_batch.size(-1))
            recon_mel_loss = mel_loss_fn(recon_pred_masked, recon_tgt_masked)
        else:
            recon_mel_loss = torch.tensor(0.0, device=device)
        # Stop token loss for reconstruction (should match original stop pattern)
        recon_stop_loss = stop_loss_fn(recon_stop_pred, stop_target)
        # Total multi-task loss
        loss = mel_loss + stop_loss + recon_mel_loss + recon_stop_loss
        optimizer.zero_grad()
        loss.backward()
        # (Optionally clip gradients for stability)
        optimizer.step()
        total_mel_loss += mel_loss.item()
        total_stop_loss += stop_loss.item()
    print(f"Epoch {epoch}: Avg Mel Loss = {total_mel_loss/len(train_loader):.3f}, Avg Stop Loss = {total_stop_loss/len(train_loader):.3f}")

# Save model
torch.save(model.state_dict(), "tts_model_lact.pth")

# Usage:
# After training, to synthesize speech with the model:
# model = TTSModel(vocab_size=len(tts_char_to_idx)+1); model.load_state_dict(torch.load("tts_model_lact.pth"))
# model = model.to(device)
# text = "Hello, world!"
# text_tensor = torch.tensor([tts_char_to_idx[c] for c in text if c in tts_char_to_idx], 
#                            dtype=torch.long).unsqueeze(0).to(device)
# mel_output = model.infer(text_tensor)
# (Use a vocoder to convert mel_output to audio waveform.)
