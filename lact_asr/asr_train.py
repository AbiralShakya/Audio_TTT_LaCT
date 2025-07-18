import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from asr_preprocess import LibriSpeechDataset, collate_fn, char_to_idx
from asr_model import ASRModel

# Hyperparameters
batch_size = 16
learning_rate = 1e-3
num_epochs = 10
ssl_loss_weight = 1.0  # weight for the self-supervised loss relative to ASR loss

# Load dataset (LibriSpeech train) and initialize DataLoader
train_set = LibriSpeechDataset(root_dir="path/to/librispeech", url="train-clean-100", download=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss functions, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASRModel(vocab_size=len(char_to_idx)+1).to(device)  # vocab = all chars + blank
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)       # CTC with blank index 0
recon_loss_fn = nn.L1Loss()                                # L1 loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(1, num_epochs+1):
    total_asr_loss = 0.0
    total_ssl_loss = 0.0
    for mel_batch, mel_lengths, tokens_concat, token_lengths in train_loader:
        mel_batch = mel_batch.to(device)
        tokens_concat = tokens_concat.to(device)
        mel_lengths = mel_lengths.to(device)
        token_lengths = token_lengths.to(device)
        # Create random masks for 15% of frames in each sample (for SSL loss)
        batch_mask = torch.zeros(mel_batch.size(0), mel_batch.size(1), dtype=torch.float, device=device)
        for i in range(mel_batch.size(0)):
            L = mel_lengths[i].item()
            num_mask = max(1, int(0.15 * L))
            mask_idx = torch.randperm(L, device=device)[:num_mask]
            batch_mask[i, mask_idx] = 1.0
        # Forward pass (with masked input for SSL)
        asr_logits, recon_output = model(mel_batch, mel_mask=batch_mask)
        # Compute CTC loss (requires log_softmax and time-major input)
        log_probs = torch.log_softmax(asr_logits, dim=-1)           # (batch, time, vocab)
        log_probs = log_probs.permute(1, 0, 2)                      # -> (time, batch, vocab)
        asr_loss = ctc_loss_fn(log_probs, tokens_concat, mel_lengths, token_lengths)
        # Reconstruction loss on masked frames only
        mask_expanded = batch_mask.unsqueeze(-1).bool()             # (batch, time, 1) boolean mask
        recon_pred_masked = recon_output[mask_expanded].reshape(-1, recon_output.size(-1))
        recon_tgt_masked = mel_batch[mask_expanded].reshape(-1, mel_batch.size(-1))
        ssl_loss = recon_loss_fn(recon_pred_masked, recon_tgt_masked) if recon_pred_masked.numel() > 0 else 0.0
        # Combined loss
        loss = asr_loss + ssl_loss_weight * ssl_loss
        optimizer.zero_grad()
        loss.backward()
        # (Optionally clip gradients for stability: e.g., nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        total_asr_loss += asr_loss.item()
        total_ssl_loss += (ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else 0.0)
    print(f"Epoch {epoch}: ASR Loss = {total_asr_loss/len(train_loader):.4f}, SSL Loss = {total_ssl_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "asr_model_lact.pth")

# Usage:
# 1. Set the correct path to LibriSpeech data and run this script to train.
# 2. After training, use asr_adapt.py to perform test-time adaptation and inference.
