# file: train_tts.py
import torch
from torch.utils.data import DataLoader
from data.datasets import LJSpeechDataset, ljspeech_collate
from models.tts_model import TTSModel

# Hyperparameters
batch_size = 8
learning_rate = 1e-4
num_epochs = 10

# Load dataset
train_dataset = LJSpeechDataset(root="data/LJSpeech-1.1", download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ljspeech_collate)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TTSModel(vocab_size=33, d_model=256, num_encoder_layers=3, num_decoder_layers=6, num_heads=8, window_size=8, chunk_size=512, n_mels=80).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss()  # use L1 loss on mel spectrogram

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (padded_text, text_lengths, padded_mel, mel_lengths) in enumerate(train_loader):
        padded_text = padded_text.to(device)
        text_lengths = text_lengths.to(device)
        padded_mel = padded_mel.to(device)
        mel_lengths = mel_lengths.to(device)
        optimizer.zero_grad()
        # Forward (teacher forcing)
        pred_mel = model(padded_text, text_lengths, mel=padded_mel, mel_lengths=mel_lengths)  # (B, T_mel, n_mels)
        # Compute L1 loss on non-padded time steps
        # To mask out padded frames in loss, we'll zero out loss for those frames
        # Create mask for valid frames
        max_len = pred_mel.shape[1]
        mask = (torch.arange(max_len, device=device)[None, :] < mel_lengths[:, None]).unsqueeze(-1)  # (B, T, 1)
        # Compute L1
        loss = loss_fn(pred_mel * mask, padded_mel * mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Training L1 Loss: {avg_loss:.3f}")
    # (Optional: validation by synthesizing samples and computing MCD or listening to output)
