# file: train_asr.py
import torch
from torch.utils.data import DataLoader
from data.datasets import LibriSpeechDataset, librispeech_collate
from models.asr_model import ASRModel

# Hyperparameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 10

# Load training and validation datasets
train_dataset = LibriSpeechDataset(root="data/LibriSpeech", url="train-clean-100", download=True)
val_dataset = LibriSpeechDataset(root="data/LibriSpeech", url="dev-clean", download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=librispeech_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=librispeech_collate)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASRModel(input_dim=80, d_model=256, num_layers=6, num_heads=8, window_size=8, chunk_size=512, vocab_size=33).to(device)
ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)  # assuming blank index 0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (padded_mel, mel_lengths, targets_concat, target_lengths) in enumerate(train_loader):
        padded_mel = padded_mel.to(device)
        mel_lengths = mel_lengths.to(device)
        targets_concat = targets_concat.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()
        # Forward pass
        log_probs = model(padded_mel, mel_lengths)  # log_probs shape (T, B, vocab)
        # CTC loss computation
        # Note: input_length for each sequence = output time steps from model = ceil(input_frames) possibly same as mel_length here
        # We use mel_lengths as input_lengths (assuming no subsampling in model, output length == input length)
        input_lengths = mel_lengths  # (B,)
        loss = ctc_loss_fn(log_probs, targets_concat, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Training CTC Loss: {avg_loss:.3f}")
    # Validation WER
    model.eval()
    from utils.metrics import word_error_rate
    total_wer = 0.0
    count = 0
    with torch.no_grad():
        for padded_mel, mel_lengths, targets_concat, target_lengths in val_loader:
            padded_mel = padded_mel.to(device)
            mel_lengths = mel_lengths.to(device)
            # Forward
            log_probs = model(padded_mel, mel_lengths)  # (T, B, vocab) log-probs
            # Greedy decode
            # Take argmax over vocab dimension for each time step
            pred_indices = log_probs.argmax(dim=-1).transpose(0, 1)  # shape (B, T)
            # Convert each prediction to string and each target to string for WER
            for i in range(pred_indices.size(0)):
                # Collapse repeats and remove blanks (index 0)
                pred_seq = []
                prev = None
                for t in pred_indices[i, :mel_lengths[i]]:
                    idx = int(t.item())
                    if idx != 0 and idx != prev:
                        pred_seq.append(idx)
                    prev = idx
                # Convert indices to text
                from utils.text import sequence_to_text
                pred_text = sequence_to_text(pred_seq)
                # Retrieve reference text from targets_concat using target_lengths
                # Need to slice the portion corresponding to this sample
                # We have targets_concat concatenated for the batch, we use an offset
                # Compute offsets from target_lengths
                # Simplest: reconstruct target string from original dataset for WER (not efficient but straightforward)
                # (We can also store text in dataset object for val for simplicity)
                # Here we'll just recompute from concatenated: need to track offset as we iterate
            # (Simpler approach: modify val_loader to return actual text strings for WER. We'll assume we have access to transcripts.)
