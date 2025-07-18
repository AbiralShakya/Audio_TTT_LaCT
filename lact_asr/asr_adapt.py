import torch
from asr_preprocess import LibriSpeechDataset, collate_fn, idx_to_char, char_to_idx
from asr_model import ASRModel

# Load trained ASR model
model = ASRModel(vocab_size=len(char_to_idx)+1)
model.load_state_dict(torch.load("asr_model_lact.pth", map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare model for adaptation (fast weights update)
model.train()
# Freeze the ASR head (keep classifier stable) – adapt only encoder (shared) and SSL head
for param in model.asr_head.parameters():
    param.requires_grad = False

# Load an unlabeled test sample (using first sample of LibriSpeech dev-clean for demo)
test_set = LibriSpeechDataset(root_dir="path/to/librispeech", url="dev-clean", download=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
mel_batch, mel_lengths, tokens_concat, token_lengths = next(iter(test_loader))
# (In real use, tokens_concat/token_lengths would not be available for unlabeled test data)
mel_batch = mel_batch.to(device)
mel_length = mel_lengths.to(device)

# Optimizer for adaptation (fast online updates) – only update unfrozen (encoder + SSL) parameters
adapt_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(adapt_params, lr=1e-4)  # small LR for stable test-time updates

# Perform adaptation on the test sample (chunk-based)
model.train()
chunk_mask = torch.zeros(1, mel_batch.size(1), device=device)
L = mel_length.item()
# Mask a large chunk of frames (e.g., 15%) to simulate test-time training
num_mask = max(1, int(0.15 * L))
mask_idx = torch.randperm(L, device=device)[:num_mask]
chunk_mask[0, mask_idx] = 1.0
# Run a few gradient update steps on this chunk
adapt_iters = 5
for i in range(adapt_iters):
    asr_logits, recon_output = model(mel_batch, mel_mask=chunk_mask)
    # Self-supervised loss: L1 on masked frames reconstruction
    mask_bool = chunk_mask.bool().unsqueeze(-1)           # (1, time, 1)
    recon_pred_masked = recon_output[mask_bool].reshape(-1, recon_output.size(-1))
    recon_tgt_masked = mel_batch[mask_bool].reshape(-1, mel_batch.size(-1))
    ssl_loss = torch.nn.functional.l1_loss(recon_pred_masked, recon_tgt_masked)
    optimizer.zero_grad()
    ssl_loss.backward()
    optimizer.step()

# Inference after adaptation
model.eval()
with torch.no_grad():
    log_probs = model.infer(mel_batch)  # log_probs: (1, T, vocab)
# Greedy CTC decoding
pred_indices = log_probs.argmax(dim=-1).squeeze(0).cpu().numpy()  # sequence of token indices
prev = None
pred_tokens = []
for idx in pred_indices:
    if idx != 0 and idx != prev:  # skip blanks (0) and repeated chars
        pred_tokens.append(idx)
    prev = idx
pred_text = "".join([idx_to_char[i] for i in pred_tokens])
print("Transcription after adaptation:", pred_text)

# Note: In a real streaming scenario, you would repeat the above adaptation for each large chunk of input,
# carrying the updated model to the next chunk. The small learning rate and limited steps ensure minimal 
# degradation when there's no significant domain shift.
