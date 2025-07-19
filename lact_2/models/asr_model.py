# file: models/asr_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lact_layers import LaCTEncoderBlock

class ASRModel(nn.Module):
    """
    ASR model with LaCT-based encoder and CTC output.
    """
    def __init__(self, input_dim=80, d_model=256, num_layers=6, num_heads=8, window_size=8, 
                 chunk_size=512, vocab_size=33):
        super(ASRModel, self).__init__()
        # Frontend: project input features (e.g., log-Mel spectrogram) to model dimension
        self.in_proj = nn.Linear(input_dim, d_model)
        # Encoder: stack of LaCT encoder blocks
        self.layers = nn.ModuleList([
            LaCTEncoderBlock(d_model, num_heads, window_size=window_size, chunk_size=chunk_size, causal=False)
            for _ in range(num_layers)
        ])
        # Final layer norm (pre-CTC) for stability
        self.final_norm = nn.LayerNorm(d_model)
        # CTC classification head (vocab_size includes blank token)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, features, lengths):
        """
        features: padded input tensor of shape (batch, T, input_dim)
        lengths: 1D tensor of sequence lengths (before padding) for each batch item
        Returns: log-probabilities for CTC of shape (T, batch, vocab_size)
        """
        # Project input features to model dimension
        x = self.in_proj(features)  # shape: (batch, T, d_model)
        # Apply each LaCT encoder block with padding mask
        # Create padding mask for attention (True for padding positions)
        max_len = x.size(1)
        pad_mask = (torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None])  # shape (B, T)
        for layer in self.layers:
            x = layer(x, pad_mask)  # each layer returns (batch, T, d_model)
        # Final layer normalization
        x = self.final_norm(x)
        # Output linear layer for CTC logits
        logits = self.fc_out(x)  # shape: (batch, T, vocab_size)
        # For CTC loss, PyTorch expects shape (T, B, C), so transpose
        logits = logits.transpose(0, 1)  # (T, batch, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
