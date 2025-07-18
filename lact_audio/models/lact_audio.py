# (Moved from lact_ar_video/minVid/models/lact_audio.py)
# -*- coding: utf-8 -*-
"""
LaCTAudio: Large Chunk Test-Time Training for Audio Transformers
Adapted from lact_llm/lact_model for audio token sequences.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.configuration_utils import PretrainedConfig
from lact_llm.lact_model.ttt_operation import block_causal_lact_swiglu

# --- Config ---
class LaCTAudioConfig(PretrainedConfig):
    model_type = 'lact_audio'
    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 12,
        num_attn_heads: int = 16,
        num_lact_heads: int = 4,
        inter_multi: int = 1,
        lact_chunk_size: int = 2048,
        use_muon: bool = False,
        window_size: int = 512,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.window_size = window_size
        super().__init__(**kwargs)

# --- LaCT Block for Audio ---
class LaCTAudioBlock(nn.Module):
    def __init__(self, config: LaCTAudioConfig):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attn_heads,
            batch_first=True
        )
        # Fast weight MLP (SwiGLU style)
        d_in = config.hidden_size // config.num_lact_heads
        d_h = int(d_in * config.inter_multi)
        self.w0 = nn.Parameter(torch.randn(config.num_lact_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(config.num_lact_heads, d_in, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(torch.randn(config.num_lact_heads, d_h, d_in) / math.sqrt(d_in))
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Window attention (local)
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        # Chunking for LaCT
        B, S, D = x.shape
        chunk_size = self.config.lact_chunk_size
        out = torch.zeros_like(x)
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            chunk = x[:, start:end, :]
            # Fast weight update/apply (block_causal_lact_swiglu)
            # For demo: use dummy lr and k/v/q split
            q = k = v = chunk.transpose(1, 2)  # [B, D, L]
            lr0 = lr1 = lr2 = torch.ones_like(q)
            chunk_out = block_causal_lact_swiglu(
                self.w0, self.w1, self.w2, q, k, v, lr0, lr1, lr2, chunk_size=chunk.shape[1], use_muon=self.config.use_muon
            )
            out[:, start:end, :] = chunk_out.transpose(1, 2)
        return out + attn_out

# --- LaCT Audio Model ---
class LaCTAudioModel(nn.Module):
    def __init__(self, config: LaCTAudioConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([LaCTAudioBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.final_norm(x) 