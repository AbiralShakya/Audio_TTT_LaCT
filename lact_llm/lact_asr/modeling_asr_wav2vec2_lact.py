import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from typing import Optional, Tuple, Dict, Any
import math

class LaCTLayer(nn.Module):
    """
    LaCT (Large Chunk Test-Time Training) layer for ASR
    Implements chunk-based fast weight updates as described in "Test Time Training Done Right"
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_lact_heads: int = 4,
        inter_multi: int = 2,
        chunk_size: int = 512,
        lr_scale: float = 0.01,
        use_momentum: bool = True,
        momentum: float = 0.9
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_lact_heads = num_lact_heads
        self.chunk_size = chunk_size
        self.lr_scale = lr_scale
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # Fast weight matrices (per head)
        d_in = hidden_size // num_lact_heads
        d_h = int(d_in * inter_multi)
        
        # Initialize fast weights with small values
        self.register_buffer('w0', torch.randn(num_lact_heads, d_h, d_in) * 0.01)
        self.register_buffer('w1', torch.randn(num_lact_heads, d_in, d_h) * 0.01)
        self.register_buffer('w2', torch.randn(num_lact_heads, d_h, d_in) * 0.01)
        
        # Momentum buffers for fast weights
        if use_momentum:
            self.register_buffer('w0_momentum', torch.zeros_like(self.w0))
            self.register_buffer('w1_momentum', torch.zeros_like(self.w1))
            self.register_buffer('w2_momentum', torch.zeros_like(self.w2))
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Learning rate parameters (learnable)
        self.lr0 = nn.Parameter(torch.tensor(lr_scale))
        self.lr1 = nn.Parameter(torch.tensor(lr_scale))
        self.lr2 = nn.Parameter(torch.tensor(lr_scale))
    
    def update_fast_weights(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Update fast weights using chunk-based test-time training
        q, k, v: [num_heads, d_in, chunk_size]
        """
        # Compute gradients for fast weight update
        # This is a simplified version - in practice, you'd compute actual gradients
        # based on the task-specific loss (e.g., next token prediction for ASR)
        
        # For ASR, we can use a simple reconstruction loss or next-frame prediction
        # Here we use a simplified approach based on the paper's methodology
        
        # Compute attention-like update signal
        attn_weights = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(q.size(-2)), dim=-1)
        update_signal = torch.matmul(attn_weights, v.transpose(-2, -1))
        
        # Update fast weights using gradient-like signals
        dw0 = torch.matmul(update_signal, q.transpose(-2, -1)) * self.lr0.item()
        dw1 = torch.matmul(q, update_signal.transpose(-2, -1)) * self.lr1.item()
        dw2 = torch.matmul(update_signal, k.transpose(-2, -1)) * self.lr2.item()
        
        if self.use_momentum:
            self.w0_momentum = self.momentum * self.w0_momentum + dw0
            self.w1_momentum = self.momentum * self.w1_momentum + dw1
            self.w2_momentum = self.momentum * self.w2_momentum + dw2
            
            self.w0 += self.w0_momentum
            self.w1 += self.w1_momentum
            self.w2 += self.w2_momentum
        else:
            self.w0 += dw0
            self.w1 += dw1
            self.w2 += dw2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunk-based test-time training
        x: [batch_size, sequence_length, hidden_size]
        """
        B, S, D = x.shape
        x = self.norm(x)
        
        # Split into chunks for LaCT processing
        output = torch.zeros_like(x)
        
        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            chunk = x[:, start:end, :]  # [B, chunk_size, D]
            
            # Reshape for multi-head processing
            chunk_reshaped = chunk.view(B, -1, self.num_lact_heads, D // self.num_lact_heads)
            chunk_reshaped = chunk_reshaped.transpose(1, 2)  # [B, num_heads, chunk_size, d_in]
            
            # Split into q, k, v (simplified - in practice you'd have separate projections)
            q = k = v = chunk_reshaped.transpose(-2, -1)  # [B, num_heads, d_in, chunk_size]
            
            # Update fast weights for this chunk
            self.update_fast_weights(q[0], k[0], v[0])  # Use first batch for simplicity
            
            # Apply fast weights (SwiGLU-style)
            # w0 -> w1 -> w2 pipeline
            h = torch.matmul(self.w0, q)  # [B, num_heads, d_h, chunk_size]
            h = F.silu(h)  # SwiGLU activation
            h = torch.matmul(self.w1, h)  # [B, num_heads, d_in, chunk_size]
            h = torch.matmul(self.w2, h)  # [B, num_heads, d_in, chunk_size]
            
            # Combine heads and reshape back
            h = h.transpose(-2, -1)  # [B, num_heads, chunk_size, d_in]
            h = h.transpose(1, 2)  # [B, chunk_size, num_heads, d_in]
            h = h.reshape(B, -1, D)  # [B, chunk_size, D]
            
            output[:, start:end, :] = h
        
        return output

class Wav2Vec2WithLaCT(nn.Module):
    """
    Wav2Vec2 model with LaCT (chunk-based test-time training) layers
    """
    def __init__(
        self,
        wav2vec2_model: Wav2Vec2ForCTC,
        num_lact_layers: int = 2,
        lact_hidden_size: int = 768,
        num_lact_heads: int = 4,
        lact_chunk_size: int = 512,
        lact_lr_scale: float = 0.01,
        use_momentum: bool = True
    ):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        
        # Add LaCT layers on top of Wav2Vec2
        self.lact_layers = nn.ModuleList([
            LaCTLayer(
                hidden_size=lact_hidden_size,
                num_lact_heads=num_lact_heads,
                chunk_size=lact_chunk_size,
                lr_scale=lact_lr_scale,
                use_momentum=use_momentum
            ) for _ in range(num_lact_layers)
        ])
        
        # Projection layer to match Wav2Vec2 output to LaCT input
        wav2vec2_hidden_size = wav2vec2_model.config.hidden_size
        self.input_projection = nn.Linear(wav2vec2_hidden_size, lact_hidden_size)
        
        # Final projection back to Wav2Vec2 vocabulary size
        self.output_projection = nn.Linear(lact_hidden_size, wav2vec2_model.config.vocab_size)
        
        # Layer norm for stability
        self.final_norm = nn.LayerNorm(lact_hidden_size)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with chunk-based test-time training
        """
        # Get Wav2Vec2 features
        wav2vec2_outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Use the last hidden state
        hidden_states = wav2vec2_outputs.hidden_states[-1]  # [B, T, hidden_size]
        
        # Project to LaCT hidden size
        x = self.input_projection(hidden_states)
        
        # Apply LaCT layers with chunk-based test-time training
        for lact_layer in self.lact_layers:
            x = lact_layer(x)
        
        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return {
            'logits': logits,
            'hidden_states': wav2vec2_outputs.hidden_states,
            'attentions': wav2vec2_outputs.attentions
        }
    
    def reset_fast_weights(self):
        """Reset fast weights for new sequence"""
        for lact_layer in self.lact_layers:
            lact_layer.w0.zero_()
            lact_layer.w1.zero_()
            lact_layer.w2.zero_()
            if lact_layer.use_momentum:
                lact_layer.w0_momentum.zero_()
                lact_layer.w1_momentum.zero_()
                lact_layer.w2_momentum.zero_()

class BaselineWav2Vec2(nn.Module):
    """
    Baseline Wav2Vec2 model without TTT (for comparison)
    """
    def __init__(self, wav2vec2_model: Wav2Vec2ForCTC):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
    
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        return self.wav2vec2(input_values=input_values, attention_mask=attention_mask, **kwargs) 