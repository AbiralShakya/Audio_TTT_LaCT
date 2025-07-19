# file: lact_layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    Local windowed self-attention layer (with multi-head attention).
    Supports causal (uni-directional) or non-causal window.
    """
    def __init__(self, d_model, n_head, window_size, causal=False):
        super(WindowAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.window_size = window_size
        self.causal = causal
        # Use PyTorch built-in MultiheadAttention 
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
    
    def forward(self, x, key_padding_mask=None):
        """
        x: Tensor of shape (batch, T, d_model)
        key_padding_mask: Bool mask of shape (batch, T) with True at padded positions (to exclude them).
        Returns: Tensor of shape (batch, T, d_model) after windowed self-attention + residual.
        """
        B, T, D = x.shape
        # Build attention mask for local window:
        # Mask shape for MultiheadAttention (batch_first=True) should be (T, T) or (B*n_head, T, T) if using different per head.
        # Here we use same mask for all heads and broadcast, so shape (T, T).
        device = x.device
        # Compute allowed positions mask (False = allowed, True = masked out)
        idx = torch.arange(T, device=device)
        if self.causal:
            # For causal: allow attending from position i to j if (i - j <= window_size) and j <= i
            # Compute difference matrix i - j
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # shape (T, T), diff[i,j] = i-j
            allowed = (diff >= 0) & (diff <= self.window_size)
        else:
            # Non-causal: allow if |i - j| <= window_size
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)
            allowed = (diff.abs() <= self.window_size)
        # Create final attn_mask: True for disallowed positions
        attn_mask = ~allowed  # invert allowed to get mask
        # If key_padding_mask is provided, we integrate it: we don't want to attend to padded positions.
        # MultiheadAttention can directly use key_padding_mask for keys/values, but not for query.
        # We'll pass key_padding_mask separately to attn, which will mask out those keys in attention.
        # Prepare query, key, value as the same (self-attention)
        # Permute x to (T, B, D) for MultiheadAttention if batch_first is False, but we set batch_first=True in attn, so we can pass as is.
        # Apply attention
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask.to(x.device), key_padding_mask=key_padding_mask)
        # Return with residual connection
        return x + attn_output  # (B, T, D)

class FastWeightTTT(nn.Module):
    """
    Large-Chunk Test-Time Training (TTT) layer implementing fast-weight memory update and apply.
    Uses a SwiGLU-MLP fast weight network f_W (·) with parameters W1, W2, W3.
    Performs chunk-wise 'update' of fast weights using keys & values, then 'apply' to queries.
    """
    def __init__(self, d_model, hidden_factor=2, fast_lr=1.0, momentum=0.9, chunk_size=512):
        super(FastWeightTTT, self).__init__()
        self.d_model = d_model
        self.hidden = int(hidden_factor * d_model)  # dimension of fast weight network hidden layer
        self.chunk_size = chunk_size
        # Fast weight network parameters (slow weights, will be adapted at test time)
        # W1, W3: (hidden, d_model), W2: (d_model, hidden)
        self.W1 = nn.Parameter(torch.randn(self.hidden, d_model) * (1.0 / math.sqrt(d_model)))
        self.W2 = nn.Parameter(torch.randn(d_model, self.hidden) * (1.0 / math.sqrt(self.hidden)))
        self.W3 = nn.Parameter(torch.randn(self.hidden, d_model) * (1.0 / math.sqrt(d_model)))
        # Linear projections for Q, K, V (project input to query/key/value space)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        # Fast weight update hyperparameters
        self.fast_lr = fast_lr
        self.momentum = momentum
    
    def forward(self, x, pad_mask=None):
        """
        x: Tensor of shape (B, T, d_model)
        pad_mask: Bool Tensor (B, T) with True indicating padding (to exclude from fast weight update).
        Returns: Tensor (B, T, d_model) after fast-weight memory layer (with residual to input added externally).
        """
        B, T, D = x.shape
        # Project inputs to Q, K, V
        Q = self.Wq(x)  # (B, T, D)
        K = self.Wk(x)  # (B, T, D)
        V = self.Wv(x)  # (B, T, D)
        # Allocate output tensor
        out = torch.zeros_like(Q)
        # Process each sequence in the batch independently (fast weights are not shared across sequences)
        for b in range(B):
            # Get actual sequence length (if pad_mask given)
            if pad_mask is not None:
                # padded positions have True, find how many are False from start
                if pad_mask[b].any():
                    L = pad_mask[b].nonzero(as_tuple=True)[0][0].item()  # first padded index as length
                else:
                    L = T
            else:
                L = T
            # Slice valid tokens
            Q_seq = Q[b, :L, :]  # (L, D)
            K_seq = K[b, :L, :]
            V_seq = V[b, :L, :]
            if L == 0:
                continue  # skip if no valid tokens
            # Initialize fast weights for this sequence (copy initial slow weights)
            W1_fw = self.W1  # (hidden, D)
            W2_fw = self.W2  # (D, hidden)
            W3_fw = self.W3  # (hidden, D)
            # Initialize momentum buffers for fast weights
            v1 = torch.zeros_like(W1_fw)
            v2 = torch.zeros_like(W2_fw)
            v3 = torch.zeros_like(W3_fw)
            # Iterate through large chunks
            chunk_size = self.chunk_size
            for start in range(0, L, chunk_size):
                end = min(L, start + chunk_size)
                K_chunk = K_seq[start:end, :]  # (chunk_len, D)
                V_chunk = V_seq[start:end, :]  # (chunk_len, D)
                chunk_len = K_chunk.size(0)
                if chunk_len == 0:
                    continue
                # 'Update' fast weights using all key/value pairs in this chunk
                # Compute fast weight network output for all keys in chunk (fW_old(K))
                # We'll compute gradient of loss sum = -sum_i fW(K_i)^T V_i
                # Forward through fast weight net for keys
                A = K_chunk @ W1_fw.T  # (chunk_len, hidden)
                B = K_chunk @ W3_fw.T  # (chunk_len, hidden)
                S = torch.sigmoid(A) * A  # SiLU: using x*sigmoid(x) for numerical stability of gradient
                U = S * B  # elementwise [chunk_len, hidden]
                O = U @ W2_fw.T  # (chunk_len, D) = fW_old(K_chunk)
                # Compute dot-product loss: L = - sum_i (O_i · V_i)
                # We compute gradients w.rt W1_fw, W2_fw, W3_fw
                # grad_O = -V_chunk
                grad_O = -V_chunk  # (chunk_len, D)
                # grad_W2 = grad_O^T @ U
                grad_W2 = grad_O.T @ U  # (D, hidden)
                # grad_U = grad_O @ W2_fw
                grad_U = grad_O @ W2_fw  # (chunk_len, hidden)
                # grad_S = grad_U * B
                grad_S = grad_U * B  # (chunk_len, hidden)
                # grad_B = grad_U * S
                grad_B = grad_U * S  # (chunk_len, hidden)
                # grad_A = grad_S * derivative of S = grad_S * (sigmoid(A) * (1 + A * (1 - sigmoid(A))))
                sigA = torch.sigmoid(A)
                grad_A = grad_S * (sigA * (1 + A * (1 - sigA)))  # (chunk_len, hidden)
                # grad_W1 = grad_A^T @ K_chunk
                grad_W1 = grad_A.T @ K_chunk  # (hidden, D)
                # grad_W3 = grad_B^T @ K_chunk
                grad_W3 = grad_B.T @ K_chunk  # (hidden, D)
                # Update fast weights with momentum (MuOn optimizer style)
                v1 = self.momentum * v1 + grad_W1
                v2 = self.momentum * v2 + grad_W2
                v3 = self.momentum * v3 + grad_W3
                W1_fw = W1_fw - self.fast_lr * v1  # (hidden, D)
                W2_fw = W2_fw - self.fast_lr * v2  # (D, hidden)
                W3_fw = W3_fw - self.fast_lr * v3  # (hidden, D)
                # 'Apply' updated fast weight to all queries in this chunk
                A_q = Q_seq[start:end, :] @ W1_fw.T  # (chunk_len, hidden)
                B_q = Q_seq[start:end, :] @ W3_fw.T  # (chunk_len, hidden)
                S_q = F.silu(A_q)  # (chunk_len, hidden) - SiLU activation
                U_q = S_q * B_q  # (chunk_len, hidden)
                O_q = U_q @ W2_fw.T  # (chunk_len, D) = fW_new(Q_chunk)
                # Write output for this chunk
                out[b, start:end, :] = O_q
            # end for chunks
        # end for batch
        return out  # (B, T, D) output (to be added to residual outside)

# Continue lact_layers.py for complete block definitions
class LaCTEncoderBlock(nn.Module):
    """
    One block of the LaCT encoder, with:
    - Local window self-attention (not causal, captures local context within chunk)
    - Large-chunk TTT layer (fast-weight memory update/apply)
    - Feed-forward network
    - Residual connections and layer normalization
    """
    def __init__(self, d_model, n_head, window_size, chunk_size, causal=False):
        super(LaCTEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Local windowed self-attention layer
        self.window_attn = WindowAttention(d_model, n_head, window_size, causal=causal)
        # Fast-weight memory layer (TTT)
        self.fast_weight = FastWeightTTT(d_model, hidden_factor=2, fast_lr=1.0, momentum=0.9, chunk_size=chunk_size)
        # Feed-forward layer (Transformer FFN with expansion and activation)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x, pad_mask=None):
        # x: (B, T, d_model)
        # Windowed self-attention with residual
        y = self.window_attn(self.norm1(x), key_padding_mask=pad_mask)  # (B, T, d_model)
        x = x + y  # residual addition (note: WindowAttention already adds residual internally as well, but we add again to ensure proper scaling)
        # Fast-weight memory layer with residual
        y2 = self.fast_weight(self.norm2(x), pad_mask=pad_mask)  # (B, T, d_model)
        x = x + y2  # residual add
        # Feed-forward network with residual
        y3 = self.ffn(self.norm3(x))  # (B, T, d_model)
        out = x + y3
        return out  # (B, T, d_model)

class LaCTDecoderBlock(nn.Module):
    """
    One block of the LaCT decoder, with:
    - Local windowed self-attention (causal)
    - Cross-attention to text encoder output
    - Large-chunk TTT layer (fast-weight memory for decoder)
    - Feed-forward network
    - Residual connections and layer normalization
    """
    def __init__(self, d_model, n_head, window_size, chunk_size):
        super(LaCTDecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        # Local causal self-attention (attending to previous decoder outputs within window)
        self.self_attn = WindowAttention(d_model, n_head, window_size, causal=True)
        # Cross-attention (decoder query to encoder key/values)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        # Fast-weight memory layer (TTT) for decoder
        self.fast_weight = FastWeightTTT(d_model, hidden_factor=2, fast_lr=1.0, momentum=0.9, chunk_size=chunk_size)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x, encoder_out, tgt_pad_mask=None, src_pad_mask=None):
        """
        x: Tensor (B, T_dec, d_model) decoder input
        encoder_out: Tensor (B, T_enc, d_model) text encoder outputs
        tgt_pad_mask: Bool mask (B, T_dec) for decoder padded positions
        src_pad_mask: Bool mask (B, T_enc) for text padded positions
        """
        # Self-attention (causal, local)
        y = self.self_attn(self.norm1(x), key_padding_mask=tgt_pad_mask)  # (B, T_dec, d_model)
        x = x + y  # residual
        # Cross-attention (decoder queries attend to all encoder outputs)
        # Prepare query, key, value for cross-attention
        q = self.norm2(x)  # (B, T_dec, d_model)
        k = encoder_out  # (B, T_enc, d_model)
        v = encoder_out  # (B, T_enc, d_model)
        # Use MultiheadAttention (batch_first=True)
        attn_out, _ = self.cross_attn(q, k, v, key_padding_mask=src_pad_mask)
        x = x + attn_out  # residual
        # Fast-weight memory layer
        y2 = self.fast_weight(self.norm3(x), pad_mask=tgt_pad_mask)  # (B, T_dec, d_model)
        x = x + y2  # residual
        # Feed-forward
        y3 = self.ffn(self.norm4(x))  # (B, T_dec, d_model)
        out = x + y3
        return out  # (B, T_dec, d_model)
