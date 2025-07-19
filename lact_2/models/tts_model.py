# file: models/tts_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lact_layers import LaCTDecoderBlock
from utils.text import char_to_idx

class TTSModel(nn.Module):
    """
    TTS model with optional text encoder and LaCT-based decoder.
    Generates mel-spectrograms from input text.
    """
    def __init__(self, vocab_size=33, d_model=256, num_encoder_layers=3, num_decoder_layers=6, 
                 num_heads=8, window_size=8, chunk_size=512, n_mels=80):
        super(TTSModel, self).__init__()
        self.d_model = d_model
        self.n_mels = n_mels
        # Text embedding and encoder (optional cross-attention encoder)
        self.text_embed = nn.Embedding(vocab_size, d_model)
        # Use a simple Transformer encoder for text (global attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=4*d_model, dropout=0.1)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # Decoder prenet: process previous mel frames before decoder (2-layer FC with dropout)
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Decoder: stack of LaCT decoder blocks
        self.dec_layers = nn.ModuleList([
            LaCTDecoderBlock(d_model, num_heads, window_size=window_size, chunk_size=chunk_size)
            for _ in range(num_decoder_layers)
        ])
        # Final layer norm on decoder output
        self.final_norm = nn.LayerNorm(d_model)
        # Output projection to mel-spectrogram dimension
        self.mel_out = nn.Linear(d_model, n_mels)
    
    def forward(self, text, text_lengths, mel=None, mel_lengths=None):
        """
        text: LongTensor of shape (batch, T_text) of text token indices.
        text_lengths: lengths of each text sequence in batch.
        mel: (Optional) FloatTensor (batch, T_mel, n_mels) of target mel for teacher-forcing.
        mel_lengths: lengths of each mel sequence.
        If mel is provided, uses teacher-forcing and returns predicted mel frames (for training).
        If mel is None, runs auto-regressive inference (greedy) and returns generated mel (for inference).
        """
        # Encode text
        # Transpose to (T_text, batch) for TransformerEncoder
        text_embed = self.text_embed(text).transpose(0, 1)  # shape: (T_text, B, d_model)
        # Generate key padding mask for text (True where pad)
        text_pad_mask = (torch.arange(text.size(1), device=text.device)[None, :] >= text_lengths[:, None])
        text_enc = self.text_encoder(text_embed, src_key_padding_mask=text_pad_mask)  # (T_text, B, d_model)
        text_enc = text_enc.transpose(0, 1)  # (B, T_text, d_model)
        # Prepare target sequence input
        if mel is not None:
            # Teacher-forcing mode: prepend <sos> frame (zeros) and remove last frame for inputs
            B, T_mel, _ = mel.shape
            # <sos> as zero vector of shape (B, 1, n_mels)
            sos_frame = torch.zeros((B, 1, self.n_mels), device=mel.device, dtype=mel.dtype)
            mel_in = torch.cat([sos_frame, mel[:, :-1, :]], dim=1)  # (B, T_mel, n_mels) after adding sos
            target_len = mel_in.size(1)
        else:
            # Inference mode: generate step by step
            mel_outputs = []
            B = text.size(0)
            # Initialize with <sos> frame
            prev_frame = torch.zeros((B, 1, self.n_mels), device=text.device)
            # We will generate until a stopping condition; here use a max length heuristic (e.g.,  max_frames = text_len * 50)
            max_frames = text_lengths.max().item() * 50
            for t in range(max_frames):
                # Prenet on last output frame
                prev_emb = self.prenet(prev_frame)  # (B, 1, d_model)
                # Decoder forward one step (use available context)
                # We call decoder on the entire sequence generated so far for simplicity
                if t == 0:
                    dec_input = prev_emb  # shape (B, 1, d_model) for first step (just sos)
                else:
                    dec_input = torch.cat([dec_input, prev_emb], dim=1)  # append new frame embedding
                # Create target pad mask (none of the generated frames are pad, so all False)
                tgt_pad_mask = None
                # Pass through decoder layers
                out = dec_input  # (B, t+1, d_model)
                for layer in self.dec_layers:
                    out = layer(out, text_enc, tgt_pad_mask, text_pad_mask)
                out = self.final_norm(out)
                pred_mel = self.mel_out(out)  # (B, t+1, n_mels)
                # Take the last frame of pred_mel as the newly generated frame
                new_frame = pred_mel[:, -1:, :]
                mel_outputs.append(new_frame)
                # Prepare for next iteration
                prev_frame = new_frame
            # Concatenate generated frames
            mel_outputs = torch.cat(mel_outputs, dim=1)  # (B, T_gen, n_mels)
            return mel_outputs  # return generated mel spectrogram
        # If teacher forcing:
        # Prenet on all input frames at once
        dec_in_emb = self.prenet(mel_in)  # shape: (B, target_len, d_model)
        # Create target padding mask for decoder (True where pad in mel sequence)
        tgt_pad_mask = (torch.arange(target_len, device=mel.device)[None, :] >= mel_lengths[:, None]) if mel_lengths is not None else None
        # Decoder forward through all layers
        out = dec_in_emb  # (B, T_mel, d_model)
        for layer in self.dec_layers:
            out = layer(out, text_enc, tgt_pad_mask, text_pad_mask)
        out = self.final_norm(out)
        pred_mel = self.mel_out(out)  # (B, T_mel, n_mels)
        return pred_mel  # return predicted mel spectrogram (same shape as target mel)
