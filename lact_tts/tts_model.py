import torch
import torch.nn as nn

class TTSModel(nn.Module):
    """
    TTS model with:
    - Text encoder (shared during TTS mode).
    - Audio encoder (used for unsupervised reconstruction).
    - Attention-based decoder (shared for generation, adapts at test-time).
    """
    def __init__(self, vocab_size, n_mels=80, txt_enc_dim=128, aud_enc_dim=128, dec_hidden_dim=256, attn_dim=256):
        """
        vocab_size: size of character vocab (including pad).
        n_mels: number of mel channels.
        txt_enc_dim: hidden size for text encoder (per direction).
        aud_enc_dim: hidden size for audio encoder (per direction).
        dec_hidden_dim: hidden size for decoder LSTM.
        attn_dim: size of attention intermediate layer.
        """
        super(TTSModel, self).__init__()
        # Text Encoder: Embedding + BiLSTM
        self.embedding = nn.Embedding(vocab_size, txt_enc_dim, padding_idx=0)
        self.text_encoder = nn.LSTM(input_size=txt_enc_dim, hidden_size=txt_enc_dim, num_layers=1,
                                    batch_first=True, bidirectional=True)
        self.txt_enc_out_dim = txt_enc_dim * 2
        # Audio Encoder: BiLSTM (for mel input in reconstruction task)
        self.audio_encoder = nn.LSTM(input_size=n_mels, hidden_size=aud_enc_dim, num_layers=1,
                                     batch_first=True, bidirectional=True)
        self.aud_enc_out_dim = aud_enc_dim * 2
        # Decoder attention and LSTM
        self.attn_W_enc = nn.Linear(self.txt_enc_out_dim, attn_dim)
        self.attn_W_dec = nn.Linear(dec_hidden_dim, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1)
        # Decoder LSTMCell (takes prenet output + context vector)
        self.decoder_lstm = nn.LSTMCell(input_size=dec_hidden_dim + self.txt_enc_out_dim, hidden_size=dec_hidden_dim)
        # Prenet to process last output frame before feeding to LSTM
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, dec_hidden_dim),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(dec_hidden_dim, dec_hidden_dim),
            nn.ReLU(), nn.Dropout(0.5)
        )
        # Output linear layers: predict next mel frame and stop token
        self.mel_out = nn.Linear(dec_hidden_dim + self.txt_enc_out_dim, n_mels)
        self.stop_out = nn.Linear(dec_hidden_dim + self.txt_enc_out_dim, 1)
    
    def forward_text_encoder(self, text_seq, text_lengths):
        """
        Encode text sequence.
        Returns: text_enc_outputs (batch, T_text, txt_enc_out_dim), text_mask (batch, T_text) for attention.
        """
        embedded = self.embedding(text_seq)  # (batch, T_text, txt_enc_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_outputs, _ = self.text_encoder(packed)
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=True)
        # Mask: 1 for actual tokens, 0 for padding
        max_len = text_seq.size(1)
        mask = (torch.arange(max_len, device=text_seq.device)[None, :] < text_lengths[:, None]).float()
        return enc_outputs, mask
    
    def forward_audio_encoder(self, mel_seq, mel_lengths):
        """
        Encode mel spectrogram sequence.
        Returns: audio_enc_outputs (batch, T_mel, aud_enc_out_dim), mask (batch, T_mel).
        """
        packed = nn.utils.rnn.pack_padded_sequence(mel_seq, mel_lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_outputs, _ = self.audio_encoder(packed)
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=True)
        max_len = mel_seq.size(1)
        mask = (torch.arange(max_len, device=mel_seq.device)[None, :] < mel_lengths[:, None]).float()
        return enc_outputs, mask
    
    def decode(self, enc_outputs, enc_mask, target_mel=None, teacher_forcing=True):
        """
        Decode a mel spectrogram given encoder outputs (supports teacher forcing).
        enc_outputs: (batch, T_enc, enc_out_dim) from either text or audio encoder.
        enc_mask: (batch, T_enc) mask for encoder outputs (0 = padding).
        target_mel: (batch, T_out, n_mels) ground truth mel for teacher forcing.
        teacher_forcing: if True, use ground-truth frames as inputs; if False, use last predicted frame.
        Returns: mel_pred (batch, T_out, n_mels), stop_pred (batch, T_out) logits.
        """
        batch_size = enc_outputs.size(0)
        device = enc_outputs.device
        # Initialize decoder state
        h = torch.zeros(batch_size, self.decoder_lstm.hidden_size, device=device)
        c = torch.zeros(batch_size, self.decoder_lstm.hidden_size, device=device)
        prev_frame = torch.zeros(batch_size, self.mel_out.out_features, device=device)  # <GO> frame (all zeros)
        T_out = target_mel.size(1) if target_mel is not None else 0
        mel_outputs = []
        stop_outputs = []
        # Precompute transformed encoder outputs for attention efficiency
        attn_enc = self.attn_W_enc(enc_outputs)  # (batch, T_enc, attn_dim)
        for t in range(T_out):
            # Attention: compute context vector
            attn_dec = self.attn_W_dec(h).unsqueeze(1)                     # (batch, 1, attn_dim)
            energy = torch.tanh(attn_enc + attn_dec)                       # (batch, T_enc, attn_dim)
            energy = self.attn_v(energy).squeeze(-1)                       # (batch, T_enc)
            if enc_mask is not None:
                energy = energy.masked_fill(enc_mask == 0, -1e9)
            attn_weights = torch.softmax(energy, dim=-1)                   # (batch, T_enc)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, enc_out_dim)
            # Prepare decoder input
            if teacher_forcing and target_mel is not None:
                prev_input = target_mel[:, t-1, :] if t > 0 else torch.zeros_like(prev_frame)
            else:
                prev_input = prev_frame
            prenet_out = self.prenet(prev_input)                          # (batch, dec_hidden_dim)
            # One step of decoder LSTM
            lstm_input = torch.cat([prenet_out, context], dim=1)           # (batch, dec_hidden_dim + enc_out_dim)
            h, c = self.decoder_lstm(lstm_input, (h, c))
            # Output mel frame and stop token
            out_combined = torch.cat([h, context], dim=1)                  # (batch, dec_hidden_dim+enc_out_dim)
            mel_frame = self.mel_out(out_combined)                        # (batch, n_mels)
            stop_logit = self.stop_out(out_combined).squeeze(-1)          # (batch,)
            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_logit)
            # Update prev_frame for next step
            if teacher_forcing and target_mel is not None:
                prev_frame = target_mel[:, t, :]  # use ground truth frame at time t as next input
            else:
                prev_frame = mel_frame           # use last predicted frame as next input
        mel_outputs = torch.stack(mel_outputs, dim=1)   # (batch, T_out, n_mels)
        stop_outputs = torch.stack(stop_outputs, dim=1) # (batch, T_out)
        return mel_outputs, stop_outputs
    
    def infer(self, text_seq):
        """
        Synthesize mel spectrogram from input text (greedy autoregressive inference).
        text_seq: (1, T_text) input text indices.
        Returns: mel_out (1, T_out, n_mels) generated mel spectrogram.
        """
        self.eval()
        with torch.no_grad():
            text_len = torch.tensor([text_seq.size(1)], device=text_seq.device)
            enc_outputs, enc_mask = self.forward_text_encoder(text_seq, text_len)
            # Initialize decoder state and inputs
            h = torch.zeros(1, self.decoder_lstm.hidden_size, device=text_seq.device)
            c = torch.zeros(1, self.decoder_lstm.hidden_size, device=text_seq.device)
            prev_frame = torch.zeros(1, self.mel_out.out_features, device=text_seq.device)
            attn_enc = self.attn_W_enc(enc_outputs)
            mel_frames = []
            for _ in range(1000):  # max generation steps to prevent infinite loop
                attn_dec = self.attn_W_dec(h).unsqueeze(1)          # (1, 1, attn_dim)
                energy = torch.tanh(attn_enc + attn_dec)            # (1, T_text, attn_dim)
                energy = self.attn_v(energy).squeeze(-1)            # (1, T_text)
                energy = energy.masked_fill(enc_mask == 0, -1e9)
                attn_weights = torch.softmax(energy, dim=-1)        # (1, T_text)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (1, enc_out_dim)
                prenet_out = self.prenet(prev_frame)                # (1, dec_hidden_dim)
                lstm_input = torch.cat([prenet_out, context], dim=1)  # (1, dec_hidden_dim+enc_out_dim)
                h, c = self.decoder_lstm(lstm_input, (h, c))
                out_combined = torch.cat([h, context], dim=1)
                mel_frame = self.mel_out(out_combined)              # (1, n_mels)
                stop_logit = self.stop_out(out_combined)            # (1, 1)
                mel_frames.append(mel_frame.squeeze(0))
                prev_frame = mel_frame
                # Check stop token
                if torch.sigmoid(stop_logit) > 0.5:
                    break
            mel_out = torch.stack(mel_frames, dim=0).unsqueeze(0)   # (1, T_out, n_mels)
            return mel_out
