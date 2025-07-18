import torch
import torch.nn as nn

class ASRModel(nn.Module):
    """
    ASR model with a shared acoustic encoder and two heads:
    - ASR head for CTC (maps features to character logits).
    - SSL head for self-supervised masked reconstruction of mel spectrogram.
    """
    def __init__(self, n_mels=80, enc_hidden_dim=256, enc_layers=2, vocab_size=29):
        """
        n_mels: number of mel frequency bins (input feature size).
        enc_hidden_dim: hidden size of LSTM encoder.
        enc_layers: number of LSTM layers.
        vocab_size: number of output tokens including CTC blank.
        """
        super(ASRModel, self).__init__()
        # Shared feature extractor (Bidirectional LSTM encoder)
        self.encoder = nn.LSTM(input_size=n_mels, hidden_size=enc_hidden_dim, num_layers=enc_layers, 
                                batch_first=True, bidirectional=True)
        enc_out_dim = enc_hidden_dim * 2  # bidirectional double dimension
        # Task-specific heads
        self.asr_head = nn.Linear(enc_out_dim, vocab_size)   # CTC character logits
        self.ssl_head = nn.Linear(enc_out_dim, n_mels)       # reconstruct input features
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, mel_inputs, mel_mask=None):
        """
        Forward pass for training.
        mel_inputs: (batch, time, n_mels) input features.
        mel_mask: (batch, time) binary mask for frames to reconstruct (1 = masked).
                  Masked frames in mel_inputs will be zeroed out.
        Returns:
          asr_logits: (batch, time, vocab_size) raw logits for CTC.
          recon_output: (batch, time, n_mels) predicted mel spectrogram (for all frames, compare masked ones).
        """
        x = mel_inputs
        if mel_mask is not None:
            # Zero out masked frames to simulate missing data
            mask = mel_mask.unsqueeze(-1).to(x.dtype)     # (batch, time, 1)
            x = x * (1 - mask)                            # set masked positions to 0
        enc_outputs, _ = self.encoder(x)                  # (batch, time, enc_out_dim)
        enc_outputs = self.dropout(enc_outputs)
        asr_logits = self.asr_head(enc_outputs)
        recon_output = self.ssl_head(enc_outputs)
        return asr_logits, recon_output
    
    def infer(self, mel_inputs):
        """
        Inference (no adaptation): returns log-probabilities for CTC decoding.
        """
        self.eval()
        with torch.no_grad():
            enc_outputs, _ = self.encoder(mel_inputs)              # (1, time, enc_out_dim)
            logits = self.asr_head(enc_outputs)                   # (1, time, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)          # log-probs for CTC
        return log_probs

