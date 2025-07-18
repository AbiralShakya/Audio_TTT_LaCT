import os
import torchaudio
import torch
from torch.utils.data import Dataset

# Define a mapping from characters to indices for transcripts (CTC blank = 0).
BLANK_TOKEN = 0
CHAR_LIST = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ") + ["'"]  # Letters A-Z, space, apostrophe
char_to_idx = {c: i+1 for i, c in enumerate(CHAR_LIST)}  # map chars to 1...N (reserve 0 for blank)
idx_to_char = {i+1: c for i, c in enumerate(CHAR_LIST)}
idx_to_char[BLANK_TOKEN] = ""  # CTC blank maps to empty string

class LibriSpeechDataset(Dataset):
    """
    Dataset for LibriSpeech audio-transcript pairs. Produces log-mel spectrogram features and tokenized text.
    """
    def __init__(self, root_dir, url="train-clean-100", download=False):
        """
        root_dir: path to LibriSpeech data (or where to download to).
        url: which subset to use (e.g., 'train-clean-100', 'dev-clean').
        """
        self.dataset = torchaudio.datasets.LIBRISPEECH(root_dir, url=url, download=download)
        # Transformation: waveform -> Mel spectrogram (80 mel bins)
        self.sample_rate = 16000
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=80, 
                                                                   n_fft=400, win_length=400, hop_length=160)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        waveform, sr, transcript, speaker_id, chapter_id, utterance_id = self.dataset[index]
        # Resample if needed (LibriSpeech is 16kHz by default)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        # Compute log-mel spectrogram
        mel_spec = self.mel_transform(waveform)  # shape: (1, n_mels, time)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)         # -> (time, n_mels)
        mel_spec = torch.log(mel_spec + 1e-6)                  # log-scale mel
        # Tokenize transcript (uppercase letters, space, apostrophe only)
        transcript = transcript.strip().upper()
        tokens = [char_to_idx[c] for c in transcript if c in char_to_idx]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return mel_spec, tokens

def collate_fn(batch):
    """
    Collate function to pad sequences and prepare lengths for a batch of (mel, tokens) pairs.
    Returns:
      mel_padded (batch, T_max, n_mels), mel_lengths,
      tokens_concat (concat of all token sequences), token_lengths.
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # sort by descending audio length
    mels, token_seqs = zip(*batch)
    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    token_lengths = torch.tensor([t.shape[0] for t in token_seqs], dtype=torch.long)
    # Pad mel sequences with zeros
    max_mel_len = mel_lengths.max().item()
    n_mels = mels[0].shape[1]
    mel_padded = torch.zeros(len(mels), max_mel_len, n_mels)
    for i, mel in enumerate(mels):
        mel_padded[i, :mel.shape[0], :] = mel
    # Concatenate token sequences (for CTC loss input)
    tokens_concat = torch.cat(token_seqs, dim=0)
    return mel_padded, mel_lengths, tokens_concat, token_lengths
