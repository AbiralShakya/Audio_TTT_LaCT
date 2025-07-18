import os
import re
import torch
import torchaudio
from torch.utils.data import Dataset

# Define character set for TTS (letters, punctuation, space). 0 will be used for padding.
TTS_CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + list("!'\",.-?;: ")
TTS_CHARACTERS = sorted(set(TTS_CHARACTERS), key=TTS_CHARACTERS.index)  # unique, preserve order
tts_char_to_idx = {c: i+1 for i, c in enumerate(TTS_CHARACTERS)}  # map characters to 1...N (0 is pad)
tts_idx_to_char = {i+1: c for i, c in enumerate(TTS_CHARACTERS)}

class LJSpeechDataset(Dataset):
    """
    Dataset for LJSpeech TTS. Provides (text_indices, mel_spectrogram) pairs.
    """
    def __init__(self, root_dir):
        """
        root_dir: path to LJSpeech dataset (contains 'metadata.csv' and 'wavs/' folder).
        """
        metadata_path = os.path.join(root_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        # Read metadata lines
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.entries = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            file_id = parts[0]
            text = parts[-1] if len(parts) > 2 else parts[1]  # use normalized text if available
            text = text.strip()
            # Remove characters not in our set (simple cleaning)
            text = re.sub(f"[^{''.join(TTS_CHARACTERS)}]", "", text)
            self.entries.append((file_id, text))
        # Audio processing setup
        self.sr = 22050  # LJSpeech sampling rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=80, 
                                                                   n_fft=1024, hop_length=256, win_length=1024)
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        file_id, text = self.entries[index]
        # Load audio waveform
        wav_path = os.path.join(self.root_dir, "wavs", f"{file_id}.wav")
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        # Compute log-mel spectrogram
        mel_spec = self.mel_transform(waveform)              # (1, n_mels, time)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)       # -> (time, n_mels)
        mel_spec = torch.log(mel_spec + 1e-6)
        # Convert text to indices
        text_indices = [tts_char_to_idx[c] for c in text if c in tts_char_to_idx]
        text_tensor = torch.tensor(text_indices, dtype=torch.long)
        return text_tensor, mel_spec

def tts_collate_fn(batch):
    """
    Collate function for LJSpeech data.
    Returns:
      padded_text (batch, max_text_len), text_lengths,
      padded_mel (batch, max_mel_len, n_mels), mel_lengths.
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # sort by text length
    texts, mels = zip(*batch)
    text_lengths = [t.size(0) for t in texts]
    mel_lengths = [m.size(0) for m in mels]
    max_text_len = max(text_lengths)
    max_mel_len = max(mel_lengths)
    # Pad text sequences with 0 (padding idx)
    padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_text[i, :t.size(0)] = t
    # Pad mel spectrograms with 0 (silence)
    n_mels = mels[0].shape[1] if mels else 80
    padded_mel = torch.zeros(len(batch), max_mel_len, n_mels)
    for i, m in enumerate(mels):
        padded_mel[i, :m.shape[0], :] = m
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
    return padded_text, text_lengths, padded_mel, mel_lengths
