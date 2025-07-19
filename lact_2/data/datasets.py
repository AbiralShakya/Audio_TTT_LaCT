# file: data/datasets.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.text import text_to_sequence

class LibriSpeechDataset(Dataset):
    """
    LibriSpeech ASR Dataset wrapper that outputs log-mel spectrogram features and text transcripts.
    """
    def __init__(self, root, url="train-clean-100", download=False):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)
        # Define mel spectrogram transform (for 16kHz audio, 25ms window, 10ms hop, 80 mel bins)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        # Resample if needed (LibriSpeech is 16kHz already, but just in case):
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        # Compute mel spectrogram (use only the first channel if stereo)
        if waveform.dim() > 1:
            waveform = waveform[0:1, :]  # take first channel
        mel_spec = self.mel_transform(waveform)  # shape: (n_mels, time)
        mel_spec = self.amplitude_to_db(mel_spec)  # log-scale mel
        mel_spec = mel_spec.transpose(0, 1)  # shape: (time, n_mels)
        # Normalize (optional): could subtract mean and divide std per feature (not done here for simplicity)
        # Prepare transcript (uppercase and remove unsupported chars)
        text = transcript.upper()
        text_indices = text_to_sequence(text)
        text_indices = torch.LongTensor(text_indices)
        # Return features and transcript indices
        return mel_spec, text_indices

def librispeech_collate(batch):
    """
    Collate function for LibriSpeech ASR.
    Pads spectrograms and transcripts in the batch.
    Returns:
      - padded_mel: (batch, T_max, n_mels)
      - mel_lengths: (batch,) lengths of each spectrogram
      - concatenated_targets: (sum_all_target_len,) int tensor of all transcription tokens concatenated
      - target_lengths: (batch,) lengths of each transcription
    """
    # Separate mel and text from batch
    mel_batch = [item[0] for item in batch]
    text_batch = [item[1] for item in batch]
    # Sort by descending mel length (for RNNT/CTC efficiency if needed, not strictly necessary here)
    # but maintain parallel sorting for text accordingly
    lengths = [mel.shape[0] for mel in mel_batch]
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    mel_batch = [mel_batch[i] for i in sorted_idx]
    text_batch = [text_batch[i] for i in sorted_idx]
    mel_lengths = torch.LongTensor([mel.shape[0] for mel in mel_batch])
    # Pad mel sequences
    n_mels = mel_batch[0].shape[1]
    max_len = mel_batch[0].shape[0]
    padded_mels = torch.zeros(len(mel_batch), max_len, n_mels)
    for i, mel in enumerate(mel_batch):
        T = mel.shape[0]
        padded_mels[i, :T, :] = mel
    # Concatenate text targets and record lengths
    target_lengths = torch.LongTensor([t.shape[0] for t in text_batch])
    concatenated_targets = torch.cat(text_batch, dim=0)  # 1D tensor of all tokens
    return padded_mels, mel_lengths, concatenated_targets, target_lengths

class LJSpeechDataset(Dataset):
    """
    LJSpeech TTS Dataset wrapper that outputs text and mel-spectrogram pairs.
    """
    def __init__(self, root, download=False):
        self.dataset = torchaudio.datasets.LJSPEECH(root=root, download=download)
        # Mel spectrogram transform for LJSpeech (22.05kHz audio)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, transcript, normalized_text = self.dataset[idx]
        # Resample if needed (LJSpeech is 22050Hz)
        if sample_rate != 22050:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=22050)
            waveform = resampler(waveform)
            sample_rate = 22050
        # Compute mel spectrogram
        if waveform.dim() > 1:
            waveform = waveform[0:1, :]  # use mono
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = mel_spec.transpose(0, 1)  # (time, n_mels)
        # Prepare text
        text = normalized_text.upper()
        text_indices = text_to_sequence(text)
        text_indices = torch.LongTensor(text_indices)
        return text_indices, mel_spec

def ljspeech_collate(batch):
    """
    Collate function for LJSpeech TTS.
    Pads text and mel sequences.
    Returns:
      - padded_text: (batch, max_T_text) LongTensor
      - text_lengths: (batch,) lengths of each text
      - padded_mel: (batch, max_T_mel, n_mels) FloatTensor
      - mel_lengths: (batch,) lengths of each mel
    """
    text_batch = [item[0] for item in batch]
    mel_batch = [item[1] for item in batch]
    # Sort by descending text length (optional)
    text_lengths = [t.shape[0] for t in text_batch]
    sorted_idx = sorted(range(len(text_lengths)), key=lambda i: text_lengths[i], reverse=True)
    text_batch = [text_batch[i] for i in sorted_idx]
    mel_batch = [mel_batch[i] for i in sorted_idx]
    text_lengths = torch.LongTensor([t.shape[0] for t in text_batch])
    mel_lengths = torch.LongTensor([mel.shape[0] for mel in mel_batch])
    # Pad text
    max_text = text_batch[0].shape[0]
    padded_text = torch.zeros(len(text_batch), max_text, dtype=torch.long)
    for i, t in enumerate(text_batch):
        padded_text[i, :t.shape[0]] = t
    # Pad mel
    n_mels = mel_batch[0].shape[1]
    max_mel = mel_batch[0].shape[0]
    padded_mel = torch.zeros(len(mel_batch), max_mel, n_mels)
    for i, mel in enumerate(mel_batch):
        padded_mel[i, :mel.shape[0], :] = mel
    return padded_text, text_lengths, padded_mel, mel_lengths
