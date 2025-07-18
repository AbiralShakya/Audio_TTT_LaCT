import torchaudio
from torch.utils.data import DataLoader, Dataset

class LibriSpeechASRDataset(Dataset):
    def __init__(self, root, url="test-clean", download=True):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=download)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        return waveform, sample_rate, transcript

def get_librispeech_loader(root, url="test-clean", batch_size=1, num_workers=0):
    dataset = LibriSpeechASRDataset(root, url=url)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 