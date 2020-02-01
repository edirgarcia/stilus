import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MidiDataset(Dataset):

    def __init__(self, npy_file):
        """
        Args:
            npy_file (string): Path to the npy file with the midi time series.
        """
        np_training_data = np.load(npy_file).astype('float32')
        self.time_series = torch.from_numpy(np_training_data)

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx]