import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MidiDataset(Dataset):

    def __init__(self, npy_training_file, npy_labels_file):
        """
        Args:
            npy_training_file (string): Path to the npy file with the training midi time series.
            npy_labels_file (string): Path to the npy file with the labels to the corresponding training file
        """
        np_training_data = np.load(npy_training_file).astype('float32')
        np_labels_data = np.load(npy_labels_file).astype('float32')

        
        self.training_data = torch.from_numpy(np_training_data)
        self.labels_data = torch.from_numpy(np_labels_data)
                          
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx,:,:], self.labels_data[idx,:]