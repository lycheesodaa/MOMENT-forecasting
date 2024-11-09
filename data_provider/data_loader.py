import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

class CS702TrainDataset(Dataset):
    def __init__(self, file_name="train.npy", folder_path="./dataset", seq_len=14, candidate_len=3,
                 flag='full', percent=100):
        super().__init__()
        seed = 123
        train_ratio = 0.8
        val_ratio = 0.2

        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.data = self.data[:int(len(self.data) * percent // 100)]

        if flag == 'train':
            self.data, _ = random_split(self.data, [train_ratio, val_ratio], torch.Generator().manual_seed(seed))
        elif flag == 'val':
            _, self.data = random_split(self.data, [train_ratio, val_ratio], torch.Generator().manual_seed(seed))
            
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len - 1

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        input_mask = np.ones(self.seq_len - self.candidate_len - 1)
        
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len - 1]  # exclude the last one
        next_point = self.data[(idx + 1) * self.seq_len - 1]

        labels = torch.tensor([3.0, 1, 0])

        return seq, cdd, next_point, labels, input_mask


class CS702TestDataset(Dataset):
    def __init__(self, file_name="public.npy", folder_path="./dataset", seq_len=13, candidate_len=3,
                 flag='train', percent=100):
        super().__init__()
        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        input_mask = np.ones(self.seq_len - self.candidate_len)
        
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len]

        return seq, cdd, input_mask