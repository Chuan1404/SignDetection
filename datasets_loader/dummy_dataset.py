from torch.utils.data import Dataset, DataLoader
import torch

class DummySignDataset(Dataset):
    def __init__(self, n_samples=50):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sign_x = torch.randn(20, 512)   # continuous signing
        finger_x = torch.randn(128, 20) # fingerspelling
        lip_x = torch.randn(20, 256)    # lipreading
        tgt_seq = torch.randint(0, 5000, (20,)) # target tokens
        return sign_x, finger_x, lip_x, tgt_seq