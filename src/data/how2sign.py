import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

class How2SignDataset(Dataset):
    def __init__(self, csv_path, tokenizer, base_i3d='i3d', base_mp='mp'):
        self.df = pd.read_csv(csv_path,sep="\t")
        self.tokenizer = tokenizer
        self.base_i3d = base_i3d
        self.base_mp = base_mp

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # (frames, feature_dimension)
        i3d = np.load(os.path.join(self.base_i3d, f"{row['id']}.npy"))

        # (frames, body landmarks, xyz)
        mp = np.load(os.path.join(self.base_mp, f"{row['id']}.npy"))

        text = self.tokenizer.encode(row["translation"])

        return (
            torch.tensor(i3d, dtype=torch.float32),
            torch.tensor(mp, dtype=torch.float32),
            torch.tensor(text, dtype=torch.long)
        )
