import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

base_i3d = r"C:\Users\ADMIN\OneDrive\Desktop\SignDetection\datasets\how2sign\i3d_features_how2sign\i3d_features_how2sign\train"
base_mp = r"C:\Users\ADMIN\OneDrive\Desktop\SignDetection\datasets\how2sign\mediapipe_features_how2sign\mediapipe_features\train"

class How2SignDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path,sep="\t")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # (frames, feature_dimension)
        i3d = np.load(os.path.join(base_i3d, row["id"] + ".npy"))

        # (frames, body landmarks, xyz)
        mp = np.load(os.path.join(base_mp, row["id"] + ".npy"))
        T = mp.shape[0]
        mp = np.reshape(mp, (T, -1))


        text = self.tokenizer.encode(row["translation"])

        return (
            torch.tensor(i3d, dtype=torch.float32),
            torch.tensor(mp, dtype=torch.float32),
            torch.tensor(text, dtype=torch.long)
        )
