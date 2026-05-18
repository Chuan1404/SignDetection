import os
from torch.utils.data import Dataset
import numpy as np
import torch

class SLTDataset(Dataset):

    def __init__(self, feature_dir):

        self.feature_dir = feature_dir

        self.samples = []

        for video_name in os.listdir(feature_dir):

            video_dir = os.path.join(
                feature_dir,
                video_name
            )

            if not os.path.isdir(video_dir):
                continue

            left_path = os.path.join(
                video_dir,
                "left_feat.npy"
            )

            right_path = os.path.join(
                video_dir,
                "right_feat.npy"
            )

            text_path = os.path.join(
                video_dir,
                "text_ids.pt"
            )

            if (
                os.path.exists(left_path)
                and os.path.exists(right_path)
                and os.path.exists(text_path)
            ):

                self.samples.append({
                    "left": left_path,
                    "right": right_path,
                    "text": text_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        left = np.load(item["left"])
        right = np.load(item["right"])

        text = torch.load(item["text"])

        left = torch.tensor(left).float()
        right = torch.tensor(right).float()

        text = torch.tensor(text).long()

        return left, right, text