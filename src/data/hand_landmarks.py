import os
from torch.utils.data import Dataset
import numpy as np
import torch

class HandLandmarksDataset(Dataset):

    def __init__(self, feature_dir, vocabulary):
        self.vocabulary = vocabulary
        self.feature_dir = feature_dir

        self.samples = []

        for index, video_name in enumerate(os.listdir(feature_dir)):
            # if index > 10000:
            #     break
            video_dir = os.path.join(
                feature_dir,
                video_name
            )

            if not os.path.isdir(video_dir):
                continue

            hand_path = os.path.join(
                video_dir,
                "hand_features.npy"
            )

            text_path = os.path.join(
                video_dir,
                "text.txt"
            )

            if (
                os.path.exists(hand_path)
                and os.path.exists(text_path)
            ):

                self.samples.append({
                    "hand_features": hand_path,
                    "text": text_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        hand_features = np.load(item["hand_features"])

        with open(item["text"], "r", encoding="utf-8") as f:
            sentence = f.read().strip()

        tokens = self.vocabulary(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )

        hand_features = torch.tensor(hand_features).float()

        text_ids = tokens["input_ids"].squeeze(0)

        hand_features = self.normalize_features(hand_features)

        return hand_features, text_ids

    # def normalize_features(self, hand_features):
    #     """
    #     hand_features: (T, 126)
    #     """
    #
    #     T = hand_features.shape[0]
    #
    #     x = hand_features.reshape(T, 2, 21, 3)
    #
    #     right_wrist = x[:, 0, 0, :].unsqueeze(1)  # (T, 1, 3)
    #
    #     left_wrist = x[:, 1, 0, :].unsqueeze(1)  # (T, 1, 3)
    #
    #     x[:, 0] = x[:, 0] - right_wrist
    #     x[:, 1] = x[:, 1] - left_wrist
    #
    #     # # Right hand scale (use joint 9 like your original)
    #     # right_scale = torch.norm(x[:, 0, 9, :], dim=-1, keepdim=True)
    #     # right_scale = torch.clamp(right_scale, min=1e-6)
    #     # x[:, 0] = x[:, 0] / right_scale.unsqueeze(-1)
    #     #
    #     # # Left hand scale
    #     # left_scale = torch.norm(x[:, 1, 9, :], dim=-1, keepdim=True)
    #     # left_scale = torch.clamp(left_scale, min=1e-6)
    #     # x[:, 1] = x[:, 1] / left_scale.unsqueeze(-1)
    #
    #     return x.reshape(T, 126)

    def normalize_features(self, hand_features):
        """
        hand_features: (T, 126)
        format assumed:
            2 hands × 21 joints × 3 coords
        """

        T = hand_features.shape[0]

        x = hand_features.reshape(T, 2, 21, 3)

        # use first frame as root
        right_wrist_0 = x[0, 0, 0, :].clone()
        left_wrist_0 = x[0, 1, 0, :].clone()

        x[:, 0] = x[:, 0] - right_wrist_0
        x[:, 1] = x[:, 1] - left_wrist_0



        scale_right = torch.norm(x[:, 0], dim=-1).mean()
        scale_left = torch.norm(x[:, 1], dim=-1).mean()

        scale = torch.clamp((scale_right + scale_left) / 2, min=1e-6)

        x = x / scale


        # velocity = torch.zeros_like(x)
        # velocity[1:] = x[1:] - x[:-1]

        # concat position + velocity (helps semantic learning)
        # x = torch.cat([x, velocity], dim=-1)  # (T,2,21,6)

        x = x.reshape(T, -1)

        return x