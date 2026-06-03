import os
from torch.utils.data import Dataset
import numpy as np
import torch

class HandLandmarksDataset(Dataset):

    def __init__(self, feature_dir, vocabulary, max_length=64):
        self.vocabulary = vocabulary
        self.feature_dir = feature_dir

        self.samples = []

        for index, video_name in enumerate(os.listdir(feature_dir)):
            if index >= max_length:
                break

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
        text_mask = tokens["attention_mask"].squeeze(0)

        hand_features = self.normalize_features(hand_features)

        return hand_features, text_ids, text_mask

    def normalize_features(self, hand_features):
        T = hand_features.shape[0]

        hand_features = hand_features.reshape(T, 2, 21, 3)
        # Right hand
        right_wrist = hand_features[:, 0, 0:1, :]

        right_scale = torch.norm(hand_features[:, 0, 9], dim=-1, keepdim=True)
        right_scale = torch.clamp(right_scale, min=1e-6)
        hand_features[:, 0] /= right_scale.unsqueeze(-1)

        # Left hand
        left_wrist = hand_features[:, 1, 0:1, :]
        hand_features[:, 1] -= left_wrist

        left_scale = torch.norm(hand_features[:, 1, 9], dim=-1, keepdim=True)
        left_scale = torch.clamp(left_scale, min=1e-6)

        hand_features[:, 1] /= left_scale.unsqueeze(-1)

        return hand_features.reshape(T, 126)
