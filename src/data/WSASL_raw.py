import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

class WLASLLandmarksDataset(Dataset):

    def __init__(self, feature_dir, annotation_dir, fusion_component, mode="train"):
        self.feature_dir = feature_dir
        self.fusion_component = fusion_component

        self.samples = []

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
            data = json.load(f)
            video_ids = [d["video_id"] for d in data]
            all_video_names = sorted(video_ids)

        for index, video_name in enumerate(all_video_names):

            video_dir = os.path.join(
                feature_dir,
                video_name
            )

            if not os.path.isdir(video_dir):
                continue

            left_hand_path = os.path.join(video_dir, "left_hand.npy")
            right_hand_path = os.path.join(video_dir, "right_hand.npy")
            pose_path = os.path.join(video_dir, "pose.npy")

            text_path = os.path.join(video_dir, "gloss.txt")

            if (
                    os.path.exists(left_hand_path)
                    and os.path.exists(right_hand_path)
                    and os.path.exists(pose_path)
                    and os.path.exists(text_path)
            ):
                self.samples.append({
                    "left_hand_path": left_hand_path,
                    "right_hand_path": right_hand_path,
                    "pose_path": pose_path,
                    "text_path": text_path,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        left_features = np.load(item["left_hand_path"])
        right_features = np.load(item["right_hand_path"])
        pose_features = np.load(item["pose_path"])

        with open(item["text_path"], "r", encoding="utf-8") as f:
            gloss = f.read().strip()
        label_id = self.gloss2idx[gloss]

        left_present = ~np.all(left_features == 0, axis=-1)  # (T,)
        right_present = ~np.all(right_features == 0, axis=-1)  # (T,)
        any_present = left_present | right_present

        start_frame = int(np.argmax(any_present)) if any_present.any() else 0

        left_features = left_features[start_frame:]
        right_features = right_features[start_frame:]
        pose_features = pose_features[start_frame:]

        left_features = torch.tensor(left_features).float()
        right_features = torch.tensor(right_features).float()

        hand_features = torch.concatenate([left_features, right_features], dim=-1)
        pose_features = torch.tensor(pose_features).float()

        # hand_normalize_features = self.normalize_features(hand_features)
        features = self.fusion_component.fuse(pose_features, hand_features)
        T = features.shape[0]
        features = features.reshape(T, -1)

        return features, label_id

    def normalize_features(self, hand_features):
        T = hand_features.shape[0]

        x = hand_features.reshape(T, 2, 21, 3).clone()

        # index 0 is left hand, index 1 is right hand
        left_wrist = x[:, 0, 0:1, :]  # (T,1,3)
        right_wrist = x[:, 1, 0:1, :]
        
        x[:, 0] = x[:, 0] - left_wrist
        x[:, 1] = x[:, 1] - right_wrist

        # Find valid hands (not all zeros)
        valid_left = (x[:, 0].abs().sum(dim=(1, 2)) > 1e-5)
        valid_right = (x[:, 1].abs().sum(dim=(1, 2)) > 1e-5)

        # Calculate scale using only valid frames. Fallback to 1.0 if completely missing
        scale_left = torch.norm(x[valid_left, 0], dim=-1).mean() if valid_left.any() else torch.tensor(1.0)
        scale_right = torch.norm(x[valid_right, 1], dim=-1).mean() if valid_right.any() else torch.tensor(1.0)

        if valid_left.any() and valid_right.any():
            scale = (scale_left + scale_right) / 2.0
        elif valid_left.any():
            scale = scale_left
        else:
            scale = scale_right

        scale = torch.clamp(torch.tensor(scale), min=1e-6)

        x = x / scale
        x = x.reshape(T, -1)

        return x