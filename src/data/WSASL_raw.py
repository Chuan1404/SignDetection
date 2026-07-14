import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from config import ROOT


class WLASLLandmarksDataset(Dataset):

    def __init__(self, feature_dir, annotation_dir, fusion_component, max_samples=1000):
        self.feature_dir = feature_dir
        self.fusion_component = fusion_component

        self.samples = []

        with open(os.path.join(ROOT, "datasets", "annotations", "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        with open(os.path.join(annotation_dir), "r") as f:
            data = json.load(f)
            video_ids = [d["video_id"] for d in data]
            all_video_names = sorted(video_ids)

        for index, video_name in enumerate(all_video_names):
            if max_samples is not None and index >= max_samples:
                break

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
                    "text_path": text_path
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

        left_features = torch.tensor(left_features).float()
        right_features = torch.tensor(right_features).float()

        hand_features = torch.concatenate([left_features, right_features], dim=-1)
        pose_features = torch.tensor(pose_features).float()

        # hand_features = self.normalize_features(hand_features)

        features = self.fusion_component.fuse(pose_features, hand_features)
        T = features.shape[0]
        features = features.reshape(T, -1)

        return features, label_id

    def normalize_features(self, hand_features):

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

        x = x.reshape(T, -1)

        return x