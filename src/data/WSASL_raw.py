import os
import json
import random
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset
from config import ROOT


class WLASLLandmarksDataset(Dataset):

    def __init__(
        self,
        feature_dir,
        fusion_component,
        split="train",
        max_samples=None
    ):

        self.feature_dir = feature_dir
        self.fusion_component = fusion_component

        # Accept either a single split name or a list of split names
        # e.g. split="train"  or  split=["train", "val"]
        splits = [split] if isinstance(split, str) else split

        annotations = []
        for s in splits:
            annotation_path = os.path.join(
                ROOT, "datasets", "annotations", f"{s}.json"
            )
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotations.extend(json.load(f))

        with open(os.path.join(ROOT, "datasets", "annotations", "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        if max_samples is not None:
            annotations = annotations[:max_samples]

        self.samples = []

        for sample in annotations:

            video_dir = os.path.join(
                feature_dir,
                sample["video_id"]
            )

            fused_path = os.path.join(video_dir, "fused.npy")
            left_hand_path = os.path.join(video_dir, "left_hand.npy")
            right_hand_path = os.path.join(video_dir, "right_hand.npy")
            pose_path = os.path.join(video_dir, "pose.npy")

            text_path = os.path.join(video_dir, "gloss.txt")

            if os.path.exists(fused_path) and os.path.exists(text_path):
                self.samples.append(
                    {
                        "fused_path": fused_path,
                        "text_path": text_path,
                        "has_prefused": True
                    }
                )
            elif (
                os.path.exists(left_hand_path)
                and os.path.exists(right_hand_path)
                and os.path.exists(pose_path)
                and os.path.exists(text_path)
            ):
                self.samples.append(
                    {
                        "left_hand_path": left_hand_path,
                        "right_hand_path": right_hand_path,
                        "pose_path": pose_path,
                        "text_path": text_path,
                        "has_prefused": False
                    }
                )

        split_label = "+".join(splits)
        print(f"{split_label}: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        """
        Reads class labels for all samples without loading heavy feature files.
        Used by stratified_split() to group samples by class.
        Returns a list of integer class indices, one per sample.
        """
        labels = []
        for item in self.samples:
            with open(item["text_path"], "r", encoding="utf-8") as f:
                gloss = f.read().strip().lower()
            labels.append(self.gloss2idx[gloss])
        return labels

    def __getitem__(self, idx):

        item = self.samples[idx]

        with open(item["text_path"], "r", encoding="utf-8") as f:
            gloss = f.read().strip().lower()

        label = self.gloss2idx[gloss]

        if item.get("has_prefused", False):
            features = np.load(item["fused_path"])
        else:
            left_features = np.load(item["left_hand_path"])
            right_features = np.load(item["right_hand_path"])
            pose_features = np.load(item["pose_path"])

            left_features = left_features.astype(np.float32)
            right_features = right_features.astype(np.float32)
            pose_features = pose_features.astype(np.float32)

            hand_features = np.concatenate(
                [left_features, right_features],
                axis=-1
            )

            features = self.fusion_component.fuse(
                pose_features,
                hand_features
            )

            T = features.shape[0]
            features = features.reshape(T, -1)

        return features, label


def stratified_split(dataset, val_ratio=0.1, seed=42):
    """
    Stratified train/val split that mirrors the original implementation but
    works directly with WLASLLandmarksDataset.

    Instead of calling __getitem__ (which loads heavy .npy files), we use
    dataset.get_labels() to read only the lightweight gloss .txt files,
    then group their indices by class label before splitting.

    Args:
        dataset   : WLASLLandmarksDataset — the full combined dataset.
        val_ratio : float — fraction of each class's samples to put in val.
        seed      : int   — random seed for reproducibility.

    Returns:
        train_subset, val_subset : torch.utils.data.Subset
    """
    rng = random.Random(seed)

    # Read all labels without loading feature files
    labels = dataset.get_labels()

    # Group indices by class
    class_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_to_indices[label].append(i)

    train_indices, val_indices = [], []
    for label, indices in class_to_indices.items():
        rng.shuffle(indices)
        # Classes with only 1 sample go entirely to train
        n_val = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)