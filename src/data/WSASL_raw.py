import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import ROOT

NUM_JOINTS = 47
_COORD_DIM = NUM_JOINTS * 3   # 141
_TOTAL_DIM = _COORD_DIM + 2   # 143

_N_POSE = 7
_N_HAND = 20
_LEFT_HAND_START = _N_POSE            # 7
_RIGHT_HAND_START = _N_POSE + _N_HAND  # 27


# -----------------------------------------------------------------------
# Bảng hoán đổi trái/phải cho phép mirror (lật gương theo trục dọc cơ thể)
# -----------------------------------------------------------------------
_POSE_LR_SWAP = {
    1: 2, 2: 1,  # shoulder
    3: 4, 4: 3,  # wrist*
    5: 6, 6: 5,  # hip
    # idx 0 (nose) nằm trên trục giữa -> giữ nguyên
}


def _build_swap_index():
    swap = np.arange(NUM_JOINTS)
    for a, b in _POSE_LR_SWAP.items():
        swap[a] = b
    # left hand (7-26) <-> right hand (27-46), cùng thứ tự ngón/đốt
    for i in range(_N_HAND):
        swap[_LEFT_HAND_START + i] = _RIGHT_HAND_START + i
        swap[_RIGHT_HAND_START + i] = _LEFT_HAND_START + i
    return swap


_SWAP_IDX = _build_swap_index()

# Kiểm tra bất biến ngay khi import: swap phải là involution (áp 2 lần = identity)
assert np.array_equal(_SWAP_IDX[_SWAP_IDX], np.arange(NUM_JOINTS)), \
    "SWAP_IDX không hợp lệ — kiểm tra lại _POSE_LR_SWAP"

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



class SkeletonAugmentor:
    def __init__(
        self,
        mirror_prob=0.5,
        rotation_deg=13.0,
        scale_range=(0.9, 1.1),
        shift_range=0.05,
        noise_std=0.01,
        temporal_crop_ratio=0.1,
        frame_dropout_prob=0.05,
        speed_perturb_range=(0.85, 1.15),
        min_frames=5,
    ):
        self.mirror_prob = mirror_prob
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.noise_std = noise_std
        self.temporal_crop_ratio = temporal_crop_ratio
        self.frame_dropout_prob = frame_dropout_prob
        self.speed_perturb_range = speed_perturb_range
        self.min_frames = min_frames

    def __call__(self, fused):
        """fused: array-like (T, 143) -> np.ndarray (T', 143) float32"""
        fused = np.asarray(fused, dtype=np.float32)
        assert fused.shape[1] == _TOTAL_DIM, (
            f"Kỳ vọng chiều cuối = {_TOTAL_DIM} (fusion mới, 47 node), "
            f"nhận được {fused.shape[1]}. Kiểm tra lại fusion_component.py "
            f"có đang dùng đúng _REMOVE_POSE_IDX hiện tại không."
        )
        T = fused.shape[0]

        coords = fused[:, :_COORD_DIM].reshape(T, NUM_JOINTS, 3).copy()
        flags = fused[:, _COORD_DIM:].copy()  # (T, 2)

        # ---------- Không gian ----------
        if np.random.rand() < self.mirror_prob:
            coords, flags = self._mirror(coords, flags)

        coords = self._rotate(coords)
        coords = self._scale(coords)
        coords = self._shift(coords)
        coords = self._add_noise(coords)

        # ---------- Thời gian ----------
        coords, flags = self._temporal_crop(coords, flags)
        coords, flags = self._frame_dropout(coords, flags)
        coords, flags = self._speed_perturb(coords, flags)

        out = np.concatenate(
            [coords.reshape(coords.shape[0], -1), flags], axis=1
        ).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Không gian
    # ------------------------------------------------------------------
    def _mirror(self, coords, flags):
        coords = coords[:, _SWAP_IDX, :].copy()
        coords[..., 0] *= -1.0            # lật trục x (trái <-> phải)
        flags = flags[:, [1, 0]].copy()   # swap cờ left/right present
        return coords, flags

    def _rotate(self, coords):
        # Xoay nhẹ quanh trục z (mặt phẳng ảnh) — mô phỏng lệch góc camera
        # / góc nghiêng đầu-thân của người ký hiệu.
        deg = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        xy = coords[..., :2]
        coords[..., :2] = xy @ R.T
        return coords

    def _scale(self, coords):
        s = np.random.uniform(*self.scale_range)
        return coords * s

    def _shift(self, coords):
        shift = np.random.uniform(-self.shift_range, self.shift_range, size=3).astype(np.float32)
        return coords + shift

    def _add_noise(self, coords):
        noise = np.random.normal(0, self.noise_std, size=coords.shape).astype(np.float32)
        return coords + noise

    # ------------------------------------------------------------------
    # Thời gian
    # ------------------------------------------------------------------
    def _temporal_crop(self, coords, flags):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.temporal_crop_ratio <= 0:
            return coords, flags
        max_cut = max(1, int(T * self.temporal_crop_ratio))
        start = np.random.randint(0, max_cut + 1)
        end = T - np.random.randint(0, max_cut + 1)
        end = max(end, start + self.min_frames)
        end = min(end, T)
        return coords[start:end], flags[start:end]

    def _frame_dropout(self, coords, flags):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.frame_dropout_prob <= 0:
            return coords, flags
        keep = np.random.rand(T) > self.frame_dropout_prob
        if keep.sum() < self.min_frames:
            return coords, flags
        return coords[keep], flags[keep]

    def _speed_perturb(self, coords, flags):
        T = coords.shape[0]
        if T < 2 * self.min_frames:
            return coords, flags
        factor = np.random.uniform(*self.speed_perturb_range)
        new_T = max(self.min_frames, int(round(T * factor)))
        idx = np.linspace(0, T - 1, new_T).round().astype(int)
        return coords[idx], flags[idx]


# -----------------------------------------------------------------------
# Wrapper Dataset — KHÔNG cần sửa WLASLLandmarksDataset gốc.
# Chỉ bọc quanh train_dataset (Subset), val_dataset giữ nguyên như cũ.
# -----------------------------------------------------------------------
class AugmentedSkeletonDataset:
    """
    Bọc quanh 1 Dataset/Subset đã có sẵn (trả về (feature, label)),
    áp dụng SkeletonAugmentor lên `feature` trước khi trả ra.

    Dùng trong main.py:

        from skeleton_augment import SkeletonAugmentor, AugmentedSkeletonDataset

        train_dataset = AugmentedSkeletonDataset(
            Subset(base_dataset, train_indices), SkeletonAugmentor()
        )
        val_dataset = Subset(base_dataset, val_indices)   # KHÔNG augment val/test
    """

    def __init__(self, base_dataset, augmentor):
        self.base_dataset = base_dataset
        self.augmentor = augmentor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature, label = self.base_dataset[idx]
        feature = self.augmentor(feature)
        return feature, label