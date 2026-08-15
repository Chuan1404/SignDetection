import numpy as np

from config import _N_POSE, _N_HAND

_COORD_DIM = 2
NUM_JOINTS = _N_POSE + 2 * _N_HAND

_LEFT_HAND_START = _N_POSE
_RIGHT_HAND_START = _N_POSE + _N_HAND

_POSE_LR_SWAP = {
    1: 2, 2: 1,  # vai trái <-> vai phải
    3: 4, 4: 3
}


def _build_swap_index():
    swap = np.arange(NUM_JOINTS)
    for a, b in _POSE_LR_SWAP.items():
        swap[a] = b
    for i in range(_N_HAND):
        swap[_LEFT_HAND_START + i] = _RIGHT_HAND_START + i
        swap[_RIGHT_HAND_START + i] = _LEFT_HAND_START + i
    return swap


_SWAP_IDX = _build_swap_index()


class SkeletonAugmentor:
    def __init__(
        self,
        mirror_prob=0.3,
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
        fused = np.asarray(fused, dtype=np.float32)
        T = fused.shape[0]

        # Reshape directly to (T, N, 2) since input is [x0, y0, ..., xN, yN]
        coords = fused.reshape(T, NUM_JOINTS, _COORD_DIM).copy()

        if np.random.rand() < self.mirror_prob:
            coords = self._mirror(coords)

        coords = self._rotate(coords)
        coords = self._scale(coords)
        coords = self._shift(coords)
        coords = self._add_noise(coords)

        coords = self._temporal_crop(coords)
        coords = self._frame_dropout(coords)
        coords = self._speed_perturb(coords)

        # Flatten directly back to [x0, y0, ..., xN, yN]
        coords = coords.reshape(coords.shape[0], -1)
        return coords.astype(np.float32)

    def _mirror(self, coords):
        coords = coords[:, _SWAP_IDX, :].copy()
        coords[..., 0] *= -1.0            # lật trục x (trái <-> phải)
        return coords

    def _rotate(self, coords):
        # Xoay nhẹ trong mặt phẳng ảnh — mô phỏng lệch góc camera / nghiêng đầu-thân.
        deg = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        return coords @ R.T               # (T, N, 2) @ (2, 2)

    def _scale(self, coords):
        s = np.random.uniform(*self.scale_range)
        return coords * s

    def _shift(self, coords):
        shift = np.random.uniform(-self.shift_range, self.shift_range, size=_COORD_DIM).astype(np.float32)
        return coords + shift

    def _add_noise(self, coords):
        noise = np.random.normal(0, self.noise_std, size=coords.shape).astype(np.float32)
        return coords + noise

    # ------------------------------------------------------------------
    # Thời gian
    # ------------------------------------------------------------------
    def _temporal_crop(self, coords):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.temporal_crop_ratio <= 0:
            return coords
        max_cut = max(1, int(T * self.temporal_crop_ratio))
        start = np.random.randint(0, max_cut + 1)
        end = T - np.random.randint(0, max_cut + 1)
        end = max(end, start + self.min_frames)
        end = min(end, T)
        return coords[start:end]

    def _frame_dropout(self, coords):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.frame_dropout_prob <= 0:
            return coords
        keep = np.random.rand(T) > self.frame_dropout_prob
        if keep.sum() < self.min_frames:
            return coords
        return coords[keep]

    def _speed_perturb(self, coords):
        T = coords.shape[0]
        if T < 2 * self.min_frames:
            return coords
        factor = np.random.uniform(*self.speed_perturb_range)
        new_T = max(self.min_frames, int(round(T * factor)))
        idx = np.linspace(0, T - 1, new_T).round().astype(int)
        return coords[idx]


# =============================================================================
# HandAugmentor — augment CHỈ riêng 2 bàn tay (không đụng tới pose)
# -----------------------------------------------------------------------------
# Dùng cho dữ liệu chỉ gồm 2 bàn tay đã fuse/chuẩn hoá riêng (ví dụ nhánh
# Stream B trong kiến trúc 2 luồng), KHÔNG phải toàn bộ khung xương pose+tay
# như SkeletonAugmentor ở trên.
#
# Input kỳ vọng: (T, NUM_HAND_JOINTS * 2), thứ tự khớp = [tay trái (N_HAND
# khớp)..., tay phải (N_HAND khớp)...], mỗi khớp là (x, y) — đúng thứ tự nối
# `[left, right]` như fusion_component.fuse() đang dùng.

NUM_HAND_JOINTS = 2 * _N_HAND
_HAND_LEFT_START = 0
_HAND_RIGHT_START = _N_HAND


def _build_hand_swap_index():
    """Chỉ hoán đổi khối tay trái <-> khối tay phải, không có pose để xử lý."""
    swap = np.arange(NUM_HAND_JOINTS)
    for i in range(_N_HAND):
        swap[_HAND_LEFT_START + i] = _HAND_RIGHT_START + i
        swap[_HAND_RIGHT_START + i] = _HAND_LEFT_START + i
    return swap


_HAND_SWAP_IDX = _build_hand_swap_index()


class HandAugmentor:
    """
    Augment riêng cho 2 bàn tay. Cùng bộ phép biến đổi không gian/thời gian
    như SkeletonAugmentor, chỉ khác:
        - _mirror hoán đổi trái<->phải CHỈ trong phạm vi 2 khối tay (không có
          pose nên không cần _POSE_LR_SWAP).
        - Mặc định biên độ rotate/scale/shift/noise nhỏ hơn SkeletonAugmentor
          vì handshape (hình dạng ngón tay) nhạy với biến dạng hơn nhiều so
          với pose/vị trí toàn thân — xoay/co giãn quá tay rất dễ biến 1
          ký hiệu thành ký hiệu khác (đổi hẳn nghĩa), khác với pose (chỉ ảnh
          hưởng vị trí/quỹ đạo, ít đổi nghĩa hơn khi biến dạng nhẹ).
    """

    def __init__(
        self,
        mirror_prob=0.3,
        rotation_deg=6.0,
        scale_range=(0.95, 1.05),
        shift_range=0.03,
        noise_std=0.005,
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

    def __call__(self, hand_fused):
        hand_fused = np.asarray(hand_fused, dtype=np.float32)
        T = hand_fused.shape[0]

        coords = hand_fused.reshape(T, NUM_HAND_JOINTS, _COORD_DIM).copy()

        if np.random.rand() < self.mirror_prob:
            coords = self._mirror(coords)

        coords = self._rotate(coords)
        coords = self._scale(coords)
        coords = self._shift(coords)
        coords = self._add_noise(coords)

        coords = self._temporal_crop(coords)
        coords = self._frame_dropout(coords)
        coords = self._speed_perturb(coords)

        coords = coords.reshape(coords.shape[0], -1)
        return coords.astype(np.float32)

    # ------------------------------------------------------------------
    # Không gian
    # ------------------------------------------------------------------
    def _mirror(self, coords):
        coords = coords[:, _HAND_SWAP_IDX, :].copy()
        coords[..., 0] *= -1.0
        return coords

    def _rotate(self, coords):
        deg = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        return coords @ R.T

    def _scale(self, coords):
        s = np.random.uniform(*self.scale_range)
        return coords * s

    def _shift(self, coords):
        shift = np.random.uniform(-self.shift_range, self.shift_range, size=_COORD_DIM).astype(np.float32)
        return coords + shift

    def _add_noise(self, coords):
        noise = np.random.normal(0, self.noise_std, size=coords.shape).astype(np.float32)
        return coords + noise

    # ------------------------------------------------------------------
    # Thời gian (giống hệt SkeletonAugmentor, tách riêng để 2 class độc lập,
    # không phụ thuộc lẫn nhau nếu sau này chỉnh riêng từng bên)
    # ------------------------------------------------------------------
    def _temporal_crop(self, coords):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.temporal_crop_ratio <= 0:
            return coords
        max_cut = max(1, int(T * self.temporal_crop_ratio))
        start = np.random.randint(0, max_cut + 1)
        end = T - np.random.randint(0, max_cut + 1)
        end = max(end, start + self.min_frames)
        end = min(end, T)
        return coords[start:end]

    def _frame_dropout(self, coords):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.frame_dropout_prob <= 0:
            return coords
        keep = np.random.rand(T) > self.frame_dropout_prob
        if keep.sum() < self.min_frames:
            return coords
        return coords[keep]

    def _speed_perturb(self, coords):
        T = coords.shape[0]
        if T < 2 * self.min_frames:
            return coords
        factor = np.random.uniform(*self.speed_perturb_range)
        new_T = max(self.min_frames, int(round(T * factor)))
        idx = np.linspace(0, T - 1, new_T).round().astype(int)
        return coords[idx]


class AugmentedSkeletonDataset:

    def __init__(self, base_dataset, augmentor=None, num_augmentations=1):
        if num_augmentations < 0:
            raise ValueError("num_augmentations phải >= 0")

        self.base_dataset = base_dataset
        self.augmentor = augmentor or SkeletonAugmentor()
        self.num_augmentations = num_augmentations
        self.base_len = len(base_dataset)

    def __len__(self):
        return self.base_len * (1 + self.num_augmentations)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        base_idx = idx % self.base_len
        variant = idx // self.base_len          # 0 = gốc, >=1 = bản augment

        feature, label = self.base_dataset[base_idx]

        if variant == 0:
            return np.array(feature, dtype=np.float32, copy=True), label

        feature_aug = self.augmentor(feature)   # augmentor tự copy bên trong, không đụng `feature` gốc
        return feature_aug, label