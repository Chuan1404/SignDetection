import numpy as np

from config import _N_POSE, _N_LIPS, _N_HAND, _N_HAND_FLAGS

NUM_JOINTS = 45
_TOTAL_DIM = NUM_JOINTS * 3 + _N_HAND_FLAGS  # 257 total per frame
_COORD_DIM = 3

# Layout node trong FusionComponent.fuse():
#   [0        : _N_POSE          ]  pose   (3 nodes)
#   [_N_POSE  : _N_POSE+_N_LIPS  ]  lips   (40 nodes)
#   [_LEFT_HAND_START : +_N_HAND ]  left hand  (21 nodes)
#   [_RIGHT_HAND_START: +_N_HAND ]  right hand (21 nodes)
_LEFT_HAND_START  = _N_POSE            # 43
_RIGHT_HAND_START = _N_POSE + _N_HAND  # 64


# _POSE_LR_SWAP = {
#     1: 2, 2: 1,  # shoulder
# }
#
#
# def _build_swap_index():
#     swap = np.arange(NUM_JOINTS)
#     for a, b in _POSE_LR_SWAP.items():
#         swap[a] = b
#     for i in range(_N_HAND):
#         swap[_LEFT_HAND_START + i] = _RIGHT_HAND_START + i
#         swap[_RIGHT_HAND_START + i] = _LEFT_HAND_START + i
#     return swap
#
# _SWAP_IDX = _build_swap_index()


class SkeletonAugmentor:
    def __init__(
        self,
        mirror_prob=0.3,  # giảm xuống 0.3 vì một số ký hiệu phân biệt tay thuận
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

    def __call__(self, fused, hand_normalized):
        fused = np.asarray(fused, dtype=np.float32)
        hand_normalized = np.asarray(hand_normalized, dtype=np.float32)

        T = fused.shape[0]
        assert hand_normalized.shape[0] == T, (
            f"feature (T={T}) và hand_normalized_feature "
            f"(T={hand_normalized.shape[0]}) phải cùng số frame trước khi augment"
        )

        coords = fused[:, :NUM_JOINTS * 3].reshape(T, NUM_JOINTS, 3).copy()
        flags  = fused[:, NUM_JOINTS * 3:].copy()  # (T, 2)
        hand   = hand_normalized.reshape(T, 2, 21, 3).copy()  # (T, 2, 21, 3)

        # ---------- Không gian ----------
        # Áp dụng CHUNG một bộ tham số ngẫu nhiên cho cả coords lẫn hand
        # để hai luồng luôn nhất quán (quan trọng khi mirror được bật lại).
        # if np.random.rand() < self.mirror_prob:
        #     coords = self._mirror(coords)
        #     flags = flags[:, [1, 0]]
        #     hand = hand[:, [1, 0]].copy()
        #     hand[..., 0] *= -1.0

        coords, hand = self._rotate(coords, hand)
        coords, hand = self._scale(coords, hand)
        coords, hand = self._shift(coords, hand)
        coords, hand = self._add_noise(coords, hand)

        # ---------- Thời gian (dùng CHUNG chỉ số cho cả 2 luồng) ----------
        coords, flags, hand = self._temporal_crop(coords, flags, hand)
        coords, flags, hand = self._frame_dropout(coords, flags, hand)
        coords, flags, hand = self._speed_perturb(coords, flags, hand)

        out_fused = np.concatenate(
            [coords.reshape(coords.shape[0], -1), flags], axis=1
        ).astype(np.float32)
        out_hand = hand.reshape(hand.shape[0], -1).astype(np.float32)

        return out_fused, out_hand

    # ------------------------------------------------------------------
    # Không gian
    # ------------------------------------------------------------------
    # def _mirror(self, coords):
    #     coords = coords[:, _SWAP_IDX, :].copy()
    #     coords[..., 0] *= -1.0            # lật trục x (trái <-> phải)
    #     return coords

    def _rotate(self, coords, hand):
        # Xoay nhẹ quanh trục z (mặt phẳng ảnh) — mô phỏng lệch góc camera
        # / góc nghiêng đầu-thân của người ký hiệu.
        # Dùng CÙNG góc cho cả coords và hand để hai luồng nhất quán.
        deg = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        rad = np.deg2rad(deg)
        cos, sin = np.cos(rad), np.sin(rad)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        coords[..., :2] = coords[..., :2] @ R.T
        hand[..., :2]   = hand[..., :2]   @ R.T
        return coords, hand

    def _scale(self, coords, hand):
        s = np.random.uniform(*self.scale_range)
        # Scale coords nhưng KHÔNG scale hand vì hand đã normalize theo wrist
        # (scale theo wrist riêng), nhân thêm s sẽ phá scale đó.
        return coords * s, hand

    def _shift(self, coords, hand):
        shift = np.random.uniform(-self.shift_range, self.shift_range, size=3).astype(np.float32)
        # Shift chỉ áp cho coords (shoulder-normalized).
        # hand đã normalize theo wrist nên shift toàn cục không có nghĩa.
        return coords + shift, hand

    def _add_noise(self, coords, hand):
        coords_noise = np.random.normal(0, self.noise_std, size=coords.shape).astype(np.float32)
        hand_noise   = np.random.normal(0, self.noise_std, size=hand.shape).astype(np.float32)
        return coords + coords_noise, hand + hand_noise

    # ------------------------------------------------------------------
    # Thời gian
    # ------------------------------------------------------------------
    def _temporal_crop(self, coords, flags, hand):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.temporal_crop_ratio <= 0:
            return coords, flags, hand
        max_cut = max(1, int(T * self.temporal_crop_ratio))
        start = np.random.randint(0, max_cut + 1)
        end = T - np.random.randint(0, max_cut + 1)
        end = max(end, start + self.min_frames)
        end = min(end, T)
        return coords[start:end], flags[start:end], hand[start:end]

    def _frame_dropout(self, coords, flags, hand):
        T = coords.shape[0]
        if T < 2 * self.min_frames or self.frame_dropout_prob <= 0:
            return coords, flags, hand
        keep = np.random.rand(T) > self.frame_dropout_prob
        if keep.sum() < self.min_frames:
            return coords, flags, hand
        return coords[keep], flags[keep], hand[keep]

    def _speed_perturb(self, coords, flags, hand):
        T = coords.shape[0]
        if T < 2 * self.min_frames:
            return coords, flags, hand
        factor = np.random.uniform(*self.speed_perturb_range)
        new_T = max(self.min_frames, int(round(T * factor)))
        idx = np.linspace(0, T - 1, new_T).round().astype(int)
        # Dùng nearest-neighbour cho flags/hand để giữ giá trị rời rạc nguyên vẹn
        return coords[idx], flags[idx], hand[idx]

class AugmentedSkeletonDataset:

    def __init__(self, base_dataset, augmentor):
        self.base_dataset = base_dataset
        self.augmentor = augmentor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature, hand_normalized_feature, label = self.base_dataset[idx]
        feature, hand_normalized_feature = self.augmentor(feature, hand_normalized_feature)
        return feature, hand_normalized_feature, label