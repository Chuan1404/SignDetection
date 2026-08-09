import numpy as np
from config import _REMOVE_POSE_IDX

_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12

_EPS = 1e-6

class FusionComponent:

    def __init__(self):
        pass

    def fuse(self, pose_feature, hand_feature):

        pose = np.asarray(pose_feature, dtype=np.float32) # (T, 99)
        hand = np.asarray(hand_feature, dtype=np.float32) # (T, 126)
        # lips = np.asarray(lips_feature, dtype=np.float32) # (T, 120)

        T = pose.shape[0]

        left_hand = hand[:, :63].reshape(T, 21, 3).copy()
        right_hand = hand[:, 63:].reshape(T, 21, 3).copy()
        pose = pose.reshape(T, 33, 3).copy()
        # lips = lips.reshape(T, 40, 3).copy()

        left_missing  = np.all(left_hand  == 0, axis=(1, 2))
        right_missing = np.all(right_hand == 0, axis=(1, 2))

        left_present  = (~left_missing).astype(np.float32)
        right_present = (~right_missing).astype(np.float32)

        root = (pose[:, _LEFT_SHOULDER_IDX] + pose[:, _RIGHT_SHOULDER_IDX]) / 2.0

        scale = np.linalg.norm(
            pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX],
            axis=-1,
            keepdims=True
        )  # (T, 1)

        scale = np.where(scale > _EPS, scale, 1.0)

        # Reshape for broadcasting
        root = root[:, np.newaxis, :]  # (T, 1, 3)
        scale = scale[:, np.newaxis, :]  # (T, 1, 1)

        pose = (pose - root) / scale
        # lips = (lips - root) / scale
        left_hand = (left_hand - root) / scale
        right_hand = (right_hand - root) / scale

        left_hand[left_missing] = 0
        right_hand[right_missing] = 0

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        fused_coords = np.concatenate([pose, left_hand, right_hand], axis=1)
        fused_flat = fused_coords.reshape(T, -1)  # (T, D)

        hand_flags = np.stack([left_present, right_present], axis=1)  # (T, 2)
        fused_flat = np.concatenate([fused_flat, hand_flags], axis=1)

        return fused_flat