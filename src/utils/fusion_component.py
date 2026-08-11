import numpy as np
from config import _REMOVE_POSE_IDX

_NOSE_IDX = 0
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12

_EPS = 1e-6

class FusionComponent:

    def __init__(self):
        pass

    def fuse(self, pose_feature, hand_feature):

        pose = np.asarray(pose_feature, dtype=np.float32) # (T, 99)
        hand = np.asarray(hand_feature, dtype=np.float32) # (T, 126)

        T = pose.shape[0]

        left_hand = hand[:, :63].reshape(T, 21, 3)[:, :, :2].copy()
        right_hand = hand[:, 63:].reshape(T, 21, 3)[:, :, :2].copy()
        pose = pose.reshape(T, 33, 3)[:, :, :2].copy()


        root = pose[:, _NOSE_IDX]

        scale = np.linalg.norm(
            pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX],
            axis=-1,
            keepdims=True
        )  # (T, 1)

        scale = np.where(scale > _EPS, scale, 1.0)

        root = root[:, np.newaxis, :]  # (T, 1, 3)
        scale = scale[:, np.newaxis, :]  # (T, 1, 1)

        pose = (pose - root) / scale
        left_hand = (left_hand - root) / scale
        right_hand = (right_hand - root) / scale

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        fused_coords = np.concatenate([pose, left_hand, right_hand], axis=1)
        fused_flat = fused_coords.reshape(T, -1)  # (T, D)

        return fused_flat