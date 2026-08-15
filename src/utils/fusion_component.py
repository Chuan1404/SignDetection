import numpy as np

from config import _REMOVE_POSE_IDX

_NOSE_IDX = 0
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12

_EPS = 1e-6

class FusionComponent:

    def __init__(self):
        pass

    def fuse(self, pose_feature, left_feature, right_feature):

        T = pose_feature.shape[0]

        pose = pose_feature.reshape(T, 33, 3)[:, :, :2].copy()
        left = left_feature.reshape(T, 21, 3)[:, :, :2].copy()
        right = right_feature.reshape(T, 21, 3)[:, :, :2].copy()

        all_points = np.concatenate([pose, left, right],axis=1)  # (T, N, 2)

        left_present_mask = ~np.all(left == 0, axis=-1)
        right_present_mask = ~np.all(right == 0, axis=-1)
        pose_present_mask = ~np.all(pose == 0, axis=-1)

        all_present_mask = np.concatenate([pose_present_mask, left_present_mask, right_present_mask], axis=1) # (T, N)

        valid_points = (all_points * all_present_mask[..., None])

        valid_count = all_present_mask.sum(axis=1,keepdims=True)

        average_point = (valid_points.sum(axis=1)/ np.maximum(valid_count, 1))

        average_point = average_point.reshape(T, 1, 2)

        scale = np.linalg.norm(
            pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX],
            axis=-1,
            keepdims=True
        )

        scale = np.where(scale > _EPS, scale, 1.0)

        scale = scale[:, np.newaxis, :]
        root = pose[:, np.newaxis, _NOSE_IDX]

        pose = (pose - root) / scale
        left = (left - root) / scale
        right = (right - root) / scale

        left = left * left_present_mask[..., None]
        right = right * right_present_mask[..., None]
        pose = pose * pose_present_mask[..., None]

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)

        fused_coords = np.concatenate([pose, left, right], axis=1)
        fused_flat = fused_coords.reshape(T, -1)  # (T, D)

        return fused_flat