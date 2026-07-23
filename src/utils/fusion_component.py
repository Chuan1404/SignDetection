import numpy as np

_REMOVE_POSE_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]

_LEFT_WRIST_IDX = 15
_RIGHT_WRIST_IDX = 16
_LEFT_HIP_IDX = 23
_RIGHT_HIP_IDX = 24
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

        left_hand = hand[:, :63].reshape(T, 21, 3).copy()
        right_hand = hand[:, 63:].reshape(T, 21, 3).copy()
        pose = pose.reshape(T, 33, 3).copy()

        has_left = np.any(left_hand != 0.0, axis=(1, 2))  # (T,)
        has_right = np.any(right_hand != 0.0, axis=(1, 2))  # (T,)

        left_present = has_left.astype(np.float32).reshape(T, 1)
        right_present = has_right.astype(np.float32).reshape(T, 1)

        root = (pose[:, _LEFT_HIP_IDX] + pose[:, _RIGHT_HIP_IDX]) / 2.0  # (T, 3)

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
        left_hand = (left_hand - root) / scale
        right_hand = (right_hand - root) / scale

        left_hand = np.where(has_left[:, np.newaxis, np.newaxis], left_hand, 0.0)
        right_hand = np.where(has_right[:, np.newaxis, np.newaxis], right_hand, 0.0)

        pose[:, _RIGHT_WRIST_IDX] = right_hand[:, 0]
        pose[:, _LEFT_WRIST_IDX] = left_hand[:, 0]

        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)  # axis=1 represents landmarks
        left_hand = left_hand[:, 1:]
        right_hand = right_hand[:, 1:]

        fused_coords = np.concatenate([pose, left_hand, right_hand], axis=1)  # (T, 59, 3)
        fused_flat = fused_coords.reshape(T, -1)  # (T, 177)

        fused_final = np.concatenate([fused_flat, left_present, right_present], axis=1)  # (T, 179)

        return fused_final