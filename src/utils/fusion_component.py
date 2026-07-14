import numpy as np

# Pose landmark indices removed after merging wrists with hand landmarks.
# 17,19,21 = left hand pinky/index/thumb tips (redundant with detailed hand landmarks)
# 18,20,22 = right hand pinky/index/thumb tips (redundant with detailed hand landmarks)
# 25-31   = leg/foot points (not needed for upper-body / sign-language tasks)
_REMOVE_POSE_IDX = [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]

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
        """
        Fuses pose and hand features over all frames.
        
        Args:
            pose_feature: Array-like of shape (T, 33, 3) or (T, 99)
            hand_feature: Array-like of shape (T, 126)
            
        Returns:
            fused_features: NumPy array of shape (T, 185) where the last 2 columns
                            are [left_present, right_present] flags.
        """
        pose = np.asarray(pose_feature, dtype=np.float32)
        hand = np.asarray(hand_feature, dtype=np.float32)
        
        T = pose.shape[0]
        
        # Reshape to 3D coordinate tensors
        left_hand = hand[:, :63].reshape(T, 21, 3).copy()
        right_hand = hand[:, 63:].reshape(T, 21, 3).copy()
        pose = pose.reshape(T, 33, 3).copy()

        # 1. Detect hand presence before normalization (any non-zero coordinate)
        has_left = np.any(left_hand != 0.0, axis=(1, 2))   # (T,)
        has_right = np.any(right_hand != 0.0, axis=(1, 2)) # (T,)

        left_present = has_left.astype(np.float32).reshape(T, 1)
        right_present = has_right.astype(np.float32).reshape(T, 1)

        # 2. Normalize pose and hands coordinates relative to hip root and shoulder scale
        root = (pose[:, _LEFT_HIP_IDX] + pose[:, _RIGHT_HIP_IDX]) / 2.0  # (T, 3)
        scale = np.linalg.norm(
            pose[:, _LEFT_SHOULDER_IDX] - pose[:, _RIGHT_SHOULDER_IDX],
            axis=-1,
            keepdims=True
        )  # (T, 1)
        scale = np.where(scale > _EPS, scale, 1.0)

        # Reshape for broadcasting
        root = root[:, np.newaxis, :]    # (T, 1, 3)
        scale = scale[:, np.newaxis, :]  # (T, 1, 1)

        pose = (pose - root) / scale
        left_hand = (left_hand - root) / scale
        right_hand = (right_hand - root) / scale

        # 3. If a hand is missing, force all its coordinates to exactly 0.0 (pelvis root)
        left_hand = np.where(has_left[:, np.newaxis, np.newaxis], left_hand, 0.0)
        right_hand = np.where(has_right[:, np.newaxis, np.newaxis], right_hand, 0.0)

        # 4. Merge wrist coordinate from hand detection into pose wrists
        pose[:, _RIGHT_WRIST_IDX] = right_hand[:, 0]
        pose[:, _LEFT_WRIST_IDX] = left_hand[:, 0]

        # 5. Remove redundant/unused landmarks from pose
        pose = np.delete(pose, _REMOVE_POSE_IDX, axis=1)  # axis=1 represents landmarks

        # 6. Concatenate along landmarks dimension
        fused_coords = np.concatenate([pose, left_hand, right_hand], axis=1) # (T, 61, 3)
        fused_flat = fused_coords.reshape(T, -1) # (T, 183)

        # 7. Append the global hand presence flags
        fused_final = np.concatenate([fused_flat, left_present, right_present], axis=1) # (T, 185)

        return fused_final