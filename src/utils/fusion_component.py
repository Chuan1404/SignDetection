import cv2
import numpy as np

class ExtractingFusionComponent:
    def __init__(self):
        pass

    # def _get_root(self, pose):
    #     # hip center as root
    #     left_hip = pose[23]
    #     right_hip = pose[24]
    #     return (left_hip + right_hip) / 2

    def fuse(self, pose_results, hand_results):
        handedness = hand_results.handedness
        hand_landmarks = hand_results.hand_landmarks

        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)

        if hand_landmarks is not None and len(hand_landmarks) > 0:

            for i, hand_info in enumerate(handedness):
                category = hand_info[0]

                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks[i]],
                    dtype=np.float32
                )

                coords = np.nan_to_num(coords)

                if category.index == 0:
                    right_hand = coords

                elif category.index == 1:
                    left_hand = coords

        # remove 18th, 20th, 22th pose points, merge 16th point with 0th right hand point
        # remove 17th, 19th, 21th pose points, merge 15th point with 0th point of left hand
        # remove unneeded points (25-31)

        pose_landmarks = pose_results.pose_landmarks[0]
        pose_coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks],
            dtype=np.float32
        )

        pose_coords = np.nan_to_num(pose_coords)
        # root = self._get_root(pose_coords)
        # pose_coords = pose_coords - root
        # left_hand = left_hand - root
        # right_hand = right_hand - root

        # Replace wrist by hand wrist
        if np.any(right_hand):
            pose_coords[16] = right_hand[0]  # right wrist

        if np.any(left_hand):
            pose_coords[15] = left_hand[0]  # left wrist

        # Remove redundant hand points in pose
        remove_idx = [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31]

        pose_coords = np.delete(
            pose_coords,
            remove_idx,
            axis=0
        )

        # Final landmarks
        fused_landmarks = np.concatenate(
            [
                pose_coords,
                left_hand,
                right_hand
            ],
            axis=0
        )

        return fused_landmarks

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        annotated = rgb_image.copy()

        h, w = annotated.shape[:2]

        for x, y, z in detection_result:
            px = int(x * w)
            py = int(y * h)

            cv2.circle(
                annotated,
                (px, py),
                radius=2,
                color=(0, 255, 0),
                thickness=-1
            )

        return annotated


class FusionComponent:
    def __init__(self):
        pass

    def fuse(self, pose_feature, hand_feature):

        T, _ = pose_feature.shape
        fused_features = []

        for i in range(T):
            left_hand = hand_feature[i][:63].reshape(21, 3)
            right_hand = hand_feature[i][63:].reshape(21, 3)
            pose = pose_feature[i].reshape(33, 3)

            # remove 18th, 20th, 22th pose points, merge 16th point with 0th right hand point
            # remove 17th, 19th, 21th pose points, merge 15th point with 0th point of left hand
            # remove unneeded points (25-31)

            # Replace wrist by hand wrist
            pose[16] = right_hand[0]  # right wrist

            pose[15] = left_hand[0]  # left wrist

            # Remove redundant hand points in pose
            remove_idx = [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31]

            pose = np.delete(
                pose,
                remove_idx,
                axis=0
            )

            fused_feature = np.concatenate(
                [
                    pose, # 20
                    left_hand, # 21
                    right_hand # 21
                ],
                axis=0
            )

            fused_features.append(fused_feature)

        fused_features = np.array(fused_features)
        return fused_features