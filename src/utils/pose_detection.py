import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

mp_drawing_styles = vision.drawing_styles
mp_drawing_utils = vision.drawing_utils

class PoseDetection:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=r'/pretrained/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)

    def detect_pose(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.pose_detector.detect(mp_image)

        return detection_result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = mp_drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_list:
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)

        return annotated_image
