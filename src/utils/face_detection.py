import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import time

mp_drawing_styles = vision.drawing_styles
mp_drawing_utils = vision.drawing_utils

class FaceDetection:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=r'/pretrained/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
            )
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

    def detect_face(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.face_detector.detect(mp_image)
        print(detection_result)
        return detection_result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.

            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        return annotated_image

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in
                                  face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()