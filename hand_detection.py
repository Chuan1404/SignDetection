import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import cv2

mp_hands = vision.HandLandmarkerOptions
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


class HandDetection():
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='./pretrained/hand_landmarker.task')

        # Video
        video_options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO)
        video_detector = vision.HandLandmarker.create_from_options(video_options)

        # Image
        image_options = vision.HandLandmarkerOptions(
            base_options=base_options)
        image_detector = vision.HandLandmarker.create_from_options(image_options)

        self.video_detector = video_detector
        self.image_detector = image_detector

    def detect_image(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = self.image_detector.detect(mp_image)

        return detection_result

    def detect_video(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = self.video_detector.detect_for_video(mp_image, int(time.time() * 1000))

        return detection_result

    def draw_landmarks_on_image(self, rgb_image, detection_result: vision.HandLandmarkerResult, predicted_label=None):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                None,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                # mp_drawing_styles
            )

            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = 50
            text_y = 50

            # Draw handedness (left or right hand) on the image.
            if predicted_label is not None:
                cv2.putText(annotated_image, f"Letter: {predicted_label}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
