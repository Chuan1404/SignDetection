from turtle import width

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

FINGERS = {
    "Thumb": [0, 1, 2, 3, 4],
    "Index": [0, 5, 6, 7, 8],
    "Middle": [0, 9, 10, 11, 12],
    "Ring": [0, 13, 14, 15, 16],
    "Pinky": [0, 17, 18, 19, 20]
}

# Assign colors (BGR) to each finger
FINGER_COLORS = {
    "Thumb": (0, 0, 255),   # Red
    "Index": (0, 255, 0),   # Green
    "Middle": (255, 0, 0),  # Blue
    "Ring": (0, 255, 255),  # Yellow
    "Pinky": (255, 0, 255)  # Magenta
}
class HandDetection:
    def __init__(self):
        self.prev_hand_crops = hand_crops = {
            "Left": None,
            "Right": None
        }

        base_options = python.BaseOptions(model_asset_path='../../pretrained/hand_landmarker.task')

        # Video
        video_options = vision.HandLandmarkerOptions(
            base_options=base_options,
            min_hand_detection_confidence = 0.3,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO)
        self.video_detector = vision.HandLandmarker.create_from_options(video_options)

        # Image
        image_options = vision.HandLandmarkerOptions(
            base_options=base_options)
        self.image_detector = vision.HandLandmarker.create_from_options(image_options)

    def detect_image(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.image_detector.detect(mp_image)

        return detection_result

    def detect_video(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = self.video_detector.detect_for_video(mp_image, int(time.time() * 1000))

        return detection_result

    def extract_hand_regions(self, frame, detection_result, padding=20):

        hand_crops = self.prev_hand_crops.copy()

        height, width, _ = frame.shape

        if not detection_result.hand_landmarks:
            return hand_crops

        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):

            hand_label = detection_result.handedness[idx][0].category_name

            x_list = []
            y_list = []

            for lm in hand_landmarks:
                x_list.append(int(lm.x * width))
                y_list.append(int(lm.y * height))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            box_size = max(x_max - x_min, y_max - y_min) + padding
            half = box_size // 2

            x1 = max(cx - half, 0)
            y1 = max(cy - half, 0)
            x2 = min(cx + half, width)
            y2 = min(cy + half, height)

            hand_crop = frame[y1:y2, x1:x2]

            hand_crop = cv2.resize(hand_crop, (224, 224))
            hand_crops[hand_label] = hand_crop

        self.prev_hand_crops = hand_crops.copy()

        return hand_crops

    def draw_landmarks_on_image(self, rgb_image, detection_result, predicted_label=None):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape

        # Scale parameters with image size
        point_radius = max(2, int(min(width, height) * 0.01))
        line_thickness = max(1, int(min(width, height) * 0.002))

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Convert normalized landmarks to pixel coordinates
            landmark_points = [
                (int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks
            ]

            # Draw connections per finger with assigned colors
            for finger_name, indices in FINGERS.items():
                color = FINGER_COLORS[finger_name]
                for i in range(len(indices)-1):
                    start_idx = indices[i]
                    end_idx = indices[i+1]
                    cv2.line(
                        annotated_image,
                        landmark_points[start_idx],
                        landmark_points[end_idx],
                        color,
                        line_thickness
                    )

            # Draw all landmarks as small circles
            for x, y in landmark_points:
                cv2.circle(annotated_image, (x, y), point_radius, (0, 255, 0), -1)

            # Draw handedness text
            x_coordinates = [lm.x for lm in hand_landmarks]
            y_coordinates = [lm.y for lm in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN
            # cv2.putText(
            #     annotated_image,
            #     # f"{handedness[0].category_name}",
            #     f"{handedness[0].category_name}",
            #     (text_x, text_y),
            #     cv2.FONT_HERSHEY_DUPLEX,
            #     FONT_SIZE,
            #     HANDEDNESS_TEXT_COLOR,
            #     FONT_THICKNESS,
            #     cv2.LINE_AA
            # )

            # Optional: predicted label
            if predicted_label is not None:
                cv2.putText(
                    annotated_image,
                    f"Letter: {predicted_label}",
                    (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE,
                    (0, 0, 255),
                    line_thickness,
                    cv2.LINE_AA
                )

        return annotated_image

    def close(self):
        # When done (e.g., on cleanup)
        if self.video_detector:
            self.video_detector.close()
        if self.image_detector:
            self.image_detector.close()

