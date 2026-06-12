import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.utils.pose_detection import PoseDetection

from src.utils.face_detection import FaceDetection
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import HOW2SIGN_RAW_DATA, ROOT
from src.utils.hand_detection import HandDetection

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "mediapipe", "face")
os.makedirs(SAVE_DIR, exist_ok=True)

csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
df = pd.read_csv(csv_path, sep="\t")


# cap = cv2.VideoCapture(os.path.join(r"D:\archive\videos", "00335.mp4"))
# cap = cv2.VideoCapture(os.path.join(ROOT, "datasets", "processed", "how2gisn_resized", "_2u0MkRqpjA_3-5-rgb_front.mp4"))
cap = cv2.VideoCapture(os.path.join(HOW2SIGN_RAW_DATA, "_2u0MkRqpjA_3-5-rgb_front.mp4"))

face_detection = FaceDetection()
hand_detection = HandDetection()
pose_detection = PoseDetection()

frame_index = 0
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 25.0

while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp_ms = int(frame_index * 1000 / fps)

    detection_face_results = face_detection.detect_video(frame, timestamp_ms)
    detection_hand_results = hand_detection.detect_video(frame, timestamp_ms)
    detection_pose_results = pose_detection.detect_video(frame, timestamp_ms)

    frame = face_detection.draw_landmarks_on_image(frame, detection_face_results)
    frame = hand_detection.draw_landmarks_on_image(frame, detection_hand_results)
    frame = pose_detection.draw_landmarks_on_image(frame, detection_pose_results)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("test", frame)
    frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()
