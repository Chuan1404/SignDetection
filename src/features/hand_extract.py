import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import HOW2SIGN_RAW_DATA, ROOT
from src.utils.hand_detection import HandDetection

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "mediapipe")
os.makedirs(SAVE_DIR, exist_ok=True)

csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
df = pd.read_csv(csv_path, sep="\t")

for _, row in tqdm(df.iterrows(), total=len(df)):
    hand_detection = HandDetection()
    video_name = row["SENTENCE_NAME"]
    sentence = row["SENTENCE"]

    video_path = os.path.join(HOW2SIGN_RAW_DATA, f"{video_name}.mp4")

    if not os.path.exists(video_path):
        print("Missing:", video_path)
        continue

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    frame_index = 0
    hand_features = []

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(frame_index * 1000 / fps)

        detection_results = hand_detection.detect_video(
            frame,
            timestamp_ms
        )

        handedness = detection_results.handedness
        hand_landmarks = detection_results.hand_landmarks

        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)

        if hand_landmarks is not None and len(hand_landmarks) > 0:

            for i, hand_info in enumerate(handedness):

                if i >= len(hand_landmarks):
                    continue

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

        frame_feature = np.concatenate([
            right_hand.flatten(),
            left_hand.flatten()
        ])

        frame_feature = np.nan_to_num(
            frame_feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        hand_features.append(frame_feature)

        frame_index += 1

    cap.release()
    hand_detection.close()

    if len(hand_features) == 0:
        print(f"Skip empty video: {video_name}")
        continue

    hand_features = np.array(hand_features, dtype=np.float32)

    # final safety check
    hand_features = np.nan_to_num(hand_features)

    dir_name = os.path.join(SAVE_DIR, video_name)
    os.makedirs(dir_name, exist_ok=True)
    np.save(
        os.path.join(dir_name, "hand_features.npy"),
        hand_features
    )

    with open(
        os.path.join(dir_name, "text.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(sentence.lower().strip())

    print(f"Saved: {video_name}")



print("FINISH")