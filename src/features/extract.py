import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import ROOT, HOW2SIGN_RAW_DATA
from src.models.encoder import FrameEncoder
from src.utils.hand_detection import HandDetection


device = "cuda" if torch.cuda.is_available() else "cpu"

csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
df = pd.read_csv(csv_path, sep="\t")
df = df.sample(frac=0.1)

hand_detection = HandDetection()

CLIP_SIZE = 16
STRIDE = 4

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "26.05.27")
encoder = FrameEncoder().to(device).eval()

def preprocess_clip(clip):
    """
    clip: list of frames (T, H, W, C)
    return: tensor (1, T, C, H, W)
    """

    # ---------------------------------
    # 1. stack first (IMPORTANT)
    # ---------------------------------
    clip = np.stack(clip).astype(np.float32)  # (T,H,W,C)

    # ---------------------------------
    # 2. convert to tensor
    # ---------------------------------
    clip = torch.from_numpy(clip)

    # ---------------------------------
    # 3. normalize pixel to [0,1]
    # ---------------------------------
    clip = clip / 255.0

    # ---------------------------------
    # 4. permute to (T,C,H,W)
    # ---------------------------------
    clip = clip.permute(0, 3, 1, 2)

    # ---------------------------------
    # 5. ImageNet / Kinetics normalize
    # ---------------------------------
    mean = torch.tensor(
        [0.43216, 0.394666, 0.37645]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.22803, 0.22145, 0.216989]
    ).view(1, 3, 1, 1)

    clip = (clip - mean) / std

    return clip.unsqueeze(0)


for idx, row in tqdm(df.iterrows(), total=len(df)):

    video_name = row["SENTENCE_NAME"]
    sentence = row["SENTENCE"]
    clean_sentence = sentence.lower().strip()

    video_path = os.path.join(HOW2SIGN_RAW_DATA, f"{video_name}.mp4")

    if not os.path.exists(video_path):
        print("Missing:", video_path)
        continue

    current_dir = os.path.join(SAVE_DIR, video_name)
    os.makedirs(current_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    left_buffer = []
    right_buffer = []

    left_features = []
    right_features = []

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp_ms = frame_idx * 33

        detect_results = hand_detection.detect_video(
            frame_rgb,
            timestamp_ms
        )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        hand_crops = hand_detection.extract_hand_regions(
            frame_bgr,
            detect_results,
            15
        )

        left = hand_crops.get("Left", None)
        right = hand_crops.get("Right", None)

        if left is None and right is None:
            continue

        if left is None:
            left = np.zeros(right.shape, dtype=np.uint8)
        elif right is None:
            right = np.zeros(left.shape, dtype=np.uint8)

        left_buffer.append(left)
        right_buffer.append(right)

        if len(left_buffer) == CLIP_SIZE:

            left_clip = preprocess_clip(left_buffer).to(device)
            right_clip = preprocess_clip(right_buffer).to(device)

            with torch.no_grad():
                left_feat = encoder(left_clip).cpu().numpy().squeeze(0)
                right_feat = encoder(right_clip).cpu().numpy().squeeze(0)

            left_features.append(left_feat)
            right_features.append(right_feat)

            # sliding window
            left_buffer = left_buffer[STRIDE:]
            right_buffer = right_buffer[STRIDE:]

    cap.release()

    if len(left_features) == 0:
        print(f"Skipping empty video: {video_name}")
        continue

    left_features = np.concatenate(left_features, axis=0)
    right_features = np.concatenate(right_features, axis=0)

    np.save(os.path.join(current_dir, "left_feat.npy"), left_features)
    np.save(os.path.join(current_dir, "right_feat.npy"), right_features)

    with open(os.path.join(current_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(clean_sentence)

    print(f"\tSaved: {video_name}")
    hand_detection.close()



print("FINISH")