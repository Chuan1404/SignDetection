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
from src.utils.vocabulary import Vocabulary


device = "cuda" if torch.cuda.is_available() else "cpu"

csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
df = pd.read_csv(csv_path, sep="\t")

hand_detection = HandDetection()
vocab = Vocabulary()

for s in tqdm(df["SENTENCE"].tolist()):
    vocab.build_vocab(s.lower().split())


encoder = FrameEncoder().to(device).eval()



def preprocess_clip(clip):
    """
    clip: list of frames (T, H, W, C)
    return: tensor (1, T, C, H, W)
    """
    clip = np.stack(clip)  # (T, H, W, C)

    clip = torch.tensor(clip).float() / 255.0
    clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)

    return clip.unsqueeze(0)  # (1, T, C, H, W)

CLIP_SIZE = 16
STRIDE = 8


for idx, row in tqdm(df.iterrows(), total=len(df)):

    video_name = row["SENTENCE_NAME"]
    sentence = row["SENTENCE"]

    video_path = os.path.join(HOW2SIGN_RAW_DATA, f"{video_name}.mp4")

    if not os.path.exists(video_path):
        print("Missing:", video_path)
        continue

    save_dir = os.path.join(ROOT, "datasets", "processed", "i3d_features", video_name)
    os.makedirs(save_dir, exist_ok=True)

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

        detect_results = hand_detection.detect_video(frame_rgb)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        hand_crops = hand_detection.extract_hand_regions(
            frame_bgr,
            detect_results,
            15
        )

        left = hand_crops.get("Left", None)
        right = hand_crops.get("Right", None)

        if left is None or right is None:
            continue

        left_buffer.append(left)
        right_buffer.append(right)

        if len(left_buffer) == CLIP_SIZE:

            left_clip = preprocess_clip(left_buffer).to(device)
            right_clip = preprocess_clip(right_buffer).to(device)

            with torch.no_grad():
                left_feat = encoder(left_clip).cpu().numpy()
                right_feat = encoder(right_clip).cpu().numpy()

            left_features.append(left_feat)
            right_features.append(right_feat)

            # sliding window
            left_buffer = left_buffer[STRIDE:]
            right_buffer = right_buffer[STRIDE:]

    cap.release()

    if len(left_features) > 0:
        left_features = np.concatenate(left_features, axis=0)
        right_features = np.concatenate(right_features, axis=0)
    else:
        left_features = np.array([])
        right_features = np.array([])

    np.save(os.path.join(save_dir, "left_feat.npy"), left_features)
    np.save(os.path.join(save_dir, "right_feat.npy"), right_features)

    text_ids = vocab.encode(sentence)

    torch.save(
        text_ids,
        os.path.join(save_dir, "text_ids.pt")
    )

    print(f"\tSaved: {video_name}")

hand_detection.close()

print("FINISH")