import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import ROOT, HOW2SIGN_RAW_DATA
from transformers import VideoMAEModel, AutoImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
df = pd.read_csv(csv_path, sep="\t")
df = df.sample(frac=0.1)

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "videomae_features")
os.makedirs(SAVE_DIR, exist_ok=True)

model = VideoMAEModel.from_pretrained(
    "MCG-NJU/videomae-base"
).to(device).eval()

processor = AutoImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base"
)

# =========================
# CONFIG IMPORTANT
# =========================
CLIP_SIZE = 16
STRIDE = 8
MAX_T = 128

# =========================
# PREPROCESS
# =========================
def preprocess(frames):
    inputs = processor(frames, return_tensors="pt")
    return inputs["pixel_values"]

# =========================
# FRAME SAMPLER (LIMIT LENGTH)
# =========================
def sample_video_frames(cap, max_frames=2000):
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

        count += 1
        if count >= max_frames:
            break

    return frames

# =========================
# MAIN LOOP
# =========================
for idx, row in tqdm(df.iterrows(), total=len(df)):

    video_name = row["SENTENCE_NAME"]
    sentence = row["SENTENCE"].lower().strip()

    video_path = os.path.join(HOW2SIGN_RAW_DATA, f"{video_name}.mp4")

    if not os.path.exists(video_path):
        print("Missing:", video_path)
        continue

    cap = cv2.VideoCapture(video_path)

    frames = sample_video_frames(cap, max_frames=3000)
    cap.release()

    if len(frames) < CLIP_SIZE:
        print("Too short:", video_name)
        continue

    clip_features = []

    i = 0
    while i + CLIP_SIZE <= len(frames):

        clip = frames[i:i + CLIP_SIZE]
        pixel_values = preprocess(clip).to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            feat = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        clip_features.append(feat)

        i += STRIDE

    if len(clip_features) == 0:
        print(f"Skipping empty video: {video_name}")
        continue

    # =========================
    # FIX SEQUENCE LENGTH
    # =========================
    clip_features = np.concatenate(clip_features, axis=0)

    T = clip_features.shape[0]

    if T > MAX_T:
        idxs = np.linspace(0, T - 1, MAX_T).astype(int)
        clip_features = clip_features[idxs]

    # =========================
    # SAVE
    # =========================
    save_dir = os.path.join(SAVE_DIR, video_name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "video_feat.npy"), clip_features)

    with open(os.path.join(save_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(sentence)

    print(f"Saved: {video_name}, shape={clip_features.shape}")

print("FINISH")