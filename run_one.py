import os
import numpy as np
from tqdm import tqdm

from config import ROOT

SAVE_DIR = os.path.join(
    ROOT,
    "datasets",
    "processed",
    "i3d_features"
)

removed = 0
kept = 0

for video_name in tqdm(os.listdir(SAVE_DIR)):

    video_dir = os.path.join(SAVE_DIR, video_name)

    if not os.path.isdir(video_dir):
        continue

    left_path = os.path.join(video_dir, "left_feat.npy")
    right_path = os.path.join(video_dir, "right_feat.npy")

    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        continue

    try:
        left = np.load(left_path)
        right = np.load(right_path)

        # -----------------------------
        # REMOVE EMPTY
        # -----------------------------
        if left.size == 0 or right.size == 0:
            print(f"❌ REMOVE EMPTY: {video_name}")
            removed += 1

            os.remove(left_path)
            os.remove(right_path)
            continue

        # -----------------------------
        # REMOVE WRONG SHAPE
        # -----------------------------
        if left.ndim != 2 or right.ndim != 2:
            print(f"❌ REMOVE NDIM: {video_name}")
            removed += 1

            os.remove(left_path)
            os.remove(right_path)
            continue

        if left.shape[-1] != 512 or right.shape[-1] != 512:
            print(f"❌ REMOVE DIM ERROR: {video_name}")
            removed += 1

            os.remove(left_path)
            os.remove(right_path)
            continue

        # -----------------------------
        # KEEP VALID
        # -----------------------------
        kept += 1

        print(f"✔ KEEP: {video_name} | {left.shape}")

    except Exception as e:
        print(f"❌ ERROR: {video_name} -> {e}")
        removed += 1

        try:
            os.remove(left_path)
            os.remove(right_path)
        except:
            pass

print("\n====================")
print("CLEANING DONE")
print("====================")
print("KEPT   :", kept)
print("REMOVED:", removed)