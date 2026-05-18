import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import ROOT, HOW2SIGN_RAW_DATA
from src.models.encoder import FrameEncoder
from src.utils.hand_detection import HandDetection
from src.utils.vocabulary import Vocabulary

def frame_to_feature(frame):

    frame = cv2.resize(frame, (224, 224))

    frame = torch.tensor(frame).float() / 255.0

    frame = frame.permute(2, 0, 1)

    frame = frame.unsqueeze(0).unsqueeze(0)

    frame = frame.to(device)

    with torch.no_grad():

        feat = encoder(frame)

    feat = feat.squeeze(0).squeeze(0).cpu().numpy()

    del frame

    return feat

device = "cuda" if torch.cuda.is_available() else "cpu"

csv_path = os.path.join(
    ROOT,
    r"datasets\annotations\how2sign_train.csv"
)

df = pd.read_csv(csv_path, sep="\t")

hand_detection = HandDetection()

vocab = Vocabulary()

for s in tqdm(df["SENTENCE"].tolist()):
    vocab.build_vocab(s.lower().split())

# df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
encoder = FrameEncoder().eval()
encoder = encoder.to(device)

for idx, row in tqdm(df.iterrows(), total=len(df)):

    video_name = row["SENTENCE_NAME"]
    sentence = row["SENTENCE"]

    video_path = os.path.join(
        HOW2SIGN_RAW_DATA,
        f"{video_name}.mp4"
    )

    if not os.path.exists(video_path):
        print("Missing:", video_path)
        continue

    save_dir = os.path.join(
        ROOT,
        "datasets",
        "processed",
        "features",
        video_name
    )

    os.makedirs(save_dir, exist_ok=True)

    left_features = []
    right_features = []

    cap = cv2.VideoCapture(video_path)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        detect_results = hand_detection.detect_video(frame_rgb)

        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        hand_crops = hand_detection.extract_hand_regions(
            frame_bgr,
            detect_results,
            15
        )

        # arrange hands
        for i, (name, img) in enumerate(hand_crops.items()):

            if name == "Left" and img is not None:
                left_feat = frame_to_feature(
                    hand_crops["Left"]
                )
                left_features.append(left_feat)

            if name == "Right" and img is not None:
                right_feat = frame_to_feature(
                    hand_crops["Right"]
                )

                right_features.append(right_feat)

    cap.release()

    left_features = np.stack(left_features)

    right_features = np.stack(right_features)

    np.save(
        os.path.join(save_dir, "left_feat.npy"),
        left_features
    )

    np.save(
        os.path.join(save_dir, "right_feat.npy"),
        right_features
    )

    text_ids = vocab.encode(sentence)

    torch.save(
        text_ids,
        os.path.join(save_dir, "text_ids.pt")
    )

    del left_features
    del right_features

    torch.cuda.empty_cache()

    print(f"\tSaved: {video_name}")

hand_detection.close()

print("FINISH")

