import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import cv2

from src.data import how2sign_dataset
from src.utils.hand_detection import HandDetection

how2sign_dataset = how2sign_dataset.How2SignDataset(r"/datasets\annotations\how2sign_train.csv")
loader = DataLoader(how2sign_dataset)

hand_detection = HandDetection()

# load video -> media pipe -> 21 NormalizedLandmark points
for video_frames, sentence in loader:
    video_frames = video_frames.squeeze()

    for frame in video_frames:
        np_frame = frame.numpy().astype(np.uint8)
        rgb_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)

        results = hand_detection.detect_video(rgb_frame)
        print(results)
    break

hand_detection.close()
