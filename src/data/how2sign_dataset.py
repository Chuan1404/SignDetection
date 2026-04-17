import pandas as pd
import torch
import numpy as np
import cv2
from sympy.codegen.ast import float16
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader

class How2SignDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.df = pd.read_csv(csv_file, sep="\t")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        i3d = np.load(row["i3d_path"])      # (T, D1)
        mp  = np.load(row["mp_path"])       # (T, D2)

        text = self.tokenizer.encode(row["text"])

        return (
            torch.tensor(i3d, dtype=torch.float32),
            torch.tensor(mp, dtype=torch.float32),
            torch.tensor(text, dtype=torch.long)
        )

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     # VIDEO_ID - cmG4MzqyjE
    #     # VIDEO_NAME - cmG4MzqyjE - 5 - rgb_front
    #     # SENTENCE_ID - cmG4MzqyjE_4
    #     # SENTENCE_NAME - cmG4MzqyjE_4 - 5 - rgb_front
    #     # START 20.44
    #     # END25.48
    #     # SENTENCE I want you
    #     # Name: 1672, dtype: object
    #     # Output shape: tensor([22514, 1672])
    #
    #     video_id = row["VIDEO_ID"]
    #     video_name = row["VIDEO_NAME"]
    #     sentence = row["SENTENCE"]
    #
    #     video_path = r"D:\SignDetection\datasets\raw\how2sign_raw\0-0kX3XoMPQ_0-3-rgb_front.mp4"
    #     cap = cv2.VideoCapture(video_path) # 149 frames
    #
    #     success, frame = cap.read()
    #     video_frames = []
    #     while success:
    #         frame = cv2.resize(frame, (224, 224))
    #         video_frames.append(frame)
    #         success, frame = cap.read()
    #
    #     cap.release()
    #
    #     video_frames = np.array(video_frames, dtype="float32")
    #     # video_frames = torch.from_numpy(video_frames)
    #     # video_frames = torch.permute(video_frames, (0, 3, 1, 2))
    #
    #     return video_frames, sentence