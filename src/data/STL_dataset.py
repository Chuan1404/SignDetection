import os
from torch.utils.data import Dataset
import numpy as np
import torch

# class SLTDataset(Dataset):
#
#     def __init__(self, feature_dir, vocabulary):
#         self.vocabulary = vocabulary
#         self.feature_dir = feature_dir
#
#         self.samples = []
#
#         for video_name in os.listdir(feature_dir):
#
#             video_dir = os.path.join(
#                 feature_dir,
#                 video_name
#             )
#
#             if not os.path.isdir(video_dir):
#                 continue
#
#             left_path = os.path.join(video_dir,"left_feat.npy")
#
#             right_path = os.path.join(video_dir, "right_feat.npy")
#
#             text_path = os.path.join(video_dir,"text.txt")
#
#             if (
#                 os.path.exists(left_path)
#                 and os.path.exists(right_path)
#                 and os.path.exists(text_path)
#             ):
#
#                 self.samples.append({
#                     "left": left_path,
#                     "right": right_path,
#                     "text": text_path
#                 })
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#
#         item = self.samples[idx]
#
#         left = np.load(item["left"])
#         right = np.load(item["right"])
#
#         if left.size == 0 or right.size == 0:
#             return self.__getitem__((idx + 1) % len(self))
#
#         with open(item["text"], "r", encoding="utf-8") as f:
#             sentence = f.read().strip()
#
#         text_ids = self.vocabulary.encode(sentence)
#
#         left = torch.tensor(left).float()
#         right = torch.tensor(right).float()
#
#         text = torch.tensor(text_ids).long()
#
#         return left, right, text
#

class SLTDataset(Dataset):

    def __init__(self, feature_dir, vocabulary):
        self.vocabulary = vocabulary
        self.feature_dir = feature_dir

        self.samples = []

        for video_name in os.listdir(feature_dir):

            video_dir = os.path.join(
                feature_dir,
                video_name
            )

            if not os.path.isdir(video_dir):
                continue


            video_path = os.path.join(video_dir, "video_feat.npy")

            text_path = os.path.join(video_dir,"text.txt")

            if (
                os.path.exists(video_path)
                and os.path.exists(text_path)
            ):

                self.samples.append({
                    "video_feat": video_path,
                    "text": text_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        video = np.load(item["video_feat"])

        if video.size == 0:
            return self.__getitem__((idx + 1) % len(self))

        with open(item["text"], "r", encoding="utf-8") as f:
            sentence = f.read().strip()

        tokens = self.vocabulary(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        video = torch.tensor(video).float()

        text = tokens["input_ids"].squeeze(0)

        return video, text