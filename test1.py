import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from functools import partial

from src.data.hand_landmarks import HandLandmarksDataset
from src.models.SLT_model import SignLanguageTranslator, SignLanguageTranslatorV1
from config import ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "mediapipe")

def collate_fn(batch, tokenizer):

    features, texts = [], []

    for feature, text in batch:
        features.append(feature)
        texts.append(text)

    real_lengths = [f.shape[0] for f in features]

    features = pad_sequence(features, batch_first=True)

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    labels = texts.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return features, labels, video_mask


tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False
)

dataset = HandLandmarksDataset(FEATURE_DIR, tokenizer)

train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)

train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tokenizer),
    num_workers=2,
    pin_memory=True
)

