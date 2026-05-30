import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from transformers import AutoTokenizer

import torch
import pandas as pd

from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import ROOT
from pretrained_model import SLTModel
from src.data.STL_dataset import SLTDataset


DATA_PATH = rf"{ROOT}\datasets\processed\videomae_features"
CSV_PATH = rf"{ROOT}\datasets\annotations\how2sign_train.csv"
SAVE_DIR = rf"{ROOT}\models"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4

import torch

tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False
)

PAD_IDX = tokenizer.pad_token_id

def collate_fn(batch):

    pixels = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    pixel_lengths = [p.size(0) for p in pixels]

    pixels = pad_sequence(
        pixels,
        batch_first=True
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=PAD_IDX
    )

    attention_mask = torch.zeros(
        pixels.size(0),
        pixels.size(1),
        dtype=torch.long
    )

    for i, length in enumerate(pixel_lengths):
        attention_mask[i, :length] = 1

    return pixels, labels, attention_mask


df = pd.read_csv(CSV_PATH, sep="\t")

dataset = SLTDataset(
    DATA_PATH,
    tokenizer
)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


model = SLTModel().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


def train_one_epoch():

    model.train()

    total_loss = 0

    for pixel_values, labels, attention_mask in tqdm(train_loader):

        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
            attention_mask=attention_mask
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0

    for pixel_values, labels, attention_mask in val_loader:

        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
            attention_mask=attention_mask
        )

        total_loss += outputs.loss.item()

    return total_loss / len(val_loader)

best_loss = float("inf")

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()

    val_loss = evaluate()

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        os.path.join(
            SAVE_DIR,
            "last_model.pt"
        )
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            os.path.join(
                SAVE_DIR,
                "best_model.pt"
            )
        )

        print("Best model saved!")