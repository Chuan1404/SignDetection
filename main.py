import os

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config import ROOT, MIN_FREQ

from src.data.STL_dataset import SLTDataset
from src.models.SLT_model import SLTModel
from src.utils.vocabulary import Vocabulary

import gc
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DATA_PATH = rf"{ROOT}\datasets\processed\i3d_features"
CSV_PATH = rf"{ROOT}\datasets\annotations\how2sign_train.csv"
SAVE_DIR = rf"{ROOT}\models"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def align(left, right):

    T = min(left.shape[1], right.shape[1])

    left = left[:, :T]
    right = right[:, :T]

    return left, right


def collate_fn(batch):

    lefts = [x[0] for x in batch]
    rights = [x[1] for x in batch]
    texts = [x[2] for x in batch]

    lefts = pad_sequence(
        lefts,
        batch_first=True
    )

    rights = pad_sequence(
        rights,
        batch_first=True
    )

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return lefts, rights, texts


df = pd.read_csv(CSV_PATH, sep="\t")

vocabulary = Vocabulary(MIN_FREQ)

for s in tqdm(df["SENTENCE"].tolist()):

    vocabulary.build_vocab(
        s.lower().split()
    )

print("VOCAB SIZE:", len(vocabulary.word2idx))


dataset = SLTDataset(DATA_PATH, vocabulary)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

train_dataset = torch.utils.data.Subset(
    dataset,
    range(0, train_size)
)

val_dataset = torch.utils.data.Subset(
    dataset,
    range(train_size, train_size + val_size)
)

test_dataset = torch.utils.data.Subset(
    dataset,
    range(train_size + val_size, total_size)
)

print(f"TRAIN SIZE: {len(train_dataset)}")
print(f"VAL SIZE: {len(val_dataset)}")
print(f"TEST SIZE: {len(test_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

model = SLTModel(
    vocab_size=len(vocabulary.word2idx)
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX,
    label_smoothing=0.1
)

scaler = torch.cuda.amp.GradScaler()

def compute_loss(left, right, tgt):

    left, right = align(left, right)

    left = left.to(DEVICE, non_blocking=True)
    right = right.to(DEVICE, non_blocking=True)
    tgt = tgt.to(DEVICE, non_blocking=True)

    inp = tgt[:, :-1]
    label = tgt[:, 1:]

    with torch.cuda.amp.autocast():

        out = model(
            left,
            right,
            inp
        )

        loss = criterion(
            out.reshape(-1, out.size(-1)),
            label.reshape(-1)
        )

    return loss


def train_one_epoch():

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for left, right, tgt in progress_bar:

        optimizer.zero_grad(set_to_none=True)

        loss = compute_loss(
            left,
            right,
            tgt
        )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

        del loss

        torch.cuda.empty_cache()
        gc.collect()

    return total_loss / len(train_loader)


def validate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for left, right, tgt in val_loader:

            loss = compute_loss(
                left,
                right,
                tgt
            )

            total_loss += loss.item()

            del loss

    return total_loss / len(val_loader)


best_loss = float("inf")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch()

    val_loss = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if train_loss < best_loss and epoch % 10:
        best_loss = train_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(
                SAVE_DIR,
                "26.05.27.best_model.pt"
            )
        )

        print("Best model saved!")

    torch.cuda.empty_cache()
    gc.collect()
