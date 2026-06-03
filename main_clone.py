import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from src.data.hand_landmarks import HandLandmarksDataset
from src.models.SLT_model import SignLanguageTranslator
import torch

import gc
from tqdm import tqdm

from transformers import AutoTokenizer

from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from config import ROOT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIR = os.path.join(
    ROOT,
    "datasets",
    "processed",
    "mediapipe"
)

SAVE_DIR = os.path.join(
    ROOT,
    "models"
)

os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False
)

dataset = HandLandmarksDataset(
    FEATURE_DIR,
    tokenizer
)

total_size = len(dataset)

train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print("TRAIN:", len(train_dataset))
print("VAL:", len(val_dataset))
print("TEST:", len(test_dataset))

def collate_fn(batch):

    features = []
    texts = []

    for feature, text in batch:
        features.append(feature)
        texts.append(text)

    features = pad_sequence(
        features,
        batch_first=True
    )

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    labels = texts.clone()

    labels[
        labels == tokenizer.pad_token_id
    ] = -100

    return features, labels

def compute_loss(hand_features,labels):
    hand_features = hand_features.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    with torch.cuda.amp.autocast():

        outputs = model(
            hand_features,
            labels=labels
        )

        loss = outputs.loss

    return loss

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

model = SignLanguageTranslator(
    input_dim=126
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

scaler = torch.cuda.amp.GradScaler()

def compute_loss(hand_features, labels):
    hand_features = hand_features.to(DEVICE, non_blocking=True)

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    with torch.cuda.amp.autocast():
        outputs = model(
            hand_features,
            labels=labels
        )

        loss = outputs.loss

    return loss

def train_one_epoch():

    model.train()

    total_loss = 0

    pbar = tqdm(
        train_loader,
        desc="Training"
    )

    for hand_features, labels in pbar:
        optimizer.zero_grad(
            set_to_none=True
        )

        loss = compute_loss(
            hand_features,
            labels
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

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(train_loader)

def validate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for hand_features, labels in val_loader:

            loss = compute_loss(
                hand_features,
                labels
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)

best_loss = float("inf")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch()

    val_loss = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:

        best_loss = val_loss

        save_path = os.path.join(
            SAVE_DIR,
            "best_model.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss":
                    val_loss
            },
            save_path
        )

        print("Best model saved!")

    torch.cuda.empty_cache()
    gc.collect()
