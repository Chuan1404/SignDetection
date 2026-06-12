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
from src.models.SLT_model import SignLanguageTranslatorV1, SignLanguageTranslatorV2
from config import ROOT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4


FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "mediapipe")
SAVE_DIR = os.path.join(ROOT, "models")
os.makedirs(SAVE_DIR, exist_ok=True)


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


def train_one_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for hand_features, text_ids, video_mask in pbar:

        hand_features = hand_features.to(DEVICE, non_blocking=True)
        text_ids = text_ids.to(DEVICE, non_blocking=True)
        video_mask = video_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            hand_features,
            text_ids=text_ids,
            video_mask=video_mask
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def validate(model, loader):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        for hand_features, text_ids, video_mask in loader:

            hand_features = hand_features.to(DEVICE, non_blocking=True)
            text_ids = text_ids.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)


            outputs = model(
                hand_features,
                text_ids=text_ids,
                video_mask=video_mask
            )

            loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(loader)

def main():

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=2,
        pin_memory=True
    )

    sample, _ = train_dataset[0]

    model = SignLanguageTranslatorV1(
        input_dim=sample.shape[-1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        print("Train:", train_loss)
        print("Val:", val_loss)

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, os.path.join(SAVE_DIR, "best-06-05.pt"))

            print("Saved best model!")


if __name__ == "__main__":
    main()