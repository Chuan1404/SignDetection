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
from functools import partial

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
LR = 3e-5

MAX_DATASET_LENGTH = 1000

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/mt5-small",
        use_fast=False
    )

    dataset = HandLandmarksDataset(
        FEATURE_DIR,
        tokenizer,
        MAX_DATASET_LENGTH
    )

    total_size = len(dataset)

    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = int(total_size * 0.1)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer
        ),
        pin_memory=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer
        ),
        pin_memory=True,
        num_workers=4
    )

    feature_sample, text_sample, _ = train_dataset[0]

    model = SignLanguageTranslator(
        input_dim=feature_sample.shape[-1] # 126
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer)

        val_loss = validate(model=model, val_loader=val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if (val_loss < best_loss) and ((epoch + 1) % 10 == 0):
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

def collate_fn(batch, tokenizer):

    features = []
    texts = []
    text_masks = []

    for feature, text, mask in batch:
        features.append(feature)
        texts.append(text)
        text_masks.append(mask)

    lengths = [feature.shape[0] for feature in features]

    features = pad_sequence(
        features,
        batch_first=True
    )

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    max_len = features.shape[1]

    video_mask = (
            torch.arange(max_len)[None, :]
            < torch.tensor(lengths)[:, None]
    )

    video_mask = video_mask.bool()

    labels = texts.clone()

    labels[
        labels == tokenizer.pad_token_id
    ] = -100

    return features, labels, video_mask

def train_one_epoch(model, train_loader, optimizer):

    model.train()

    total_loss = 0

    pbar = tqdm(
        train_loader,
        desc="Training"
    )

    for hand_features, text_ids, video_masks in pbar:

        optimizer.zero_grad(
            set_to_none=True
        )

        hand_features = hand_features.to(DEVICE, non_blocking=True)
        text_ids = text_ids.to(DEVICE,non_blocking=True)
        video_masks = video_masks.to(DEVICE, non_blocking=True)

        outputs = model(
            hand_features,
            text_ids=text_ids,
            attention_mask = video_masks
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = total_loss / len(train_loader)

    return train_loss

def validate(model, val_loader):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for hand_features, text_ids, video_masks in val_loader:
            hand_features = hand_features.to(DEVICE, non_blocking=True)
            text_ids = text_ids.to(DEVICE, non_blocking=True)
            video_masks = video_masks.to(DEVICE, non_blocking=True)

            outputs = model(
                hand_features,
                text_ids=text_ids,
                attention_mask=video_masks
            )

            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":
    main()

