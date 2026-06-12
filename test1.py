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
from src.models.SLT_model import SignLanguageTranslatorV1
from config import ROOT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
EPOCHS = 100
LR = 1e-3


FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "mediapipe")


# =========================
# COLLATE
# =========================
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


# =========================
# TRAIN
# =========================
def train_one_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for hand_features, text_ids, video_mask in pbar:

        hand_features = hand_features.to(DEVICE)
        text_ids = text_ids.to(DEVICE)
        video_mask = video_mask.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            hand_features,
            text_ids=text_ids,
            video_mask=video_mask
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# =========================
# SHOW PREDICTION (NO SAVE)
# =========================
@torch.no_grad()
def show_prediction(model, loader, tokenizer):

    model.eval()

    hand_features, labels, video_mask = next(iter(loader))

    hand_features = hand_features.to(DEVICE)
    video_mask = video_mask.to(DEVICE)

    generated_ids = model.generate(
        hand_features,
        video_mask=video_mask,
        max_length=64,
        num_beams=1
    )

    pred = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    gt_labels = labels.clone()
    gt_labels[gt_labels == -100] = tokenizer.pad_token_id

    gt = tokenizer.decode(
        gt_labels[0],
        skip_special_tokens=True
    )

    print("\n" + "=" * 80)
    print("GT  :", gt)
    print("PRED:", pred)
    print("=" * 80)


# =========================
# MAIN
# =========================
def main():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/mt5-small",
        use_fast=False
    )

    dataset = HandLandmarksDataset(FEATURE_DIR, tokenizer)

    # OVERFIT SINGLE SAMPLE
    dataset = Subset(dataset, [0])

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=True
    )

    sample_feature, _ = dataset[0]

    model = SignLanguageTranslatorV1(
        input_dim=sample_feature.shape[-1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("\nStarting Overfit Test...\n")

    for epoch in range(EPOCHS):

        loss = train_one_epoch(model, loader, optimizer)

        print(f"Epoch {epoch+1:03d} | Loss = {loss:.4f}")

        # show prediction every epoch (IMPORTANT for debugging)
        show_prediction(model, loader, tokenizer)


if __name__ == "__main__":
    main()