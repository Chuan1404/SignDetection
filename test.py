import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from functools import partial

from config import ROOT
from src.data.hand_landmarks import HandLandmarksDataset
from src.models.SLT_model import SignLanguageTranslator

from main import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIR = os.path.join(
    ROOT,
    "datasets",
    "processed",
    "mediapipe"
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "best_model.pt"
)

BATCH_SIZE = 8
MAX_DATASET_LENGTH = 1000


# -----------------------------
# LOAD DATASET + TOKENIZER
# -----------------------------
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
test_size = total_size - train_size - val_size

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(
        collate_fn,
        tokenizer=tokenizer
    )
)


# -----------------------------
# LOAD MODEL
# -----------------------------
sample_feature, _, _ = dataset[0]

model = SignLanguageTranslator(
    input_dim=sample_feature.shape[-1]
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -----------------------------
# EVALUATION
# -----------------------------
references = []
hypotheses = []

print("\nRunning Test Evaluation...\n")

with torch.no_grad():

    for hand_features, labels, video_masks in test_loader:

        hand_features = hand_features.to(DEVICE)
        video_masks = video_masks.to(DEVICE)

        # forward generate
        generated_ids = model.generate(
            hand_features,
            attention_mask=video_masks,
            max_length=64
        )

        preds = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # decode ground truth
        labels = labels.clone()
        labels[labels == -100] = tokenizer.pad_token_id

        refs = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        for pred, ref in zip(preds, refs):

            print("=" * 60)
            print("GT   :", ref)
            print("PRED :", pred)

            hypotheses.append(pred.split())
            references.append([ref.split()])


# -----------------------------
# BLEU 1-4 (ROUGH METRICS)
# -----------------------------
bleu1 = corpus_bleu(
    references,
    hypotheses,
    weights=(1, 0, 0, 0)
)

bleu2 = corpus_bleu(
    references,
    hypotheses,
    weights=(0.5, 0.5, 0, 0)
)

bleu3 = corpus_bleu(
    references,
    hypotheses,
    weights=(1/3, 1/3, 1/3, 0)
)

bleu4 = corpus_bleu(
    references,
    hypotheses,
    weights=(0.25, 0.25, 0.25, 0.25)
)


print("\n" + "=" * 60)
print("ROUGH EVALUATION RESULT")
print("=" * 60)

print(f"BLEU-1 : {bleu1:.4f}")
print(f"BLEU-2 : {bleu2:.4f}")
print(f"BLEU-3 : {bleu3:.4f}")
print(f"BLEU-4 : {bleu4:.4f}")