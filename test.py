import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rouge_score import rouge_scorer

from src.utils import FusionComponent


import torch

from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader, random_split, Subset
from transformers import AutoTokenizer
from functools import partial

from config import ROOT
from src.data.hand_landmarks import HandLandmarksDataset
from src.models.SLT_model import SignLanguageTranslator, SignLanguageTranslatorV1

from main import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_how2sign")

MODEL_PATH = os.path.join(
    ROOT,
    "outputs",
    "models",
    "best-06-05.pt"
)

BATCH_SIZE = 8
MAX_DATASET_LENGTH = 1000

fusion_component = FusionComponent()

tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False
)

dataset = HandLandmarksDataset(
    FEATURE_DIR,
    tokenizer,
    fusion_component
)

total_size = len(dataset)

train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

train_dataset = Subset(
    dataset,
    range(0, train_size)
)

val_dataset = Subset(
    dataset,
    range(train_size, train_size + val_size)
)

test_dataset = Subset(
    dataset,
    range(train_size + val_size, total_size)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(
        collate_fn,
        tokenizer=tokenizer
    )
)

test_loader = DataLoader(
    train_dataset,
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
feature, text_ids = train_dataset[0]

model = SignLanguageTranslatorV1(
    input_dim=feature.shape[-1]
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"],
    strict=False
)

model.eval()


# -----------------------------
# EVALUATION
# -----------------------------
references = []
hypotheses = []

rouge_scorer_obj = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

rougeL_scores = []

print("\nRunning Test Evaluation...\n")

with torch.no_grad():

    for features, text_ids, video_mask, text_mask in test_loader:

        features = features.to(DEVICE, non_blocking=True)
        text_ids = text_ids.to(DEVICE, non_blocking=True)
        video_mask = video_mask.to(DEVICE, non_blocking=True)

        generated_ids = model.generate(
            features,
            video_mask=video_mask,
            max_length=64
        )

        preds = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        text_ids = text_ids.clone()
        text_ids[text_ids == -100] = tokenizer.pad_token_id

        refs = tokenizer.batch_decode(
            text_ids,
            skip_special_tokens=True
        )

        for pred, ref in zip(preds, refs):

            pred_norm = pred.strip().lower()
            ref_norm = ref.strip().lower()

            print("=" * 60)
            print("GT   :", ref)
            print("PRED :", pred)

            # BLEU
            hypotheses.append(pred_norm.split())
            references.append([ref_norm.split()])

            # ROUGE-L
            rouge_score = rouge_scorer_obj.score(
                ref_norm,
                pred_norm
            )

            rougeL_scores.append(
                rouge_score["rougeL"].fmeasure
            )

# -----------------------------
# BLEU
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

# -----------------------------
# ROUGE-L
# -----------------------------
rougeL = sum(rougeL_scores) / len(rougeL_scores)

# -----------------------------
# RESULT
# -----------------------------
print("\n" + "=" * 60)
print("EVALUATION RESULT")
print("=" * 60)

print(f"BLEU-1  : {bleu1:.4f}")
print(f"BLEU-2  : {bleu2:.4f}")
print(f"BLEU-3  : {bleu3:.4f}")
print(f"BLEU-4  : {bleu4:.4f}")
print(f"ROUGE-L : {rougeL:.4f}")

print("=" * 60)