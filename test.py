import os
import torch
import pandas as pd

from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from config import ROOT

from src.models.SLT_model import SLTModel
from src.utils.vocabulary import Vocabulary
from src.data.STL_dataset import SLTDataset


# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = rf"{ROOT}\datasets\processed\features"

CSV_PATH = rf"{ROOT}\datasets\annotations\how2sign_train.csv"

MODEL_PATH = rf"{ROOT}\models\best_model.pt"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

MAX_LEN = 50


# =========================================================
# BUILD VOCAB
# =========================================================

df = pd.read_csv(
    CSV_PATH,
    sep="\t"
)

vocab = Vocabulary()

for s in tqdm(df["SENTENCE"].tolist()):

    vocab.build_vocab(
        s.lower().split()
    )

print("VOCAB SIZE:", len(vocab.word2idx))


# =========================================================
# LOAD MODEL
# =========================================================

model = SLTModel(
    vocab_size=len(vocab.word2idx)
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded!")


# =========================================================
# LOAD DATASET
# =========================================================

dataset = SLTDataset(DATA_PATH)


# =========================================================
# GENERATE
# =========================================================

@torch.no_grad()
def generate(left, right):

    model.eval()

    tokens = [SOS_IDX]

    left = left.unsqueeze(0).to(DEVICE)

    right = right.unsqueeze(0).to(DEVICE)

    for _ in range(MAX_LEN):

        inp = torch.tensor(
            tokens,
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        out = model(
            left,
            right,
            inp
        )

        next_token = out[:, -1].argmax(-1).item()

        tokens.append(next_token)

        if next_token == EOS_IDX:
            break

    return tokens


# =========================================================
# METRICS
# =========================================================

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

rouge_scores = []

rouge = rouge_scorer.RougeScorer(
    ['rougeL'],
    use_stemmer=True
)


# =========================================================
# TEST LOOP
# =========================================================

for idx in tqdm(range(len(dataset))):

    left, right, gt_text = dataset[idx]

    # =====================================================
    # ALIGN
    # =====================================================

    T = min(
        left.shape[0],
        right.shape[0]
    )

    left = left[:T]

    right = right[:T]

    # =====================================================
    # GENERATE
    # =====================================================

    pred_tokens = generate(
        left,
        right
    )

    # =====================================================
    # DECODE
    # =====================================================

    pred_sentence = vocab.decode(
        pred_tokens
    )

    gt_sentence = vocab.decode(
        gt_text.tolist()
    )

    pred_words = pred_sentence.split()

    gt_words = gt_sentence.split()

    # =====================================================
    # BLEU
    # =====================================================

    bleu1 = sentence_bleu(
        [gt_words],
        pred_words,
        weights=(1, 0, 0, 0)
    )

    bleu2 = sentence_bleu(
        [gt_words],
        pred_words,
        weights=(0.5, 0.5, 0, 0)
    )

    bleu3 = sentence_bleu(
        [gt_words],
        pred_words,
        weights=(0.33, 0.33, 0.33, 0)
    )

    bleu4 = sentence_bleu(
        [gt_words],
        pred_words,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

    bleu1_scores.append(bleu1)

    bleu2_scores.append(bleu2)

    bleu3_scores.append(bleu3)

    bleu4_scores.append(bleu4)

    # =====================================================
    # ROUGE-L
    # =====================================================

    rouge_result = rouge.score(
        gt_sentence,
        pred_sentence
    )

    rouge_l = rouge_result["rougeL"].fmeasure

    rouge_scores.append(rouge_l)


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n=============== FINAL ===============")

print(f"BLEU-1 : {sum(bleu1_scores)/len(bleu1_scores):.4f}")

print(f"BLEU-2 : {sum(bleu2_scores)/len(bleu2_scores):.4f}")

print(f"BLEU-3 : {sum(bleu3_scores)/len(bleu3_scores):.4f}")

print(f"BLEU-4 : {sum(bleu4_scores)/len(bleu4_scores):.4f}")

print(f"ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")

print("=====================================")
