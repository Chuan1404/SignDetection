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
df = pd.read_csv(CSV_PATH, sep="\t")

vocab = Vocabulary()

for s in tqdm(df["SENTENCE"].tolist()):
    vocab.build_vocab(s.lower().split())

print("VOCAB SIZE:", len(vocab.word2idx))


# =========================================================
# LOAD MODEL
# =========================================================
model = SLTModel(
    vocab_size=len(vocab.word2idx)
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded!")



dataset = SLTDataset(DATA_PATH)

# 🔥 USE ONLY 10% OF DATA
subset_size = int(len(dataset) * 0.01)
dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))

print(f"Using {len(dataset)} samples (10%)")


# =========================================================
# GENERATE FUNCTION
# =========================================================
@torch.no_grad()
def generate(left, right):

    model.eval()

    tokens = [SOS_IDX]

    left = left.unsqueeze(0).to(DEVICE)
    right = right.unsqueeze(0).to(DEVICE)

    for _ in range(MAX_LEN):

        inp = torch.tensor(tokens, dtype=torch.long)\
            .unsqueeze(0).to(DEVICE)

        out = model(left, right, inp)

        # ✅ FIX: model returns tuple → take logits
        logits = out[0] if isinstance(out, (tuple, list)) else out

        next_token = logits[:, -1].argmax(-1).item()

        tokens.append(next_token)

        if next_token == EOS_IDX:
            break

    return tokens


# =========================================================
# METRICS
# =========================================================
bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
rouge_scores = []

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# =========================================================
# TEST LOOP
# =========================================================
for idx in tqdm(range(len(dataset))):

    left, right, gt_text = dataset[idx]

    # align time dimension
    T = min(left.shape[0], right.shape[0])
    left = left[:T]
    right = right[:T]

    # generate prediction
    pred_tokens = generate(left, right)

    # decode
    pred_sentence = vocab.decode(pred_tokens)
    gt_sentence = vocab.decode(gt_text.tolist())

    print(f"Pred : {pred_sentence}")
    print(f"GT : {gt_sentence}")

    pred_words = pred_sentence.split()
    gt_words = gt_sentence.split()

    # BLEU
    bleu1_scores.append(sentence_bleu([gt_words], pred_words, weights=(1, 0, 0, 0)))
    bleu2_scores.append(sentence_bleu([gt_words], pred_words, weights=(0.5, 0.5, 0, 0)))
    bleu3_scores.append(sentence_bleu([gt_words], pred_words, weights=(0.33, 0.33, 0.33, 0)))
    bleu4_scores.append(sentence_bleu([gt_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25)))

    # ROUGE-L
    rouge_scores.append(
        rouge.score(gt_sentence, pred_sentence)["rougeL"].fmeasure
    )


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