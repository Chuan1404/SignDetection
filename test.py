import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from config import ROOT
from pretrained_model import SLTModel
from src.data.STL_dataset import SLTDataset


# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = rf"{ROOT}\datasets\processed\videomae_features"
CSV_PATH = rf"{ROOT}\datasets\annotations\how2sign_train.csv"
MODEL_PATH = rf"{ROOT}\models\best_model.pt"   # change if needed

BATCH_SIZE = 1
MAX_LEN = 50


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False
)

PAD_IDX = tokenizer.pad_token_id


# =========================================================
# DATASET + COLLATE
# =========================================================

def collate_fn(batch):
    pixels = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    pixel_lengths = [p.size(0) for p in pixels]

    pixels = pad_sequence(pixels, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD_IDX)

    attention_mask = torch.zeros(
        pixels.size(0),
        pixels.size(1),
        dtype=torch.long
    )

    for i, l in enumerate(pixel_lengths):
        attention_mask[i, :l] = 1

    return pixels, labels, attention_mask


dataset = SLTDataset(DATA_PATH, tokenizer)

test_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# =========================================================
# MODEL
# =========================================================

model = SLTModel().to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("MODEL LOADED")


# =========================================================
# METRICS
# =========================================================

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
rouge_scores = []


# =========================================================
# INFERENCE (IMPORTANT FIX: USE generate)
# =========================================================

@torch.no_grad()
def predict(pixel_values, attention_mask):
    """
    pixel_values: (B, T, C, H, W) or (B, T, D)
    """

    # encoder
    memory = model.input_proj(pixel_values)
    memory = model.temporal_encoder(memory)
    memory = model.cross_proj(memory)

    # generate (mT5)
    generated = model.decoder.generate(
        inputs_embeds=memory,
        attention_mask=attention_mask,
        max_length=MAX_LEN,
        num_beams=1
    )

    return generated


def decode(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)


# =========================================================
# EVALUATION LOOP
# =========================================================

with torch.no_grad():

    for pixel_values, labels, attention_mask in tqdm(test_loader):

        pixel_values = pixel_values.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        pred_tokens = predict(pixel_values, attention_mask)

        for i in range(pred_tokens.size(0)):

            pred_text = decode(pred_tokens[i].cpu().numpy())
            gt_text = decode(labels[i].cpu().numpy())

            print("PRED:", pred_text)
            print("GT  :", gt_text)

            pred_words = pred_text.split()
            gt_words = gt_text.split()

            # BLEU
            bleu1_scores.append(sentence_bleu([gt_words], pred_words, weights=(1, 0, 0, 0)))
            bleu2_scores.append(sentence_bleu([gt_words], pred_words, weights=(0.5, 0.5, 0, 0)))
            bleu3_scores.append(sentence_bleu([gt_words], pred_words, weights=(0.33, 0.33, 0.33, 0)))
            bleu4_scores.append(sentence_bleu([gt_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25)))

            # ROUGE-L
            rouge = scorer.score(gt_text, pred_text)
            rouge_scores.append(rouge["rougeL"].fmeasure)


# =========================================================
# FINAL RESULTS (B1234 + ROUGE)
# =========================================================

print("\n==============================")
print("FINAL RESULTS (B1234 + ROUGE-L)")
print("==============================")

print(f"BLEU-1 : {sum(bleu1_scores)/len(bleu1_scores)*100:.2f}")
print(f"BLEU-2 : {sum(bleu2_scores)/len(bleu2_scores)*100:.2f}")
print(f"BLEU-3 : {sum(bleu3_scores)/len(bleu3_scores)*100:.2f}")
print(f"BLEU-4 : {sum(bleu4_scores)/len(bleu4_scores)*100:.2f}")
print(f"ROUGE-L: {sum(rouge_scores)/len(rouge_scores)*100:.2f}")