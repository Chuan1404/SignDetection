import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from collections import defaultdict

from transformers import AutoTokenizer

import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.utils.fusion_component import FusionComponent
from src.data.WSASL_raw import WLASLLandmarksDataset
from src.models.SLT_model import SignLanguageTranslatorV3, SignLanguageTranslatorV1
from config import ROOT

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
TOP_K       = 5
SEED        = 42

FEATURE_DIR  = os.path.join(ROOT, "datasets", "processed", "full_body_wlasl")
ANNOTATION_DIR = os.path.join(ROOT, "datasets", "annotations", "test.json")
MODEL_PATH   = os.path.join(ROOT, "outputs", "models", "26_07_23.pt")

fusion_component = FusionComponent()


def collate_fn(batch):
    features, labels = [], []
    for feature, label in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        labels.append(label)

    real_lengths = [f.shape[0] for f in features]
    features = pad_sequence(features, batch_first=True)
    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()
    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels, video_mask


def evaluate(model, loader, idx2gloss, top_k=5):
    """Tính top-1, top-k accuracy và thu thập wrong predictions"""

    model.eval()

    total_loss     = 0
    correct_top1   = 0
    correct_topk   = 0
    total_samples  = 0
    wrong_preds    = []   # [(gt_gloss, pred_gloss), ...]

    with torch.no_grad():
        for features, labels, video_mask in tqdm(loader, desc="Evaluating"):
            features   = features.to(DEVICE)
            labels     = labels.to(DEVICE)
            video_mask = video_mask.to(DEVICE)

            outputs = model(features, labels=labels, video_mask=video_mask)
            total_loss += outputs.loss.item()

            logits = outputs.logits                         # (B, num_classes)

            # Top-1
            preds_top1 = logits.argmax(dim=-1)             # (B,)
            correct_top1 += (preds_top1 == labels).sum().item()

            # Top-K
            k = min(top_k, logits.size(-1))
            preds_topk  = logits.topk(k=k, dim=-1).indices # (B, k)
            in_topk     = (preds_topk == labels.unsqueeze(-1)).any(dim=-1)
            correct_topk += in_topk.sum().item()

            total_samples += labels.size(0)

            # In GT và PRED từng sample
            for i in range(labels.size(0)):
                gt      = idx2gloss[labels[i].item()]
                pred    = idx2gloss[preds_top1[i].item()]
                correct = "OK" if labels[i] == preds_top1[i] else "X"
                print(f"  [{correct}] GT: {gt:<20s}  PRED: {pred}")

                # Thu thập wrong predictions để phân tích
                if labels[i] != preds_top1[i]:
                    wrong_preds.append((gt, pred))

    avg_loss = total_loss / len(loader)
    top1_acc = correct_top1 / total_samples
    topk_acc = correct_topk / total_samples

    return avg_loss, top1_acc, topk_acc, wrong_preds


def print_most_confused(wrong_preds, n=10):
    """In ra các cặp (gt, pred) bị nhầm nhiều nhất"""
    counter = defaultdict(int)
    for gt, pred in wrong_preds:
        counter[(gt, pred)] += 1

    sorted_pairs = sorted(counter.items(), key=lambda x: -x[1])

    print(f"\n  Top {n} most confused pairs (GT → PRED):")
    for (gt, pred), cnt in sorted_pairs[:n]:
        print(f"    [{cnt:3d}x]  '{gt}' → '{pred}'")


def main():

    with open(os.path.join(ROOT, "datasets", "annotations", "gloss2idx.json"), "r") as f:
        gloss2idx = json.load(f)

    print(f"\n{'='*60}")
    print(f"  WLASL Test Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Khong tim thay checkpoint: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_kwargs = checkpoint["model_kwargs"]

    print(f"  Checkpoint epoch : {checkpoint['epoch']}")
    print(f"  Val loss (saved) : {checkpoint['val_loss']:.4f}")
    print(f"  Val top-1 (saved): {checkpoint.get('val_top1_acc', 'N/A')}")

    test_dataset = WLASLLandmarksDataset(
        FEATURE_DIR, ANNOTATION_DIR , fusion_component, max_samples=None
    )

    num_classes = len(gloss2idx)
    print(f"Dataset — test: {num_classes}")

    # Check for feature dimension mismatch (e.g. old 183-dim checkpoint vs new 185-dim features)
    feature, _ = test_dataset[0]

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )

    idx2gloss = {v: k for k, v in gloss2idx.items()}

    # Load model
    model = SignLanguageTranslatorV3(**model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Evaluate
    test_loss, top1_acc, topk_acc, wrong_preds = evaluate(
        model, test_loader, idx2gloss, top_k=TOP_K
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test loss    : {test_loss:.4f}")
    print(f"  Top-1 acc    : {top1_acc*100:.2f}%")
    print(f"  Top-{TOP_K} acc    : {topk_acc*100:.2f}%")
    print(f"  Wrong preds  : {len(wrong_preds)} / {len(test_dataset)}")

    print_most_confused(wrong_preds, n=10)

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()