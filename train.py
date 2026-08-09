import os

from src.data.augmentation import AugmentedSkeletonDataset, SkeletonAugmentor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from src.data.WSASL_raw import WLASLLandmarksDataset
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.utils import FusionComponent
from src.models.SLT_model import SignLanguageTranslatorV3, SignLanguageTranslatorV2, SignLanguageTranslatorV1, \
    SignLanguageTranslatorV4
from config import ROOT


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
EPOCHS      = 300
LR          = 1e-4
PATIENCE    = 10
TOP_K       = 3
VAL_RATIO   = 0
SEED        = 42
GCN_OUT_CHANNELS = 32


FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_wlasl")
ANNOTATION_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL2000")
SAVE_DIR    = os.path.join(ROOT, "outputs", "models")
SAVE_MODEL = os.path.join(SAVE_DIR, "v1_wlasl2000_26_08_09.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

fusion_component = FusionComponent()

def collate_fn(batch):
    features, hand_features, labels = [], [], []
    for feature, hand_feature, label in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        hand_features.append(torch.as_tensor(hand_feature, dtype=torch.float32))
        labels.append(label)

    real_lengths = [f.shape[0] for f in features]

    features = pad_sequence(features, batch_first=True)
    hand_features = pad_sequence(hand_features, batch_first=True)

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    labels = torch.tensor(labels, dtype=torch.long)

    return features, hand_features, labels, video_mask

def train_one_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for features, hand_normalize_features, labels, video_mask in pbar:

        features   = features.to(DEVICE, non_blocking=True)
        hand_normalize_features = hand_normalize_features.to(DEVICE, non_blocking=True)
        labels     = labels.to(DEVICE, non_blocking=True)
        video_mask = video_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            features,
            # hand_normalize_features,
            labels=labels,
            video_mask=video_mask
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def validate(model, loader, top_k=5):

    model.eval()
    total_loss = 0

    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    with torch.no_grad():
        for features, hand_normalize_features, labels, video_mask in loader:
            features   = features.to(DEVICE, non_blocking=True)
            hand_normalize_features = hand_normalize_features.to(DEVICE, non_blocking=True)
            labels     = labels.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)

            outputs = model(
                features,
                # hand_normalize_features,
                labels=labels,
                video_mask=video_mask
            )
            total_loss += outputs.loss.item()

            logits = outputs.logits                                # (B, num_classes)

            preds_top1 = logits.argmax(dim=-1)                     # (B,)
            total_correct_top1 += (preds_top1 == labels).sum().item()

            k = min(top_k, logits.size(-1))
            preds_topk = logits.topk(k=k, dim=-1).indices          # (B, k)
            in_topk = (preds_topk == labels.unsqueeze(-1)).any(dim=-1)
            total_correct_topk += in_topk.sum().item()

            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    top1_acc = total_correct_top1 / total_samples if total_samples > 0 else 0.0
    topk_acc = total_correct_topk / total_samples if total_samples > 0 else 0.0

    return avg_loss, top1_acc, topk_acc

def main():
    with open(os.path.join(ANNOTATION_DIR, "gloss2idx.json"), "r") as f:
        gloss2idx = json.load(f)

    num_classes = len(gloss2idx)

    base_train = WLASLLandmarksDataset(
        FEATURE_DIR, ANNOTATION_DIR, fusion_component, mode="train", max_samples=None
    )
    base_val = WLASLLandmarksDataset(
        FEATURE_DIR, ANNOTATION_DIR, fusion_component, mode="test", max_samples=None
    )

    train_dataset = AugmentedSkeletonDataset(base_train, SkeletonAugmentor())
    # train_dataset = base_train
    val_dataset   = base_val

    print(f"Dataset — train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_classes = set(label for _, _,  label in base_train)
    val_classes   = set(label for _, _, label in base_val)
    print(f"Classes in train: {len(train_classes)}/{num_classes}, "
          f"classes in val: {len(val_classes)}/{num_classes}, "
          f"val classes missing from train: {len(val_classes - train_classes)}")

    feature, hand_normalize_features, _ = base_train[0]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )


    model_kwargs = dict(
        # input_dim=feature.shape[-1],
        # hidden_dim=256,
        # num_encoder_layers=6,
        # nhead=8,
        # dim_feedforward=2048,
        dropout=0.2,
        # max_seq_len=5000,
        num_classes=num_classes,
        # aux_loss_weight = 2
        # hand_channels=(32, 64, 126, 256, 512)
        # gcn_out_channels=GCN_OUT_CHANNELS
    )

    model = SignLanguageTranslatorV1(**model_kwargs).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_top_1  = 0
    no_improve = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")

        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_top1_acc, val_topk_acc = validate(model, val_loader, top_k=TOP_K)

        print(f"Train loss      : {train_loss:.4f}")
        print(f"Val   loss      : {val_loss:.4f}")
        print(f"Val   top-1 acc : {val_top1_acc*100:.2f}%")
        print(f"Val   top-{TOP_K} acc : {val_topk_acc*100:.2f}%")

        print()
        if best_top_1 < val_top1_acc:
            best_top_1  = val_top1_acc
            no_improve = 0
            torch.save({
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "epoch":        epoch,
                "val_loss":     val_loss,
                "val_top1_acc": val_top1_acc,
                "val_topk_acc": val_topk_acc,
                "model_kwargs": model_kwargs,
            }, os.path.join(SAVE_MODEL))
            print(f"✓ Saved best model  {val_top1_acc*100:.2f}%")
        else:
            no_improve += 1
            print(f"  No improvement (best={best_top_1*100:.2f}%, patience={no_improve}/{PATIENCE})")

            if no_improve >= PATIENCE:
                print(f"\n⚑ Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    main()