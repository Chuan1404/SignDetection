import json
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from src.data.WSASL_raw import WLASLLandmarksDataset, stratified_split
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.utils import FusionComponent
from src.models.SLT_model import SignLanguageTranslatorV3, SignLanguageTranslatorV1
from config import ROOT


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
EPOCHS      = 100
LR          = 1e-4
PATIENCE    = 10
WARMUP_EPOCHS = 5
TOP_K       = 5
VAL_RATIO   = 0.1
TEST_RATIO = 0.1
SEED = 42


FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_wlasl")
SAVE_DIR    = os.path.join(ROOT, "outputs", "models")
os.makedirs(SAVE_DIR, exist_ok=True)

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


def get_lr_scale(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def train_one_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for features, labels, video_mask in pbar:
        features   = features.to(DEVICE, non_blocking=True)
        labels     = labels.to(DEVICE, non_blocking=True)
        video_mask = video_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            features,
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
        for features, labels, video_mask in loader:
            features   = features.to(DEVICE, non_blocking=True)
            labels     = labels.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)

            outputs = model(
                features,
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

    # Pool ALL available labelled data (train + val annotations) into one
    # dataset, then stratify-split it so every class is proportionally
    # represented in both partitions — no samples are left unused.
    full_dataset = WLASLLandmarksDataset(
        FEATURE_DIR,
        fusion_component,
        split=["train", "val", "test"]
    )

    train_dataset, val_dataset = stratified_split(
        full_dataset,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    print(f"Dataset (stratified) — train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )

    feature, _ = full_dataset[0]

    num_classes = len(full_dataset.gloss2idx)

    model_kwargs = dict(
        input_dim=feature.shape[-1],
        # hidden_dim=256,
        # nhead=8,
        # dim_feedforward=2048,
        # num_encoder_layers=8,
        dropout=0.2,
        # max_seq_len=5000,
        num_classes=num_classes,
        # use_gcn=True,
        # gcn_out_channels=32
    )

    model = SignLanguageTranslatorV3(**model_kwargs).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    param_groups = model.get_optimizer_param_groups(lr=LR)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_loss  = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")
        if epoch < WARMUP_EPOCHS:
            lr_scale = get_lr_scale(epoch, WARMUP_EPOCHS)
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = base_lrs[i] * lr_scale

        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_top1_acc, val_topk_acc = validate(model, val_loader, top_k=TOP_K)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step(val_loss)

        print(f"Train loss      : {train_loss:.4f}")
        print(f"Val   loss      : {val_loss:.4f}")
        print(f"Val   top-1 acc : {val_top1_acc*100:.2f}%")
        print(f"Val   top-{TOP_K} acc : {val_topk_acc*100:.2f}%")

        if val_loss < best_loss:
            best_loss  = val_loss
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
            }, os.path.join(SAVE_DIR, "26_07_13_best.pt"))

            print(f"✓ Saved best model  (val_loss={best_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement (best={best_loss:.4f}, patience={no_improve}/{PATIENCE})")

            if no_improve >= PATIENCE:
                print(f"\n⚑ Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    main()