import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config import ROOT

from src.data.STL_dataset import SLTDataset
from src.models.SLT_model import SLTModel
from src.utils.vocabulary import Vocabulary

import gc
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DATA_PATH = rf"{ROOT}\datasets\processed\features"
CSV_PATH = rf"{ROOT}\datasets\annotations\how2sign_train.csv"
SAVE_DIR = rf"{ROOT}\models"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def align(left, right):

    T = min(left.shape[1], right.shape[1])

    left = left[:, :T]
    right = right[:, :T]

    return left, right


def collate_fn(batch):

    lefts = [x[0] for x in batch]
    rights = [x[1] for x in batch]
    texts = [x[2] for x in batch]

    lefts = pad_sequence(
        lefts,
        batch_first=True
    )

    rights = pad_sequence(
        rights,
        batch_first=True
    )

    texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return lefts, rights, texts


df = pd.read_csv(CSV_PATH, sep="\t")

vocab = Vocabulary(min_freq=1)

for s in tqdm(df["SENTENCE"].tolist()):

    vocab.build_vocab(
        s.lower().split()
    )

print("VOCAB SIZE:", len(vocab.word2idx))


# =========================
# DATASET SPLIT 80/10/10
# ORDERED (NOT RANDOM)
# =========================

dataset = SLTDataset(DATA_PATH)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

train_dataset = torch.utils.data.Subset(
    dataset,
    range(0, train_size)
)

val_dataset = torch.utils.data.Subset(
    dataset,
    range(train_size, train_size + val_size)
)

test_dataset = torch.utils.data.Subset(
    dataset,
    range(train_size + val_size, total_size)
)

print(f"TRAIN SIZE: {len(train_dataset)}")
print(f"VAL SIZE: {len(val_dataset)}")
print(f"TEST SIZE: {len(test_dataset)}")


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

model = SLTModel(
    vocab_size=len(vocab.word2idx)
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)

ctc_criterion = nn.CTCLoss(
    blank=PAD_IDX,
    zero_infinity=True
)

scaler = torch.cuda.amp.GradScaler()


def compute_loss(left, right, tgt):

    left, right = align(left, right)

    left = left.to(DEVICE, non_blocking=True)
    right = right.to(DEVICE, non_blocking=True)
    tgt = tgt.to(DEVICE, non_blocking=True)

    inp = tgt[:, :-1]
    label = tgt[:, 1:]

    with torch.cuda.amp.autocast():

        out, ctc_out = model(
            left,
            right,
            inp
        )

        seq_loss = criterion(
            out.reshape(-1, out.size(-1)),
            label.reshape(-1)
        )

        ctc_out = ctc_out.log_softmax(-1)

        ctc_out = ctc_out.permute(1, 0, 2)

        input_lengths = torch.full(
            (left.size(0),),
            ctc_out.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        target_lengths = (
            label != PAD_IDX
        ).sum(dim=1)

        targets = []

        for i in range(label.size(0)):

            targets.append(
                label[i][label[i] != PAD_IDX]
            )

        targets = torch.cat(targets)

        ctc_loss = ctc_criterion(
            ctc_out,
            targets,
            input_lengths,
            target_lengths
        )

        loss = seq_loss + 0.3 * ctc_loss

    return loss, seq_loss, ctc_loss


def train_one_epoch():

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for left, right, tgt in progress_bar:

        optimizer.zero_grad(set_to_none=True)

        loss, seq_loss, ctc_loss = compute_loss(
            left,
            right,
            tgt
        )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "seq": f"{seq_loss.item():.4f}",
            "ctc": f"{ctc_loss.item():.4f}"
        })

        del loss
        del seq_loss
        del ctc_loss

        torch.cuda.empty_cache()
        gc.collect()

    return total_loss / len(train_loader)


def validate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for left, right, tgt in val_loader:

            loss, _, _ = compute_loss(
                left,
                right,
                tgt
            )

            total_loss += loss.item()

            del loss

    return total_loss / len(val_loader)


def test():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for left, right, tgt in test_loader:

            loss, _, _ = compute_loss(
                left,
                right,
                tgt
            )

            total_loss += loss.item()

            del loss

    return total_loss / len(test_loader)


best_loss = float("inf")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch()

    val_loss = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(
                SAVE_DIR,
                "best_model.pt"
            )
        )

        print("Best model saved!")

    torch.cuda.empty_cache()
    gc.collect()


# =========================
# FINAL TEST
# =========================

test_loss = test()

print(f"\nFINAL TEST LOSS: {test_loss:.4f}")