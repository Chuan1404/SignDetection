import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import torch
import pandas as pd

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from src.data.STL_dataset import SLTDataset
from src.models.SLT_model import SLTModel
from src.utils.vocabulary import Vocabulary
from config import ROOT

# =========================
# ENV
# =========================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================
# PATH
# =========================

DATA_PATH = rf"{ROOT}\datasets\processed\features"
CSV_PATH = rf"{ROOT}\datasets\annotations\how2sign_train.csv"
SAVE_DIR = rf"{ROOT}\models"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"DEVICE: {DEVICE}")

# =========================
# HYPERPARAMETERS
# =========================

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# =========================
# ALIGN
# =========================

def align(left, right):

    T = min(left.shape[1], right.shape[1])

    left = left[:, :T]
    right = right[:, :T]

    return left, right

# =========================
# COLLATE
# =========================

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

# =========================
# VOCAB
# =========================

df = pd.read_csv(CSV_PATH, sep="\t")

vocab = Vocabulary(min_freq=1)

for s in tqdm(df["SENTENCE"].tolist()):

    vocab.build_vocab(
        s.lower().split()
    )

print("VOCAB SIZE:", len(vocab.word2idx))

# =========================
# DATASET
# =========================

dataset = SLTDataset(DATA_PATH)

# train on 8 samples only
train_dataset = Subset(dataset, range(8))

# predict on next 8 unseen samples
predict_dataset = Subset(dataset, range(8, 16))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

predict_loader = DataLoader(
    predict_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=0
)

# =========================
# MODEL
# =========================

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

# =========================
# DECODE
# =========================

def decode(tokens):

    words = []

    for t in tokens:

        t = t.item()

        if t == EOS_IDX:
            break

        if t not in [PAD_IDX, SOS_IDX]:

            word = vocab.idx2word.get(t, "<UNK>")

            words.append(word)

    return " ".join(words)

# =========================
# TRAIN
# =========================

def train_one_epoch():

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for left, right, tgt in progress_bar:

        left, right = align(left, right)

        left = left.to(DEVICE, non_blocking=True)
        right = right.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        # teacher forcing
        inp = tgt[:, :-1]

        # expected output
        label = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():

            out, ctc_out = model(
                left,
                right,
                inp
            )

            # =====================
            # Seq Loss
            # =====================

            seq_loss = criterion(
                out.reshape(-1, out.size(-1)),
                label.reshape(-1)
            )

            # =====================
            # CTC Loss
            # =====================

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

            # =====================
            # Total Loss
            # =====================

            loss = seq_loss + 0.3 * ctc_loss

        # =====================
        # Backward
        # =====================

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

    return total_loss / len(train_loader)

# =========================
# PREDICT
# =========================

def predict(loader, title):

    print(f"\n===== {title} =====")

    model.eval()

    with torch.no_grad():

        for left, right, tgt in loader:

            left, right = align(left, right)

            left = left.to(DEVICE)
            right = right.to(DEVICE)
            tgt = tgt.to(DEVICE)

            inp = tgt[:, :-1]

            out, _ = model(
                left,
                right,
                inp
            )

            pred = out.argmax(dim=-1)

            for p, t in zip(pred, tgt):

                print("PRED :", decode(p))
                print("TRUE :", decode(t))
                print("-" * 50)

# =========================
# TRAIN LOOP
# =========================

for epoch in range(EPOCHS):

    loss = train_one_epoch()


# =========================
# PREDICT TRAIN SAMPLES
# =========================

predict(
    train_loader,
    "TRAIN SAMPLES (0-7)"
)

# =========================
# PREDICT UNSEEN SAMPLES
# =========================

predict(
    predict_loader,
    "UNSEEN SAMPLES (8-15)"
)