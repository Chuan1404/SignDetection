from config import ROOT

from src.data.STL_dataset import SLTDataset
from src.models.SLT_model import SLTModel
from src.utils.vocabulary import Vocabulary

import os
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

BATCH_SIZE = 16
EPOCHS = 10
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


dataset = SLTDataset(DATA_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
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

print("VOCAB SIZE:", len(vocab.word2idx))


def train_one_epoch():

    model.train()

    total_loss = 0

    for left, right, tgt in tqdm(loader):

        left, right = align(left, right)

        left = left.to(DEVICE)
        right = right.to(DEVICE)
        tgt = tgt.to(DEVICE)

        inp = tgt[:, :-1]
        label = tgt[:, 1:]

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
            dtype=torch.long
        ).to(DEVICE)

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

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


best_loss = float("inf")

for epoch in range(EPOCHS):

    loss = train_one_epoch()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(
            SAVE_DIR,
            "last_model.pt"
        )
    )

    if loss < best_loss:

        best_loss = loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(
                SAVE_DIR,
                "best_model.pt"
            )
        )

        print("Best model saved!")