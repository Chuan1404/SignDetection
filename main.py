from torch.utils.data import Subset
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os

from config import BASE_MP_TRAIN, BASE_I3D_TRAIN, BASE_I3D_VAL, BASE_MP_VAL, TRAIN_CSV, VAL_CSV, BATCH_SIZE, LR
from models.sign_translator import SignTranslator
from src.data.how2sign import How2SignDataset
from src.utils.tokenizer import Tokenizer

EPOCHS = 10

SAVE_DIR = os.path.abspath("outputs/models")


# Build tokenizer
train_df = pd.read_csv(os.path.abspath(TRAIN_CSV), sep="\t")
tokenizer = Tokenizer()
tokenizer.build_vocab(train_df["translation"].tolist())

def main():

    # val_df = pd.read_csv(VAL_CSV, sep="\t")

    train_ds = How2SignDataset(os.path.abspath(TRAIN_CSV), tokenizer, base_mp=BASE_MP_TRAIN, base_i3d=BASE_I3D_TRAIN)
    val_ds = How2SignDataset(os.path.abspath(VAL_CSV), tokenizer, base_mp=BASE_MP_VAL, base_i3d=BASE_I3D_VAL)

    # use 10% ds
    n = len(train_ds)
    subset_size = int(0.1 * n)

    indices = random.sample(range(n), subset_size)

    train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # detect feature dim
    sample_i3d, sample_mp, sample_txt = train_ds[0]

    model = SignTranslator(
        sample_i3d.shape[1],
        sample_mp.shape[1] * sample_mp.shape[2],
        tokenizer.vocab_size
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = 999

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}")
        print("Train:", train_loss)
        print("Val:", val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "save_model.pth"))
            print("Saved best model")

    print("Done")


def collate_fn(batch):
    i3d, mp, txt = zip(*batch)

    fixed_i3d = []
    fixed_mp = []
    fixed_txt = []

    for x in i3d:
        x = x.float()
        fixed_i3d.append(x)

    for x in mp:
        x = x.float()
        T = x.shape[0]
        x = torch.reshape(x, (T, -1))
        fixed_mp.append(x)

    for x in txt:
        fixed_txt.append(x.long())

    # pad variable-length sequences
    i3d = pad_sequence(fixed_i3d, batch_first=True)          # (B, Ti, 1024)
    mp  = pad_sequence(fixed_mp, batch_first=True)           # (B, Tm, 99)
    txt = pad_sequence(fixed_txt, batch_first=True, padding_value=0)

    return i3d, mp, txt

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0

    for i3d, mp, txt in tqdm(loader):
        print(i3d.shape)
        print(mp.shape)
        print(txt.shape)

        inp = txt[:, :-1]
        tgt = txt[:, 1:]

        print(f"txt:{txt.shape} tgt:{tgt.shape}")
        out = model(i3d, mp, inp)

        loss = criterion(
            out.reshape(-1, out.shape[-1]),
            tgt.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for i3d, mp, txt in loader:

            inp = txt[:, :-1]
            tgt = txt[:, 1:]

            out = model(i3d, mp, inp)

            loss = criterion(
                out.reshape(-1, out.shape[-1]),
                tgt.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)

if __name__ == "__main__":
    main()

