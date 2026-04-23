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

EPOCHS = 30
MAX_LEN = 2000

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
    # n = len(train_ds)
    # subset_size = int(0.1 * n)
    #
    # indices = random.sample(range(n), subset_size)
    #
    # train_ds = Subset(train_ds, indices)

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


import torch
from torch.nn.utils.rnn import pad_sequence

MAX_TEXT_LEN = 100

def collate_fn(batch):
    i3d_list = []
    mp_list = []
    txt_list = []

    for i3d, mp, txt in batch:
        try:
            i3d = i3d.float()
            mp = mp.float()
            txt = txt.long()

            # visual checks
            if i3d.shape[0] > MAX_LEN:
                continue

            if mp.shape[0] > MAX_LEN:
                continue

            # text check (VERY IMPORTANT)
            if txt.shape[0] > MAX_TEXT_LEN:
                continue

            mp = mp.reshape(mp.shape[0], -1)

            i3d_list.append(i3d)
            mp_list.append(mp)
            txt_list.append(txt)

        except:
            continue

    if len(i3d_list) == 0:
        return None

    i3d = pad_sequence(i3d_list, batch_first=True)
    mp  = pad_sequence(mp_list, batch_first=True)
    txt = pad_sequence(txt_list, batch_first=True, padding_value=0)

    return i3d, mp, txt

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
    print(f"loader: {tqdm(loader)}")
    for batch in tqdm(loader):
        if batch is None:
            continue

        i3d, mp, txt = batch
        inp = txt[:, :-1]
        tgt = txt[:, 1:]

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

