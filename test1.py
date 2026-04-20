import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
from src.data.how2sign import How2SignDataset
from src.utils.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset
import random

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
        print(x.shape)

    for x in txt:
        fixed_txt.append(x.long())

    # pad variable-length sequences
    # i3d = pad_sequence(fixed_i3d, batch_first=True)          # (B, Ti, 1024)
    # mp  = pad_sequence(fixed_mp, batch_first=True)           # (B, Tm, 99)
    # txt = pad_sequence(fixed_txt, batch_first=True, padding_value=0)

    return i3d, mp, txt



TRAIN_CSV = 'datasets/raw/how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.train.how2sign.tsv'
base_i3d_train = os.path.abspath("datasets/raw/how2sign/i3d_features_how2sign/i3d_features_how2sign/train")
base_mp_train = os.path.abspath("datasets/raw/how2sign/mediapipe_features_how2sign/mediapipe_features/train")

BATCH_SIZE = 8
train_df = pd.read_csv(os.path.abspath(TRAIN_CSV), sep="\t")

tokenizer = Tokenizer()
tokenizer.build_vocab(train_df["translation"].tolist())

train_ds = How2SignDataset(os.path.abspath(TRAIN_CSV), tokenizer, base_mp=base_mp_train, base_i3d=base_i3d_train)

# use 10% ds
n = len(train_ds)
subset_size = int(0.01 * n)

indices = random.sample(range(n), subset_size)

train_ds = Subset(train_ds, indices)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn)


for i3d, mp, text in train_loader:
    print(f"i3d: {i3d.shape}, mp: {mp.shape}, text: {text.shape}")
