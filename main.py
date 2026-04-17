import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from models.sign_translator import SignTranslator
from src.data.how2sign import How2SignDataset
from src.utils.tokenizer import Tokenizer

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
TRAIN_CSV = r'C:\Users\ADMIN\OneDrive\Desktop\SignDetection\datasets\how2sign\tsv_files_how2sign\tsv_files_how2sign\cvpr23.fairseq.i3d.train.how2sign.tsv'
VAL_CSV = r'C:\Users\ADMIN\OneDrive\Desktop\SignDetection\datasets\how2sign\tsv_files_how2sign\tsv_files_how2sign\cvpr23.fairseq.i3d.val.how2sign.tsv'

def main():
    train_df = pd.read_csv(TRAIN_CSV,sep="\t")
    # val_df = pd.read_csv(VAL_CSV, sep="\t")

    tokenizer = Tokenizer()
    tokenizer.build_vocab(train_df["translation"].tolist())

    train_ds = How2SignDataset(TRAIN_CSV, tokenizer)
    val_ds = How2SignDataset(VAL_CSV, tokenizer)

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
    print(f"sample_i3d: {sample_i3d.shape}")
    print(f"sample_mp: {sample_mp.shape}")
    print(f"sample_txt: {sample_txt.shape}")
    model = SignTranslator(
        sample_i3d.shape[1],
        sample_mp.shape[1],
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
            torch.save(model.state_dict(), r"C:\Users\ADMIN\OneDrive\Desktop\SignDetection\src\outputs\models\save_model.pth")
            print("Saved best model")

    print("Done")

def collate_fn(batch):
    i3d, mp, txt = zip(*batch)

    # ensure tensors
    i3d = [torch.tensor(x).float().squeeze() for x in i3d]
    mp  = [torch.tensor(x).float().squeeze() for x in mp]
    txt = [torch.tensor(x).long() for x in txt]

    # pad variable lengths
    i3d = pad_sequence(i3d, batch_first=True)
    mp  = pad_sequence(mp, batch_first=True)
    txt = pad_sequence(txt, batch_first=True, padding_value=0)

    return i3d, mp, txt

# def pad_sequence(seq_list, pad=0):
#     max_len = max([x.shape[0] for x in seq_list])
#
#     out = []
#
#     for x in seq_list:
#         pad_len = max_len - x.shape[0]
#
#         if len(x.shape) == 2:
#             p = torch.zeros(pad_len, x.shape[1])
#         else:
#             p = torch.full((pad_len,), pad)
#
#         out.append(torch.cat([x, p], dim=0))

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0

    for i3d, mp, txt in tqdm(loader):
        inp = txt[:, :-1]
        tgt = txt[:, 1:]
        print(f"i3d: {i3d.shape}, mp: {mp.shape}, inp: {inp.shape}")

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
# def main():
#
#     dataset = How2SignDataset("data/annotations/how2sign_train.csv")
#     loader = DataLoader(dataset)
#
#     model = MultiStreamModel()
#
#     for video_name, sentence in loader:
#         # print("sign_x:", sign_x.shape)
#         # print("finger_x:", finger_x.shape)
#         # print("lip_x:", lip_x.shape)
#         # print("tgt_seq:", tgt_seq.shape)
#
#         # logits = model(sign_x, finger_x, lip_x, tgt_seq)
#
#         print("Output:", video_name, sentence)
#
if __name__ == "__main__":
    main()

