import torch
from torch.nn.utils.rnn import pad_sequence


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

