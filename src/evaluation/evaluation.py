import os
import random
import torch
import pandas as pd

from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from config import ROOT, TRAIN_CSV, TEST_CSV, BASE_MP_TEST, BASE_I3D_TEST
from models.sign_translator import SignTranslator
from src.data.how2sign import How2SignDataset
from src.utils.tokenizer import Tokenizer


MODEL_PATH = os.path.join(ROOT, "outputs/models/save_model.pth")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
MAX_LEN = 30



def collate_fn(batch):
    i3d, mp, txt = zip(*batch)

    fixed_i3d = []
    fixed_mp = []
    fixed_txt = []

    for x in i3d:
        fixed_i3d.append(x.float())

    for x in mp:
        x = x.float()
        T = x.shape[0]
        x = x.reshape(T, -1)
        fixed_mp.append(x)

    for x in txt:
        fixed_txt.append(x.long())

    i3d = pad_sequence(fixed_i3d, batch_first=True)
    mp = pad_sequence(fixed_mp, batch_first=True)
    txt = pad_sequence(fixed_txt, batch_first=True, padding_value=0)

    return i3d, mp, txt

def generate(model, i3d, mp, bos_id, eos_id, max_len=MAX_LEN):
    tokens = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len):
        with torch.no_grad():
            out = model(i3d, mp, tokens)  # [1, L, vocab]

        next_token = out[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return tokens[0].cpu().tolist()

def evaluate_metrics(predictions, references):

    # =========================
    # BLEU-1 to BLEU-4
    # =========================
    bleu1 = BLEU(max_ngram_order=1)
    bleu2 = BLEU(max_ngram_order=2)
    bleu3 = BLEU(max_ngram_order=3)
    bleu4 = BLEU(max_ngram_order=4)

    b1 = bleu1.corpus_score(predictions, [references]).score
    b2 = bleu2.corpus_score(predictions, [references]).score
    b3 = bleu3.corpus_score(predictions, [references]).score
    b4 = bleu4.corpus_score(predictions, [references]).score

    # =========================
    # ROUGE-L
    # =========================
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_scores = []
    for p, r in zip(predictions, references):
        rouge_scores.append(
            scorer.score(r, p)["rougeL"].fmeasure
        )

    rougeL = sum(rouge_scores) / len(rouge_scores) * 100

    # =========================
    # PRINT
    # =========================
    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"BLEU-1 : {b1:.2f}")
    print(f"BLEU-2 : {b2:.2f}")
    print(f"BLEU-3 : {b3:.2f}")
    print(f"BLEU-4 : {b4:.2f}")
    print(f"ROUGE-L: {rougeL:.2f}")

def main():
    print("Device:", DEVICE)

    train_df = pd.read_csv(TRAIN_CSV, sep="\t")

    tokenizer = Tokenizer()
    tokenizer.build_vocab(train_df["translation"].tolist())

    bos_id = tokenizer.word2idx["<bos>"]
    eos_id = tokenizer.word2idx["<eos>"]

    print("Vocab size:", tokenizer.vocab_size)
    print("BOS:", bos_id)
    print("EOS:", eos_id)

    test_ds = How2SignDataset(
        TEST_CSV,
        tokenizer,
        base_mp=BASE_MP_TEST,
        base_i3d=BASE_I3D_TEST
    )

    # use 10% ds
    n = len(test_ds)
    subset_size = int(0.01 * n)

    indices = random.sample(range(n), subset_size)

    test_ds = Subset(test_ds, indices)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    sample_i3d, sample_mp, _ = test_ds[0]

    model = SignTranslator(
        sample_i3d.shape[1],
        sample_mp.shape[1] * sample_mp.shape[2],
        tokenizer.vocab_size
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    print("Model loaded.\n")

    predictions = []
    references = []

    for idx, (i3d, mp, txt) in enumerate(tqdm(test_loader)):
        i3d = i3d.to(DEVICE)
        mp = mp.to(DEVICE)

        pred_ids = generate(
            model,
            i3d,
            mp,
            bos_id=bos_id,
            eos_id=eos_id
        )

        pred_text = tokenizer.decode(pred_ids).strip()
        gt_text = tokenizer.decode(txt[0].tolist()).strip()

        predictions.append(pred_text)
        references.append(gt_text)

        # show first 20 examples
        if idx < 20:
            print("=" * 60)
            print("Sample:", idx + 1)
            print("GT  :", gt_text)
            print("PRED:", pred_text)

    evaluate_metrics(predictions, references)


if __name__ == "__main__":
    main()