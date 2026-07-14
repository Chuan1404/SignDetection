"""
debug.py — Kiểm tra các giả thuyết gây mode collapse

Chạy: python debug.py

Các giả thuyết kiểm tra theo thứ tự:
  [1] Encoder output có bị zero/explode không?
  [2] Decoder có đang ignore encoder không? (random noise test)
  [3] Checkpoint val_loss lưu có đúng không?
  [4] attention_mask có được truyền vào generate() không?
  [5] Loss curve — model có thực sự học không?
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

from src.utils import FusionComponent
from src.data.hand_landmarks import HandLandmarksDataset
from src.models.SLT_model import SignLanguageTranslatorV2
from config import ROOT

# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

MODEL_PATH  = os.path.join(ROOT, "outputs", "models", "2026_07_01_best.pt")
FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_how2sign")

fusion_component = FusionComponent()

PASS = "✅ PASS"
FAIL = "❌ FAIL"
INFO = "ℹ️  INFO"

# ─────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────
def collate_fn(batch, tokenizer):
    features, texts = [], []
    for feature, text in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        texts.append(text)
    real_lengths = [f.shape[0] for f in features]
    features = pad_sequence(features, batch_first=True)
    texts = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()
    labels = texts.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return features, labels, video_mask

# ─────────────────────────────────────────────
def load_model_and_data():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

    dataset = HandLandmarksDataset(FEATURE_DIR, tokenizer, fusion_component, max_samples=200)
    test_dataset = Subset(dataset, range(int(len(dataset) * 0.9), len(dataset)))

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    feature, _ = dataset[0]
    model = SignLanguageTranslatorV2(input_dim=feature.shape[-1]).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"{FAIL} Không tìm thấy checkpoint: {MODEL_PATH}")
        return None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    result = model.load_state_dict(checkpoint["model"], strict=False)

    if result.missing_keys:
        print(f"{INFO} Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"{INFO} Unexpected keys: {result.unexpected_keys}")

    model.eval()
    return model, loader, tokenizer, checkpoint

# ─────────────────────────────────────────────
def check_checkpoint(checkpoint):
    """[3] Checkpoint có lưu đúng val_loss không?"""
    section("GIẢ THUYẾT 3 — Checkpoint info")

    epoch    = checkpoint.get("epoch", "N/A")
    val_loss = checkpoint.get("val_loss", "N/A")

    print(f"  Epoch saved : {epoch}")
    print(f"  Val loss    : {val_loss}")

    if isinstance(val_loss, float) and val_loss < 10.0:
        print(f"  {PASS} Val loss hợp lý")
    else:
        print(f"  {FAIL} Val loss bất thường — checkpoint có thể bị save sai")

# ─────────────────────────────────────────────
def check_encoder_output(model, loader):
    """[1] Encoder output có bị zero hoặc explode không?"""
    section("GIẢ THUYẾT 1 — Encoder output norm")

    features, _, video_mask = next(iter(loader))
    features   = features.to(DEVICE)
    video_mask = video_mask.to(DEVICE)

    with torch.no_grad():
        enc = model.encode(features, video_mask)

    norms = enc.norm(dim=-1)  # (B, T)
    mean_norm = norms.mean().item()
    max_norm  = norms.max().item()
    min_norm  = norms.min().item()

    print(f"  Encoder output norm — mean: {mean_norm:.4f} | min: {min_norm:.4f} | max: {max_norm:.4f}")

    if mean_norm < 0.01:
        print(f"  {FAIL} Norm gần 0 → encoder output bị vanish, projection không học được")
    elif mean_norm > 100:
        print(f"  {FAIL} Norm quá lớn → encoder output explode, cần check LayerNorm")
    else:
        print(f"  {PASS} Norm trong khoảng bình thường")

# ─────────────────────────────────────────────
def check_encoder_ignored(model, loader, tokenizer):
    """[2] Decoder có đang ignore encoder không?"""
    section("GIẢ THUYẾT 2 — Encoder bị ignore (random noise test)")

    features, _, video_mask = next(iter(loader))
    features   = features.to(DEVICE)
    video_mask = video_mask.to(DEVICE)

    with torch.no_grad():
        # Generate với features thật
        real_ids = model.generate(features, video_mask=video_mask, max_length=32, num_beams=4)
        real_preds = tokenizer.batch_decode(real_ids, skip_special_tokens=True)

        # Generate với random noise cùng shape
        noise = torch.randn_like(features)
        noise_ids = model.generate(noise, video_mask=video_mask, max_length=32, num_beams=4)
        noise_preds = tokenizer.batch_decode(noise_ids, skip_special_tokens=True)

    print("\n  So sánh real features vs random noise:")
    all_same = True
    for i, (r, n) in enumerate(zip(real_preds[:3], noise_preds[:3])):
        same = (r.strip() == n.strip())
        if not same:
            all_same = False
        print(f"\n  Sample {i+1}:")
        print(f"    Real  : {r}")
        print(f"    Noise : {n}")
        print(f"    Same? : {'⚠️  YES' if same else '✓  NO'}")

    if all_same:
        print(f"\n  {FAIL} Tất cả output giống nhau → decoder đang IGNORE encoder hoàn toàn")
    else:
        print(f"\n  {PASS} Output khác nhau → encoder có ảnh hưởng đến decoder")

# ─────────────────────────────────────────────
def check_attention_mask(model, loader, tokenizer):
    """[4] attention_mask có thực sự ảnh hưởng đến output không?"""
    section("GIẢ THUYẾT 4 — attention_mask effect")

    features, _, video_mask = next(iter(loader))
    features   = features.to(DEVICE)
    video_mask = video_mask.to(DEVICE)

    with torch.no_grad():
        # Generate có mask
        ids_with_mask = model.generate(
            features, video_mask=video_mask, max_length=32, num_beams=4
        )
        # Generate không có mask (all ones)
        all_ones = torch.ones_like(video_mask)
        ids_no_mask = model.generate(
            features, video_mask=all_ones, max_length=32, num_beams=4
        )

    pred_with = tokenizer.batch_decode(ids_with_mask, skip_special_tokens=True)
    pred_none = tokenizer.batch_decode(ids_no_mask, skip_special_tokens=True)

    diff_count = sum(1 for a, b in zip(pred_with, pred_none) if a.strip() != b.strip())
    print(f"  Số sample output khác nhau khi có/không mask: {diff_count}/{BATCH_SIZE}")

    if diff_count == 0:
        print(f"  {FAIL} attention_mask không có tác dụng — có thể không được truyền đúng")
    else:
        print(f"  {PASS} attention_mask ảnh hưởng đến output")

# ─────────────────────────────────────────────
def check_loss_on_batch(model, loader):
    """[5] Loss trên một batch — model có học không?"""
    section("GIẢ THUYẾT 5 — Loss trên real data vs random label")

    features, text_ids, video_mask = next(iter(loader))
    features   = features.to(DEVICE)
    text_ids   = text_ids.to(DEVICE)
    video_mask = video_mask.to(DEVICE)

    with torch.no_grad():
        # Loss với label thật
        out_real = model(features, text_ids=text_ids, video_mask=video_mask)
        loss_real = out_real.loss.item()

        # Loss với label random (shuffle)
        shuffled = text_ids[torch.randperm(text_ids.size(0))]
        out_rand = model(features, text_ids=shuffled, video_mask=video_mask)
        loss_rand = out_rand.loss.item()

    print(f"  Loss (real labels)    : {loss_real:.4f}")
    print(f"  Loss (random labels)  : {loss_rand:.4f}")

    if loss_real < loss_rand:
        print(f"  {PASS} Loss real < loss random → model đã học được signal từ data")
    else:
        print(f"  {FAIL} Loss real >= loss random → model chưa học được gì, gần như random")

# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DEBUG — Mode Collapse Investigation")
    print("="*60)
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {MODEL_PATH}")

    result = load_model_and_data()
    if result[0] is None:
        return

    model, loader, tokenizer, checkpoint = result

    check_checkpoint(checkpoint)
    check_encoder_output(model, loader)
    check_encoder_ignored(model, loader, tokenizer)
    check_attention_mask(model, loader, tokenizer)
    check_loss_on_batch(model, loader)

    section("TỔNG KẾT")
    print("""
  Đọc kết quả theo thứ tự:
  - GT3 val_loss bất thường     → checkpoint bị save sai
  - GT1 norm ~0 hoặc >100       → projection/LayerNorm vấn đề
  - GT2 real == noise           → decoder ignore encoder hoàn toàn
  - GT4 mask không ảnh hưởng    → attention_mask không được dùng
  - GT5 loss_real >= loss_rand  → model chưa học được gì
    """)

if __name__ == "__main__":
    main()