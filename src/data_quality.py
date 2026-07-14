"""
data_quality.py — Kiểm tra chất lượng dataset WLASL (landmarks: left_hand/right_hand/pose + text)

Chạy: python data_quality.py

Các mục kiểm tra:
  [0] Load diagnostics   — bao nhiêu sample load được / bị skip và vì sao
  [1] Missing rate       — bao nhiêu % frame bị thiếu (tay không detect)
  [2] Variance           — gesture có phong phú không hay người đứng yên
  [3] Outlier rate       — bao nhiêu frame có giá trị bất thường
  [4] Video/Text ratio   — video dài mà text ngắn → misalign
  [5] Text diversity     — dataset có bị dominated bởi một số câu không
  [6] Sequence length    — phân phối độ dài video và text
  [7] Topic bias         — một số pattern câu chiếm quá nhiều không
"""

import os
import re
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from collections import Counter
from functools import partial

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.utils import FusionComponent
from src.data.hand_landmarks import HandLandmarksDataset
from config import ROOT

# ─────────────────────────────────────────────
FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_wlasl")
MAX_SAMPLES  = None   # None = toàn bộ dataset

PASS = "✅ PASS"
WARN = "⚠️  WARN"
FAIL = "❌ FAIL"

fusion_component = FusionComponent()

# ─────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def rate_label(value, warn_thresh, fail_thresh, low_is_bad=False):
    """Trả về PASS/WARN/FAIL dựa trên ngưỡng"""
    if low_is_bad:
        if value < fail_thresh: return FAIL
        if value < warn_thresh: return WARN
        return PASS
    else:
        if value > fail_thresh: return FAIL
        if value > warn_thresh: return WARN
        return PASS

# ─────────────────────────────────────────────
def load_raw_samples(feature_dir, max_samples):
    """Load trực tiếp từ file .npy và .txt, không qua Dataset class.
    Trả về (samples, diagnostics) — diagnostics giúp biết vì sao sample bị skip.
    """
    samples = []
    diag = {
        "dir_exists":       os.path.isdir(feature_dir),
        "total_entries":    0,
        "not_a_dir":        0,
        "missing_files":    0,
        "load_error":       0,
        "empty_features":   0,
        "loaded":           0,
        "missing_files_examples": [],
        "load_error_examples":    [],
    }

    if not diag["dir_exists"]:
        return samples, diag

    all_names = sorted(os.listdir(feature_dir))
    diag["total_entries"] = len(all_names)

    for i, name in enumerate(tqdm(all_names, desc="Loading samples")):
        if max_samples is not None and i >= max_samples:
            break

        video_dir = os.path.join(feature_dir, name)
        if not os.path.isdir(video_dir):
            diag["not_a_dir"] += 1
            continue

        lh  = os.path.join(video_dir, "left_hand.npy")
        rh  = os.path.join(video_dir, "right_hand.npy")
        ps  = os.path.join(video_dir, "pose.npy")
        txt = os.path.join(video_dir, "gloss.txt")

        missing = [p for p in [lh, rh, ps, txt] if not os.path.exists(p)]
        if missing:
            diag["missing_files"] += 1
            if len(diag["missing_files_examples"]) < 5:
                diag["missing_files_examples"].append(
                    (name, [os.path.basename(p) for p in missing])
                )
            continue

        try:
            left  = np.load(lh)
            right = np.load(rh)
            pose  = np.load(ps)

            with open(txt, "r", encoding="utf-8") as f:
                text = f.read().strip()

            hand = np.concatenate([left, right], axis=-1)  # (T, 126)
            features = np.concatenate([hand, pose], axis=-1) if pose.shape[-1] > 0 else hand
        except Exception as e:
            diag["load_error"] += 1
            if len(diag["load_error_examples"]) < 5:
                diag["load_error_examples"].append((name, str(e)))
            continue

        if features.shape[0] == 0 or features.shape[-1] == 0:
            diag["empty_features"] += 1
            continue

        samples.append({
            "name":     name,
            "features": features,   # (T, D)
            "text":     text,
            "T":        features.shape[0],
            "D":        features.shape[-1],
        })

    diag["loaded"] = len(samples)
    return samples, diag


def print_load_diagnostics(feature_dir, diag):
    """[0] In ra chi tiết vì sao sample bị load / bị skip"""
    section("KIỂM TRA 0 — Load diagnostics")

    print(f"  Feature dir       : {feature_dir}")
    print(f"  Dir tồn tại        : {diag['dir_exists']}")

    if not diag["dir_exists"]:
        print(f"\n  {FAIL} Thư mục không tồn tại — kiểm tra lại config ROOT / FEATURE_DIR")
        return

    print(f"  Tổng entries       : {diag['total_entries']}")
    print(f"  Loaded thành công  : {diag['loaded']}")
    print(f"  Skip (không phải dir)      : {diag['not_a_dir']}")
    print(f"  Skip (thiếu file)          : {diag['missing_files']}")
    print(f"  Skip (lỗi load .npy/.txt)  : {diag['load_error']}")
    print(f"  Skip (feature rỗng)        : {diag['empty_features']}")

    if diag["missing_files_examples"]:
        print(f"\n  Ví dụ sample thiếu file:")
        for name, missing in diag["missing_files_examples"]:
            print(f"    {name}: thiếu {missing}")

    if diag["load_error_examples"]:
        print(f"\n  Ví dụ sample load lỗi:")
        for name, err in diag["load_error_examples"]:
            print(f"    {name}: {err}")

    if diag["loaded"] == 0:
        print(f"\n  {FAIL} Không load được sample nào — dừng kiểm tra, xem chi tiết ở trên.")
    elif diag["loaded"] < diag["total_entries"] * 0.5:
        print(f"\n  {WARN} Hơn 50% entries bị skip — dataset có thể lỗi preprocessing diện rộng.")
    else:
        print(f"\n  {PASS} Load dataset ổn.")

# ─────────────────────────────────────────────
def check_missing_rate(samples):
    """[1] Bao nhiêu % frame bị all-zero (tay không detect được)"""
    section("KIỂM TRA 1 — Missing rate (frame all-zero)")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    missing_rates = []
    worst = []

    for s in samples:
        f = s["features"]
        is_missing = (f == 0).all(axis=-1)   # (T,)
        rate = is_missing.mean()
        missing_rates.append(rate)
        if rate > 0.3:
            worst.append((s["name"], rate))

    mean_missing = np.mean(missing_rates)
    max_missing  = np.max(missing_rates)
    pct_high     = np.mean(np.array(missing_rates) > 0.3) * 100

    label = rate_label(mean_missing, warn_thresh=0.1, fail_thresh=0.3)
    print(f"  Mean missing rate : {mean_missing*100:.2f}%  {label}")
    print(f"  Max  missing rate : {max_missing*100:.2f}%")
    print(f"  Samples > 30% missing : {pct_high:.1f}%")

    if worst[:5]:
        print(f"\n  Top samples bị missing nhiều nhất:")
        for name, rate in sorted(worst, key=lambda x: -x[1])[:5]:
            print(f"    {name}: {rate*100:.1f}%")

# ─────────────────────────────────────────────
def check_variance(samples):
    """[2] Variance theo thời gian — gesture có phong phú không"""
    section("KIỂM TRA 2 — Temporal variance (gesture phong phú)")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    variances = []
    low_var_samples = []

    for s in samples:
        f  = s["features"].astype(np.float32)
        var = f.var(axis=0).mean()   # variance trung bình theo time axis
        variances.append(var)
        if var < 0.001:
            low_var_samples.append((s["name"], var))

    mean_var = np.mean(variances)
    pct_low  = len(low_var_samples) / len(samples) * 100

    label = rate_label(mean_var, warn_thresh=0.005, fail_thresh=0.001, low_is_bad=True)
    print(f"  Mean temporal variance : {mean_var:.6f}  {label}")
    print(f"  Samples có variance thấp (<0.001) : {pct_low:.1f}%")

    if pct_low > 5:
        print(f"\n  {WARN} {pct_low:.1f}% samples gần như không có chuyển động")
        for name, var in sorted(low_var_samples, key=lambda x: x[1])[:5]:
            print(f"    {name}: var={var:.6f}")

# ─────────────────────────────────────────────
def check_outliers(samples):
    """[3] Outlier rate — bao nhiêu frame có giá trị bất thường"""
    section("KIỂM TRA 3 — Outlier rate (z-score > 3)")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    outlier_rates = []

    # Tính global mean và std trên 500 samples đầu để ước tính
    all_features = []
    for s in samples[:500]:
        all_features.append(s["features"])
    combined = np.concatenate(all_features, axis=0).astype(np.float32)
    global_mean = combined.mean(axis=0)
    global_std  = combined.std(axis=0) + 1e-6

    for s in samples:
        f = s["features"].astype(np.float32)
        z = np.abs((f - global_mean) / global_std)
        outlier_rate = (z > 3).mean()
        outlier_rates.append(outlier_rate)

    mean_outlier = np.mean(outlier_rates)
    label = rate_label(mean_outlier, warn_thresh=0.05, fail_thresh=0.15)
    print(f"  Mean outlier rate : {mean_outlier*100:.2f}%  {label}")
    print(f"  (Tỉ lệ frame/dim có |z-score| > 3)")

    if mean_outlier > 0.05:
        print(f"  {WARN} Data có nhiều outlier — normalize_features cần xem lại")

# ─────────────────────────────────────────────
def check_length_ratio(samples, tokenizer):
    """[4] Tỉ lệ độ dài video / text — phát hiện misalignment"""
    section("KIỂM TRA 4 — Video/Text length ratio")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    ratios   = []
    v_lens   = []
    t_lens   = []
    misalign = []

    for s in samples:
        tokens = tokenizer(s["text"], return_tensors="pt")
        t_len  = tokens["input_ids"].shape[-1]
        v_len  = s["T"]
        ratio  = v_len / max(t_len, 1)

        ratios.append(ratio)
        v_lens.append(v_len)
        t_lens.append(t_len)

        # Video quá dài hoặc quá ngắn so với text → misalign
        if ratio > 100 or ratio < 2:
            misalign.append((s["name"], v_len, t_len, ratio))

    mean_ratio = np.mean(ratios)
    p5   = np.percentile(ratios, 5)
    p95  = np.percentile(ratios, 95)
    pct_misalign = len(misalign) / len(samples) * 100

    print(f"  Mean ratio (frames/tokens) : {mean_ratio:.1f}")
    print(f"  P5 / P95                   : {p5:.1f} / {p95:.1f}")
    print(f"  Mean video length (frames) : {np.mean(v_lens):.0f}")
    print(f"  Mean text length (tokens)  : {np.mean(t_lens):.0f}")
    print(f"  Samples bị misalign        : {pct_misalign:.1f}%  "
          f"{'  ' + FAIL if pct_misalign > 5 else PASS}")

    if misalign[:3]:
        print(f"\n  Top misaligned samples:")
        for name, vl, tl, r in sorted(misalign, key=lambda x: -abs(x[3]-mean_ratio))[:5]:
            print(f"    {name}: {vl} frames / {tl} tokens → ratio={r:.1f}")

# ─────────────────────────────────────────────
def check_text_diversity(samples):
    """[5] Text diversity — dataset có bị dominated bởi một số câu/cụm từ không"""
    section("KIỂM TRA 5 — Text diversity (bigram/trigram distribution)")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    all_texts  = [s["text"].lower().strip() for s in samples]
    all_words  = []
    bigrams    = []
    trigrams   = []

    for text in all_texts:
        words = re.findall(r'\b\w+\b', text)
        all_words.extend(words)
        bigrams.extend(zip(words, words[1:]))
        trigrams.extend(zip(words, words[1:], words[2:]))

    # Unigram
    uni_counter  = Counter(all_words)
    top5_uni     = uni_counter.most_common(5)
    total_words  = len(all_words)

    # Bigram
    bi_counter  = Counter(bigrams)
    top5_bi     = bi_counter.most_common(5)
    total_bi    = len(bigrams)

    # Trigram
    tri_counter = Counter(trigrams)
    top5_tri    = tri_counter.most_common(5)
    total_tri   = len(trigrams)

    if total_words == 0:
        print(f"  {WARN} Toàn bộ text rỗng — bỏ qua phần diversity.")
        return

    # Type-Token Ratio (TTR) — đo độ phong phú từ vựng
    vocab_size = len(uni_counter)
    ttr        = vocab_size / total_words

    print(f"\n  Vocabulary size : {vocab_size} unique words")
    print(f"  Type-Token Ratio (TTR) : {ttr:.4f}  "
          f"({'good' if ttr > 0.05 else 'low — text bị lặp nhiều'})")

    print(f"\n  Top 5 unigrams:")
    for word, cnt in top5_uni:
        print(f"    '{word}': {cnt} ({cnt/total_words*100:.1f}%)")

    print(f"\n  Top 5 bigrams:")
    if total_bi == 0:
        print(f"    (không có bigram — text quá ngắn)")
    else:
        for bg, cnt in top5_bi:
            pct = cnt / total_bi * 100
            label = WARN if pct > 3 else ""
            print(f"    '{' '.join(bg)}': {cnt} ({pct:.1f}%)  {label}")

    print(f"\n  Top 5 trigrams:")
    if total_tri == 0:
        print(f"    (không có trigram — text quá ngắn)")
    else:
        for tg, cnt in top5_tri:
            pct = cnt / total_tri * 100
            label = WARN if pct > 2 else ""
            print(f"    '{' '.join(tg)}': {cnt} ({pct:.1f}%)  {label}")

    # Exact duplicate sentences
    text_counter = Counter(all_texts)
    duplicates   = {t: c for t, c in text_counter.items() if c > 1}
    dup_pct      = sum(duplicates.values()) / len(all_texts) * 100

    print(f"\n  Exact duplicate sentences : {len(duplicates)} unique, "
          f"{dup_pct:.1f}% of dataset  "
          f"{WARN if dup_pct > 10 else PASS}")

    if duplicates:
        print(f"\n  Top 5 câu bị lặp nhiều nhất:")
        for text, cnt in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{cnt}x] \"{text[:80]}\"")

# ─────────────────────────────────────────────
def check_sequence_length(samples):
    """[6] Phân phối độ dài video — phát hiện outlier quá dài/ngắn"""
    section("KIỂM TRA 6 — Sequence length distribution")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    lengths = np.array([s["T"] for s in samples])

    p5   = np.percentile(lengths, 5)
    p25  = np.percentile(lengths, 25)
    p50  = np.percentile(lengths, 50)
    p75  = np.percentile(lengths, 75)
    p95  = np.percentile(lengths, 95)
    pmax = lengths.max()
    pmin = lengths.min()

    print(f"  Min    : {pmin}")
    print(f"  P5     : {p5:.0f}")
    print(f"  P25    : {p25:.0f}")
    print(f"  Median : {p50:.0f}")
    print(f"  P75    : {p75:.0f}")
    print(f"  P95    : {p95:.0f}")
    print(f"  Max    : {pmax}")

    # Cảnh báo nếu có sequence quá dài (gây OOM khi train)
    very_long = (lengths > 1000).mean() * 100
    very_short = (lengths < 10).mean() * 100

    print(f"\n  Sequences > 1000 frames : {very_long:.1f}%  "
          f"{WARN if very_long > 5 else PASS}")
    print(f"  Sequences < 10  frames  : {very_short:.1f}%  "
          f"{WARN if very_short > 1 else PASS}")

# ─────────────────────────────────────────────
def check_topic_bias(samples):
    """[7] Topic bias — một số opening pattern chiếm quá nhiều không"""
    section("KIỂM TRA 7 — Topic/Pattern bias")

    if not samples:
        print(f"  {WARN} Không có sample nào để kiểm tra — bỏ qua.")
        return

    # Lấy 4 từ đầu của mỗi câu làm "opening pattern"
    openings = []
    for s in samples:
        words = re.findall(r'\b\w+\b', s["text"].lower())
        opening = " ".join(words[:4]) if len(words) >= 4 else " ".join(words)
        openings.append(opening)

    counter  = Counter(openings)
    total    = len(openings)
    top10    = counter.most_common(10)

    top1_pct = top10[0][1] / total * 100 if top10 else 0

    print(f"  Unique opening patterns : {len(counter)} / {total} samples")
    print(f"\n  Top 10 opening patterns:")
    for pattern, cnt in top10:
        pct   = cnt / total * 100
        label = FAIL if pct > 5 else (WARN if pct > 2 else "")
        print(f"    [{cnt:4d} | {pct:5.1f}%] \"{pattern}\"  {label}")

    if top1_pct > 5:
        print(f"\n  {FAIL} Dataset bị bias nặng — top pattern chiếm {top1_pct:.1f}%")
        print(f"         Đây là nguyên nhân model luôn generate câu giống nhau")
    elif top1_pct > 2:
        print(f"\n  {WARN} Dataset có bias nhẹ — cần data augmentation")
    else:
        print(f"\n  {PASS} Pattern distribution tương đối đều")

# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DATA QUALITY — WLASL Dataset")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

    print(f"\nLoading dataset (max_samples={MAX_SAMPLES})...")
    samples, diag = load_raw_samples(FEATURE_DIR, MAX_SAMPLES)
    print(f"Loaded: {len(samples)} samples")

    print_load_diagnostics(FEATURE_DIR, diag)

    if not samples:
        print(f"\n{FAIL} Dừng kiểm tra — không có sample nào để phân tích.")
        print("  Kiểm tra lại:")
        print(f"    1. FEATURE_DIR đúng chưa: {FEATURE_DIR}")
        print("    2. Mỗi video folder có đủ left_hand.npy / right_hand.npy / pose.npy / text.txt chưa")
        print("    3. Bước preprocessing đã chạy xong và ghi ra đúng thư mục chưa")
        sys.exit(1)

    check_missing_rate(samples)
    check_variance(samples)
    check_outliers(samples)
    check_length_ratio(samples, tokenizer)
    check_text_diversity(samples)
    check_sequence_length(samples)
    check_topic_bias(samples)

    section("TỔNG KẾT")
    print("""
  Đọc kết quả để ưu tiên fix:

  FAIL missing rate > 30%   → tay không được detect, cần filter hoặc interpolate
  FAIL variance thấp        → video không có gesture, cần filter
  FAIL outlier rate > 15%   → normalize_features có vấn đề
  FAIL misalign > 5%        → video và text không khớp nhau
  WARN bigram > 3%          → text bị lặp → model học shortcut
  WARN duplicate > 10%      → cần deduplicate dataset
  FAIL topic bias > 5%      → đây là nguyên nhân chính của mode collapse
    """)

if __name__ == "__main__":
    main()