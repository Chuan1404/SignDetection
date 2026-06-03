import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer

from src.models.SLT_model import SignLanguageTranslator


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. BASIC SHAPE + ENCODER TEST
# =========================================================
def test_shape_and_encoder(model, tokenizer, dataloader):
    model.eval()

    batch = next(iter(dataloader))
    hand_features, text_ids, video_mask = batch

    print("\n===== SHAPE CHECK =====")
    print("hand_features:", hand_features.shape)
    print("text_ids:", text_ids.shape)
    print("video_mask:", video_mask.shape)

    # ensure correct shape
    if video_mask.dim() == 1:
        video_mask = video_mask.unsqueeze(0)

    hand_features = hand_features.to(DEVICE)
    video_mask = video_mask.to(DEVICE)

    with torch.no_grad():
        encoder_out = model.encode(hand_features, video_mask)

    print("\n===== ENCODER OUTPUT =====")
    print("encoder_out:", encoder_out.shape)

    assert encoder_out.dim() == 3, "Encoder must be (B, T, C)"
    assert encoder_out.shape[0] == hand_features.shape[0]
    assert encoder_out.shape[1] == hand_features.shape[1]

    return hand_features, video_mask, encoder_out


# =========================================================
# 2. MODE COLLAPSE TEST
# =========================================================
def test_mode_collapse(model, tokenizer, hand_features, video_mask):
    model.eval()

    print("\n===== MODE COLLAPSE TEST =====")

    outputs = []

    with torch.no_grad():
        for i in range(3):
            out = model.generate(
                hand_features,
                video_mask,
                max_length=40
            )

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            outputs.append(decoded)

            print(f"Run {i}: {decoded}")

    unique = set(outputs)

    print("\n===== COLLAPSE RESULT =====")
    print("unique outputs:", len(unique))

    if len(unique) == 1:
        print("❌ MODE COLLAPSED (same output always)")
    else:
        print("✅ NOT collapsed")


# =========================================================
# 3. ENCODER SENSITIVITY TEST
# =========================================================
def test_encoder_sensitivity(model, tokenizer, hand_features, video_mask):
    model.eval()

    print("\n===== ENCODER SENSITIVITY TEST =====")

    with torch.no_grad():
        out1 = model.generate(hand_features, video_mask)

        shuffled = hand_features[torch.randperm(hand_features.size(0))]

        out2 = model.generate(shuffled, video_mask)

    print("Original:", tokenizer.decode(out1[0], skip_special_tokens=True))
    print("Shuffled:", tokenizer.decode(out2[0], skip_special_tokens=True))


# =========================================================
# 4. ZERO ENCODER TEST (VERY IMPORTANT)
# =========================================================
def test_zero_encoder(model, tokenizer, hand_features, video_mask):
    model.eval()

    print("\n===== ZERO ENCODER TEST =====")

    with torch.no_grad():
        real_enc = model.encode(hand_features, video_mask)
        zero_enc = torch.zeros_like(real_enc)

        out_real = model.mt5.generate(
            encoder_outputs=real_enc,
            attention_mask=video_mask,
            max_length=40
        )

        out_zero = model.mt5.generate(
            encoder_outputs=zero_enc,
            attention_mask=video_mask,
            max_length=40
        )

    print("REAL:", tokenizer.decode(out_real[0], skip_special_tokens=True))
    print("ZERO:", tokenizer.decode(out_zero[0], skip_special_tokens=True))


# =========================================================
# 5. MAIN RUNNER
# =========================================================
def run_all_tests(model, tokenizer, dataloader):

    hand_features, video_mask, encoder_out = test_shape_and_encoder(
        model, tokenizer, dataloader
    )

    test_mode_collapse(model, tokenizer, hand_features, video_mask)

    test_encoder_sensitivity(model, tokenizer, hand_features, video_mask)

    test_zero_encoder(model, tokenizer, hand_features, video_mask)


# =========================================================
# 6. ENTRY POINT
# =========================================================
if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "google/mt5-small",
        use_fast=False
    )

    model = SignLanguageTranslator().to(DEVICE)

    print("Model loaded OK")

    # ⚠️ YOU MUST PROVIDE YOUR DATALOADER HERE
    # from your training script
    #
    # run_all_tests(model, tokenizer, train_loader)