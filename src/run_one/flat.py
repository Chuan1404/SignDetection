import os
import json

from config import ROOT

WLASL_JSON_PATH = os.path.join(
    ROOT,
    "datasets",
    "annotations",
    "WLASL_v0_3.json"
)

MISSING_PATH = os.path.join(
    ROOT,
    "datasets",
    "annotations",
    "missing.txt"
)

SAVE_DIR = os.path.join(
    ROOT,
    "datasets",
    "annotations"
)

os.makedirs(SAVE_DIR, exist_ok=True)


def flatten_wlasl():

    with open(WLASL_JSON_PATH, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    with open(MISSING_PATH, "r", encoding="utf-8") as f:
        missing_ids = set(f.read().split())

    train_entries = []
    val_entries = []
    test_entries = []

    skipped = 0

    all_glosses = set()

    for gloss_entry in wlasl_data:

        gloss = gloss_entry["gloss"]
        all_glosses.add(gloss)

        for instance in gloss_entry["instances"]:

            video_id = instance["video_id"]

            if video_id in missing_ids:
                skipped += 1
                continue

            sample = {
                "video_id": video_id,
                "gloss": gloss
            }

            split = instance["split"].lower()

            if split == "train" or split == "val":
                train_entries.append(sample)

            elif split == "test":
                test_entries.append(sample)

            else:
                print(f"Unknown split: {split}")

    # -------------------------------------------------
    # Build vocabulary
    # -------------------------------------------------

    gloss_list = sorted(all_glosses)

    gloss2idx = {
        gloss: idx
        for idx, gloss in enumerate(gloss_list)
    }

    idx2gloss = {
        idx: gloss
        for gloss, idx in gloss2idx.items()
    }

    # -------------------------------------------------
    # Save annotation
    # -------------------------------------------------

    with open(os.path.join(SAVE_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2, ensure_ascii=False)

    # with open(os.path.join(SAVE_DIR, "val.json"), "w", encoding="utf-8") as f:
    #     json.dump(val_entries, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_entries, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "gloss2idx.json"), "w", encoding="utf-8") as f:
        json.dump(gloss2idx, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "idx2gloss.json"), "w", encoding="utf-8") as f:
        json.dump(idx2gloss, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("WLASL flattened successfully")
    print("=" * 60)
    print(f"Glosses        : {len(gloss_list)}")
    print(f"Missing videos : {skipped}")
    print(f"Train samples  : {len(train_entries)}")
    # print(f"Val samples    : {len(val_entries)}")
    print(f"Test samples   : {len(test_entries)}")
    print(f"Vocabulary     : {len(gloss2idx)}")
    print("=" * 60)


if __name__ == "__main__":
    flatten_wlasl()