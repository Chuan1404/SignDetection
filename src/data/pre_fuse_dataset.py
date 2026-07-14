import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.fusion_component import FusionComponent
from config import ROOT


def main():
    feature_dir = os.path.join(ROOT, "datasets", "processed", "full_body_wlasl")
    print(f"Scanning features directory: {feature_dir}")

    if not os.path.exists(feature_dir):
        print(f"Error: Directory {feature_dir} does not exist.")
        return

    video_ids = [d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))]
    print(f"Found {len(video_ids)} video directories.")

    fusion_component = FusionComponent()
    fused_count = 0
    skipped_count = 0

    for video_id in tqdm(video_ids, desc="Pre-fusing dataset"):
        video_dir = os.path.join(feature_dir, video_id)

        left_path = os.path.join(video_dir, "left_hand.npy")
        right_path = os.path.join(video_dir, "right_hand.npy")
        pose_path = os.path.join(video_dir, "pose.npy")
        fused_path = os.path.join(video_dir, "fused.npy")

        # Check if already fused
        if os.path.exists(fused_path):
            skipped_count += 1
            continue

        # Check if raw files exist
        if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(pose_path):
            try:
                left_features = np.load(left_path).astype(np.float32)
                right_features = np.load(right_path).astype(np.float32)
                pose_features = np.load(pose_path).astype(np.float32)

                hand_features = np.concatenate([left_features, right_features], axis=-1)

                fused = fusion_component.fuse(pose_features, hand_features)

                # Verify shape is (T, 185)
                T = fused.shape[0]
                fused = fused.reshape(T, -1)
                assert fused.shape[1] == 185, f"Expected 185 features, got {fused.shape[1]}"

                np.save(fused_path, fused)
                fused_count += 1
            except Exception as e:
                print(f"\nError processing {video_id}: {e}")
        else:
            skipped_count += 1

    print("\nPre-fusion completed!")
    print(f"  - Newly fused : {fused_count} samples")
    print(f"  - Skipped/Exist: {skipped_count} samples")


if __name__ == "__main__":
    main()
