import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

SAVE_DIR: str = "./ade20k_files"
# The official mirror (zhoubolei/scene_parse_150) ships a loading script, which the
# Dataset Viewer cannot serve; this mirror holds the same 20210/2000 image split.
DATASET: str = "sezer12138/ADE20k_Segementation"
SEED: int = 42


def download_ade20k_images(num_samples: int) -> int:
    """Save num_samples ADE20K validation RGB images into SAVE_DIR. Returns count saved.

    Calibration needs pixels only, so the segmentation labels are dropped.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = load_dataset(DATASET, split="val").shuffle(seed=SEED)

    saved = 0
    for sample in tqdm(dataset, total=num_samples, desc=os.path.basename(SAVE_DIR), unit="img"):
        sample["image"].convert("RGB").save(os.path.join(SAVE_DIR, f"ade20k_{saved:04d}.jpg"))
        saved += 1
        if saved >= num_samples:
            break

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ADE20K calibration images.")
    parser.add_argument("--num-samples", type=int, default=100, help="number of images to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = download_ade20k_images(args.num_samples)
    print(f"Done: {count}/{args.num_samples} images in {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
