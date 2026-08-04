import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

SAVE_DIR: str = "./imagenet_files"
DATASET: str = "ILSVRC/imagenet-1k"
DATA_FILES: dict[str, str] = {"validation": "data/validation*.parquet"}
SEED: int = 42


def download_imagenet_images(num_samples: int) -> int:
    """Save num_samples ImageNet-1k validation images into SAVE_DIR. Returns count saved.

    One image per label first, so a small calibration set still spans many classes.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = load_dataset(
        DATASET,
        data_files=DATA_FILES,
        split="validation",
        verification_mode="no_checks",  # the parquet subset has no checksum manifest
    ).shuffle(seed=SEED)

    seen: set[int] = set()
    saved = 0
    for sample in tqdm(dataset, total=num_samples, desc=os.path.basename(SAVE_DIR), unit="img"):
        label = sample["label"]
        if label in seen:
            continue
        sample["image"].convert("RGB").save(os.path.join(SAVE_DIR, f"{label}_{saved}.JPEG"))
        seen.add(label)
        saved += 1
        if saved >= num_samples:
            break

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ImageNet-1k calibration images.")
    parser.add_argument("--num-samples", type=int, default=100, help="number of images to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = download_imagenet_images(args.num_samples)
    print(f"Done: {count}/{args.num_samples} images in {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
