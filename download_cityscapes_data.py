import argparse
import json
import os
import random
from urllib.parse import urlencode
from urllib.request import urlopen

from datasets import load_dataset
from tqdm import tqdm

SAVE_DIR: str = "./cityscapes_files"
DATASET: str = "Chris1/cityscapes_segmentation"
DATASET_CONFIG: str = "default"
DATASET_SPLIT: str = "validation"
PARQUET_API_URL: str = "https://datasets-server.huggingface.co/parquet"
SEED: int = 42


def get_validation_parquet_urls() -> list[str]:
    """Return the validation Parquet URLs the Dataset Viewer reports for DATASET."""
    request_url = f"{PARQUET_API_URL}?{urlencode({'dataset': DATASET})}"
    with urlopen(request_url, timeout=30) as response:  # noqa: S310
        payload = json.load(response)

    parquet_urls = [
        parquet_file["url"]
        for parquet_file in payload.get("parquet_files", [])
        if parquet_file.get("config") == DATASET_CONFIG and parquet_file.get("split") == DATASET_SPLIT
    ]
    if not parquet_urls:
        raise RuntimeError(f"No {DATASET_SPLIT} Parquet files found for {DATASET}/{DATASET_CONFIG}")
    return parquet_urls


def download_cityscapes_images(num_samples: int) -> int:
    """Save num_samples Cityscapes validation RGB images into SAVE_DIR. Returns count saved."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = load_dataset(
        "parquet",
        data_files={DATASET_SPLIT: get_validation_parquet_urls()},
        split=DATASET_SPLIT,
        columns=["image"],
    )
    indices = sorted(random.Random(SEED).sample(range(len(dataset)), min(num_samples, len(dataset))))

    for index in tqdm(indices, desc=os.path.basename(SAVE_DIR), unit="img"):
        image = dataset[index]["image"].convert("RGB")
        image.save(os.path.join(SAVE_DIR, f"cityscapes_{index:04d}.png"))

    return len(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Cityscapes calibration images.")
    parser.add_argument("--num-samples", type=int, default=100, help="number of images to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = download_cityscapes_images(args.num_samples)
    print(f"Done: {count}/{args.num_samples} images in {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
