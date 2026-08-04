import argparse
import os
import random
import shutil
import zipfile
from urllib.request import urlretrieve

SAVE_DIR: str = "./depth_files"
ARCHIVE_PATH: str = "./depth_files.zip"
DATASET_URL: str = "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip"
SEED: int = 42


def download_archive() -> str:
    """Download the NYU Depth V2 archive unless it is already present. Returns its path."""
    if os.path.isfile(ARCHIVE_PATH):
        print(f"Using existing archive: {ARCHIVE_PATH}")
        return ARCHIVE_PATH
    print(f"Downloading {DATASET_URL}...")
    urlretrieve(DATASET_URL, ARCHIVE_PATH)
    return ARCHIVE_PATH


def download_depth_images(num_samples: int) -> int:
    """Extract num_samples random NYU Depth V2 RGB images into SAVE_DIR. Returns count saved."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    with zipfile.ZipFile(download_archive()) as archive:
        # The archive holds RGB under nyu-depth/images/ and depth maps as .npy under
        # nyu-depth/depth/, so the .jpg extension alone selects the RGB inputs.
        members = [name for name in archive.namelist() if name.lower().endswith(".jpg")]
        if not members:
            raise RuntimeError(f"No .jpg images found in {ARCHIVE_PATH}")
        random.seed(SEED)
        selected = random.sample(members, min(num_samples, len(members)))
        for member in selected:
            target = os.path.join(SAVE_DIR, os.path.basename(member))
            with archive.open(member) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)

    return len(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYU Depth V2 calibration images.")
    parser.add_argument("--num-samples", type=int, default=100, help="number of images to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = download_depth_images(args.num_samples)
    print(f"Done: {count}/{args.num_samples} images in {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
