#!/usr/bin/env python3
"""Prepare WiderFace validation data; uses an existing Hugging Face login."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from download import resolve_source
from workflow import extract_zip, replace

# The dataset's own directory: default preparation location and home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

IMAGES = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip"
ANNOTATIONS = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip"
DIFFICULTY_METADATA = {
    "wider_easy_val.mat": (
        "https://huggingface.co/amd/retinaface/resolve/main/"
        "widerface_evaluate/ground_truth/wider_easy_val.mat"
    ),
    "wider_medium_val.mat": (
        "https://huggingface.co/amd/retinaface/resolve/main/"
        "widerface_evaluate/ground_truth/wider_medium_val.mat"
    ),
    "wider_hard_val.mat": (
        "https://huggingface.co/amd/retinaface/resolve/main/"
        "widerface_evaluate/ground_truth/wider_hard_val.mat"
    ),
}
EVALUATION_METADATA = ("wider_face_val.mat", *DIFFICULTY_METADATA)
EXPECTED_CATEGORIES = 61
EXPECTED_VALIDATION_IMAGES = 3_226
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def make_calibration_dataset(
    image_root: Path, output_dir: Path, images_per_category: int, seed: int
) -> None:
    """Select a deterministic number of images from every WiderFace category."""

    if images_per_category <= 0:
        raise ValueError("images_per_category must be positive.")
    category_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if len(category_dirs) != EXPECTED_CATEGORIES:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORIES} WiderFace categories, found {len(category_dirs)}."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    generator = random.Random(seed)
    records: list[dict[str, int | str]] = []
    try:
        for category_index, category_dir in enumerate(category_dirs):
            candidates = sorted(
                path
                for path in category_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if len(candidates) < images_per_category:
                raise ValueError(
                    f"Category {category_dir.name} has only {len(candidates)} images."
                )
            for sample_index, image in enumerate(
                generator.sample(candidates, images_per_category)
            ):
                filename = f"{category_index:02d}_{category_dir.name}_{sample_index}_{image.name}"
                shutil.copy2(image, stage / filename)
                records.append(
                    {
                        "category_index": category_index,
                        "category": category_dir.name,
                        "source": str(image.relative_to(image_root)),
                        "file": filename,
                    }
                )
        (stage / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "images_per_category": images_per_category,
                    "samples": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(a):
    root = a.dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    download = root / "downloads"
    download.mkdir(exist_ok=True)
    if a.images_source == IMAGES and a.annotations_source == ANNOTATIONS:
        for filename in ("data/WIDER_val.zip", "data/wider_face_split.zip"):
            subprocess.run(
                [
                    "hf",
                    "download",
                    "CUHK-CSE/wider_face",
                    "--repo-type",
                    "dataset",
                    "--include",
                    filename,
                    "--local-dir",
                    str(download),
                ],
                check=True,
            )
        image_source = download / "data/WIDER_val.zip"
        annotation_source = download / "data/wider_face_split.zip"
        missing = [
            str(path)
            for path in (image_source, annotation_source)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Hugging Face download completed without required WiderFace archives: "
                + ", ".join(missing)
            )
    else:
        image_source = resolve_source(a.images_source, download, a.retries)
        annotation_source = resolve_source(a.annotations_source, download, a.retries)
    difficulty_metadata = {
        filename: resolve_source(url, download, a.retries)
        for filename, url in DIFFICULTY_METADATA.items()
    }
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        extract_zip(image_source, work / "images")
        extract_zip(annotation_source, work / "annotations")
        images = work / "images" / "WIDER_val" / "images"
        labels = work / "annotations" / "wider_face_split"
        files = sorted(
            p for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        category_dirs = [path for path in images.iterdir() if path.is_dir()]
        required_annotations = (
            labels / EVALUATION_METADATA[0],
            *difficulty_metadata.values(),
        )
        if (
            len(category_dirs) != EXPECTED_CATEGORIES
            or len(files) != EXPECTED_VALIDATION_IMAGES
            or not all(path.is_file() for path in required_annotations)
        ):
            raise ValueError("Invalid WiderFace validation archives")
        stage = Path(tempfile.mkdtemp(prefix=".wider-eval-", dir=root))
        shutil.copytree(images, stage / "images")
        # The dataset-owned evaluator reads these four metadata files from eval/.
        shutil.copy2(labels / "wider_face_val.mat", stage / "wider_face_val.mat")
        for filename, metadata in difficulty_metadata.items():
            shutil.copy2(metadata, stage / filename)
        replace(stage, root / "eval")
    make_calibration_dataset(
        root / "eval" / "images",
        root / "calib",
        a.images_per_category,
        a.seed,
    )


def parse(argv=None):
    p = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    p.add_argument("--images-source", default=IMAGES)
    p.add_argument("--annotations-source", default=ANNOTATIONS)
    p.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    p.add_argument(
        "--images-per-category",
        type=int,
        default=1,
        help="Images randomly selected from each of the 61 categories (default: 1).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args(argv)


def run(argv) -> int:
    main(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
