#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Prepare DOTAv1 validation data from the official Google Drive archives."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from paths import IMAGE_SUFFIXES
from workflow import extract_zip, replace

# The dataset's own directory: default preparation location and home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

FOLDER = "https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224"
EXPECTED_VALIDATION_PAIRS = 458
DEFAULT_CALIBRATION_IMAGES = 100
CLASS_TO_INDEX = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
}


def _normalized_labels(source: Path, image: Path) -> str:
    """Convert official DOTA quadrilaterals to normalized class/polygon rows."""
    with Image.open(image) as decoded:
        width, height = decoded.size
    if width <= 0 or height <= 0:
        raise ValueError(f"DOTAv1 image has invalid dimensions: {image}")
    rows: list[str] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.split()
        if not fields or fields[0].lower().startswith(("imagesource", "gsd")):
            continue
        if len(fields) < 10:
            raise ValueError(f"Malformed DOTAv1 annotation at {source}:{line_number}")
        try:
            coordinates = [float(value) for value in fields[:8]]
            class_id = CLASS_TO_INDEX[fields[8]]
            int(fields[9])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid DOTAv1 annotation at {source}:{line_number}"
            ) from exc
        normalized = [
            value / (width if index % 2 == 0 else height)
            for index, value in enumerate(coordinates)
        ]
        rows.append(
            " ".join([str(class_id), *(f"{value:.9g}" for value in normalized)])
        )
    return "\n".join(rows) + ("\n" if rows else "")


def make_calibration_dataset(
    evaluation_dir: Path,
    calibration_dir: Path,
    sample_size: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 0,
) -> list[dict[str, int | str]]:
    """Create a deterministic flat DOTAv1 calibration directory atomically.

    Same contract as the COCO and ImageNet Dataset preparers: the selection is a pure function of
    ``(this file's contents, seed, sample_size)``, the directory is replaced in
    one ``rename`` so a reader never sees a half-written set, and a
    ``manifest.json`` records the seed and every selected sample. Every Model of
    the dataset calibrates on the set one ``(prepare.py, seed)`` pair selects, so
    any non-determinism here silently makes two Models incomparable.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    evaluation_dir = evaluation_dir.expanduser().resolve()
    calibration_dir = calibration_dir.expanduser().resolve()
    if (
        calibration_dir == evaluation_dir
        or calibration_dir.is_relative_to(evaluation_dir)
        or evaluation_dir.is_relative_to(calibration_dir)
    ):
        raise ValueError("calibration_dir must not overlap evaluation_dir.")
    image_dir = evaluation_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"DOTAv1 evaluation images are missing: {image_dir}")
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if sample_size > len(images):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds the {len(images)} available images."
        )

    calibration_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{calibration_dir.name}-staging-", dir=calibration_dir.parent
        )
    )
    selected = random.Random(seed).sample(images, sample_size)
    manifest: list[dict[str, int | str]] = []
    try:
        for index, image in enumerate(selected):
            destination_name = f"{index:04d}_{image.name}"
            shutil.copy2(image, staging_dir / destination_name)
            manifest.append(
                {
                    "sample_index": index,
                    "source": str(image.relative_to(evaluation_dir)),
                    "file": destination_name,
                }
            )
        (staging_dir / "manifest.json").write_text(
            json.dumps(
                {"seed": seed, "sample_size": sample_size, "samples": manifest},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        replace(staging_dir, calibration_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return manifest


def main(a):
    root = a.dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    download = root / "downloads"
    download.mkdir(exist_ok=True)
    if not a.images_archive or not a.labels_archive:
        subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "gdown",
                "gdown",
                "--folder",
                FOLDER,
                "--output",
                str(download),
            ],
            check=True,
        )
        images = next(iter(download.rglob("part1.zip")), None)
        labels = next(iter(download.rglob("labelTxt.zip")), None)
    else:
        images = Path(a.images_archive).expanduser()
        labels = Path(a.labels_archive).expanduser()
    if not images or not labels or not images.is_file() or not labels.is_file():
        raise FileNotFoundError(
            "Provide DOTAv1 part1.zip and labelTxt.zip, or install uv for automatic download."
        )
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        extract_zip(images, work / "images")
        extract_zip(labels, work / "labels")
        image_files = sorted(
            p
            for p in (work / "images").rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )
        images_by_stem = {path.stem: path for path in image_files}
        label_files = {path.stem: path for path in (work / "labels").rglob("*.txt")}
        if len(images_by_stem) != len(image_files) or len(label_files) != len(
            list((work / "labels").rglob("*.txt"))
        ):
            raise ValueError("DOTAv1 archives contain duplicate filename stems")
        if set(images_by_stem) != set(label_files):
            raise ValueError(
                "DOTAv1 validation image and label identifiers do not match"
            )
        if len(images_by_stem) != EXPECTED_VALIDATION_PAIRS:
            raise ValueError(
                f"Expected {EXPECTED_VALIDATION_PAIRS} DOTAv1 validation pairs, "
                f"found {len(images_by_stem)}."
            )
        pairs = [images_by_stem[stem] for stem in sorted(images_by_stem)]
        stage = Path(tempfile.mkdtemp(prefix=".dota-eval-", dir=root))
        (stage / "images").mkdir()
        original_labels = stage / "labels" / "val_original"
        normalized_labels = stage / "labels" / "val"
        original_labels.mkdir(parents=True)
        normalized_labels.mkdir(parents=True)
        for image in pairs:
            shutil.copy2(image, stage / "images" / image.name)
            original = label_files[image.stem]
            shutil.copy2(original, original_labels / f"{image.stem}.txt")
            (normalized_labels / f"{image.stem}.txt").write_text(
                _normalized_labels(original, image), encoding="utf-8"
            )
        replace(stage, root / "eval")
    make_calibration_dataset(root / "eval", root / "calib", a.sample_size, a.seed)


def parse(argv=None):
    p = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    p.add_argument("--images-archive")
    p.add_argument("--labels-archive")
    p.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    p.add_argument("--sample-size", type=int, default=DEFAULT_CALIBRATION_IMAGES)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def run(argv) -> int:
    main(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
