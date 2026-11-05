#!/usr/bin/env python3
"""Prepare Cityscapes validation data from manually approved local archives."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from workflow import IMAGE_SUFFIXES, calibration, extract_zip, replace

# The dataset's own directory: default preparation location and home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

IMAGE_ARCHIVE = "leftImg8bit_trainvaltest.zip"
ANNOTATION_ARCHIVE = "gtFine_trainvaltest.zip"
IMAGE_SUFFIX = "_leftImg8bit.png"
ANNOTATION_SUFFIX = "_gtFine_labelIds.png"
EXPECTED_VALIDATION_IMAGES = 500
DEFAULT_CALIBRATION_IMAGES = 100


def _calibration_sources(evaluation_dir: Path, calibration_dir: Path) -> list[Path]:
    """The validation images a calibration set may be drawn from, in a stable order."""
    if (
        calibration_dir == evaluation_dir
        or calibration_dir.is_relative_to(evaluation_dir)
        or evaluation_dir.is_relative_to(calibration_dir)
    ):
        raise ValueError("calibration_dir must not overlap evaluation_dir.")
    images = sorted(
        path
        for path in (evaluation_dir / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(
            f"No Cityscapes validation images under {evaluation_dir / 'images'}"
        )
    return images


def make_calibration_dataset(
    evaluation_dir: Path,
    calibration_dir: Path,
    sample_size: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 0,
) -> list[dict[str, int | str]]:
    """Create a deterministic flat Cityscapes calibration directory atomically.

    Same contract as the COCO and ImageNet Dataset preparers:
    the selection is a seeded sample of the sorted validation images, so the same
    (script, seed, sample_size) always picks the same files; ``manifest.json`` records
    the seed and every selected sample; and a directory whose manifest already matches
    is returned untouched. Every Model of the dataset calibrates on the set one
    (prepare.py, seed) pair selects, so re-sampling here would silently make two Models
    incomparable.
    """
    evaluation_dir = evaluation_dir.expanduser().resolve()
    calibration_dir = calibration_dir.expanduser().resolve()
    manifest_path = calibration_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") == seed and manifest.get("sample_size") == sample_size:
            return manifest["samples"]
        raise FileExistsError(
            f"{calibration_dir} already holds a calibration set made with "
            f"seed {manifest.get('seed')!r} and sample_size {manifest.get('sample_size')!r}; "
            f"refusing to replace it with seed {seed!r} and sample_size {sample_size!r}."
        )
    calibration(
        _calibration_sources(evaluation_dir, calibration_dir),
        calibration_dir,
        sample_size,
        seed,
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]


def _validation_files(root: Path, suffix: str) -> dict[str, Path]:
    files = {
        path.name.removesuffix(suffix): path
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
    }
    if len(files) != sum(1 for path in root.rglob(f"*{suffix}") if path.is_file()):
        raise ValueError(f"Duplicate Cityscapes sample identifiers in {root}")
    return files


def prepare_cityscapes(
    dataset_dir: Path, sample_size: int = DEFAULT_CALIBRATION_IMAGES, seed: int = 0
) -> None:
    """Organize local approved archives into eval/ and calib/."""

    dataset_dir = dataset_dir.expanduser().resolve()
    image_archive = dataset_dir / "downloads" / IMAGE_ARCHIVE
    annotation_archive = dataset_dir / "downloads" / ANNOTATION_ARCHIVE
    missing = [
        str(path) for path in (image_archive, annotation_archive) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cityscapes requires manually approved archives in downloads/: "
            + ", ".join(missing)
        )

    with tempfile.TemporaryDirectory(prefix=".cityscapes-work-") as workspace_name:
        workspace = Path(workspace_name)
        extract_zip(image_archive, workspace / "images")
        extract_zip(annotation_archive, workspace / "annotations")
        images = _validation_files(
            workspace / "images" / "leftImg8bit" / "val", IMAGE_SUFFIX
        )
        annotations = _validation_files(
            workspace / "annotations" / "gtFine" / "val", ANNOTATION_SUFFIX
        )
        if set(images) != set(annotations):
            raise ValueError(
                "Cityscapes validation image and annotation identifiers do not match."
            )
        if len(images) != EXPECTED_VALIDATION_IMAGES:
            raise ValueError(
                f"Expected {EXPECTED_VALIDATION_IMAGES} Cityscapes validation pairs, found {len(images)}."
            )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=".cityscapes-eval-", dir=dataset_dir)
        )
        try:
            (staging_dir / "images").mkdir()
            (staging_dir / "annotations").mkdir()
            for sample_id in sorted(images):
                shutil.copy2(
                    images[sample_id], staging_dir / "images" / f"{sample_id}.png"
                )
                shutil.copy2(
                    annotations[sample_id],
                    staging_dir / "annotations" / f"{sample_id}.png",
                )
            replace(staging_dir, dataset_dir / "eval")
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    make_calibration_dataset(
        dataset_dir / "eval", dataset_dir / "calib", sample_size, seed
    )
    print(f"Prepared Cityscapes evaluation data at {dataset_dir / 'eval'}")
    print(
        f"Created {sample_size} Cityscapes calibration images at {dataset_dir / 'calib'}"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_CALIBRATION_IMAGES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run(argv) -> int:
    args = parse_args(argv)
    prepare_cityscapes(args.dataset_dir, args.sample_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
