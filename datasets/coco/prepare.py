#!/usr/bin/env python3
"""Prepare COCO 2017 validation data for calibration and evaluation.

The script downloads (or accepts local copies of) the COCO validation images
and annotations, writes the full validation set to ``eval/``, and creates a
deterministic flat calibration set in ``calib/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from download import resolve_source

# The dataset's own directory: default preparation location and home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

# The official COCO host serves plain HTTP only; each archive is verified against its
# SHA-256 below before extraction, so a changed or tampered download is refused.
DEFAULT_IMAGE_SOURCE = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_ANNOTATION_SOURCE = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGE_SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
ANNOTATION_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_VALIDATION_IMAGES = 5_000
DEFAULT_CALIBRATION_IMAGES = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_digest: str) -> None:
    """Verify an official COCO archive against its pinned SHA-256 digest."""

    actual_digest = _sha256(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_digest}, "
            f"got {actual_digest}."
        )


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive without allowing traversal paths or symlinks."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (root / member.filename).resolve()
            mode = member.external_attr >> 16
            if not member_path.is_relative_to(root):
                raise ValueError(
                    f"Archive member escapes extraction directory: {member.filename}"
                )
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"Archive contains unsupported symlink: {member.filename}"
                )
        zip_file.extractall(root)


def _find_image_directory(source: Path, workspace: Path) -> Path:
    if source.is_dir():
        candidates = (source / "val2017", source)
        image_dir = next((path for path in candidates if path.is_dir()), None)
        if image_dir is None:
            raise ValueError(f"No val2017 directory found in {source}")
        return image_dir
    extract_dir = workspace / "images"
    safe_extract_zip(source, extract_dir)
    image_dir = extract_dir / "val2017"
    if not image_dir.is_dir():
        raise ValueError(f"COCO image archive does not contain val2017/: {source}")
    return image_dir


def _find_annotation_directory(source: Path, workspace: Path) -> Path:
    if source.is_dir():
        candidates = (source / "annotations", source)
        annotation_dir = next((path for path in candidates if path.is_dir()), None)
        if annotation_dir is None:
            raise ValueError(f"No annotations directory found in {source}")
        return annotation_dir
    extract_dir = workspace / "annotations"
    safe_extract_zip(source, extract_dir)
    annotation_dir = extract_dir / "annotations"
    if not annotation_dir.is_dir():
        raise ValueError(
            f"COCO annotation archive does not contain annotations/: {source}"
        )
    return annotation_dir


def _replace_directory(staging_dir: Path, output_dir: Path) -> None:
    """Atomically replace output_dir with staging_dir, restoring on failure."""

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
        if backup_dir.exists():
            raise FileExistsError(
                f"Refusing to replace existing backup directory: {backup_dir}"
            )
        output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except BaseException:
        if backup_dir is not None and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def validate_evaluation_dataset(evaluation_dir: Path) -> None:
    """Validate the expected COCO 2017 validation files."""

    image_dir = evaluation_dir / "val2017"
    image_count = (
        sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in image_dir.iterdir()
        )
        if image_dir.is_dir()
        else 0
    )
    if image_count != EXPECTED_VALIDATION_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_IMAGES} COCO validation images in "
            f"{image_dir}, found {image_count}."
        )
    for filename in ("instances_val2017.json", "person_keypoints_val2017.json"):
        if not (evaluation_dir / filename).is_file():
            raise FileNotFoundError(f"Missing COCO evaluation annotation: {filename}")


def organize_coco(
    image_source: str,
    annotation_source: str,
    evaluation_dir: Path,
    download_dir: Path,
    verify_checksums: bool = True,
    retries: int = 3,
) -> None:
    """Organize COCO validation images and annotations into evaluation_dir."""

    evaluation_dir = evaluation_dir.expanduser().resolve()
    if evaluation_dir.exists() and not evaluation_dir.is_dir():
        raise ValueError(f"Evaluation path is not a directory: {evaluation_dir}")
    image_archive = resolve_source(image_source, download_dir, retries)
    annotation_archive = resolve_source(annotation_source, download_dir, retries)
    if verify_checksums:
        if image_archive.is_file():
            verify_archive(image_archive, IMAGE_SHA256)
        if annotation_archive.is_file():
            verify_archive(annotation_archive, ANNOTATION_SHA256)
    evaluation_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".coco-work-") as workspace_name:
        workspace = Path(workspace_name)
        image_dir = _find_image_directory(image_archive, workspace)
        annotation_dir = _find_annotation_directory(annotation_archive, workspace)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=".coco-eval-staging-", dir=evaluation_dir.parent)
        )
        try:
            shutil.copytree(image_dir, staging_dir / "val2017")
            for annotation in sorted(annotation_dir.glob("*_val2017.json")):
                shutil.copy2(annotation, staging_dir / annotation.name)
            validate_evaluation_dataset(staging_dir)
            _replace_directory(staging_dir, evaluation_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
    print(f"Organized COCO validation set at {evaluation_dir}")


def make_calibration_dataset(
    evaluation_dir: Path,
    calibration_dir: Path,
    sample_size: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 0,
) -> list[dict[str, int | str]]:
    """Create a deterministic flat COCO calibration directory atomically."""

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
    validate_evaluation_dataset(evaluation_dir)
    images = sorted(
        path
        for path in (evaluation_dir / "val2017").iterdir()
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
        _replace_directory(staging_dir, calibration_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return manifest


def prepare_coco(
    dataset_dir: Path,
    image_source: str,
    annotation_source: str,
    download_dir: Path | None = None,
    sample_size: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 0,
    verify_checksums: bool = True,
    retries: int = 3,
) -> None:
    """Download, organize, and prepare COCO validation data in dataset_dir."""

    dataset_dir = dataset_dir.expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = dataset_dir / "eval"
    calibration_dir = dataset_dir / "calib"
    organize_coco(
        image_source,
        annotation_source,
        evaluation_dir,
        download_dir or dataset_dir / "downloads",
        verify_checksums,
        retries,
    )
    samples = make_calibration_dataset(
        evaluation_dir, calibration_dir, sample_size, seed
    )
    print(f"Created {len(samples)} COCO calibration images at {calibration_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    parser.add_argument("--image-source", default=DEFAULT_IMAGE_SOURCE)
    parser.add_argument("--annotation-source", default=DEFAULT_ANNOTATION_SOURCE)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_CALIBRATION_IMAGES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="Allow non-official local archives without SHA-256 verification.",
    )
    return parser.parse_args(argv)


def run(argv) -> int:
    args = parse_args(argv)
    prepare_coco(
        args.dataset_dir,
        args.image_source,
        args.annotation_source,
        args.download_dir,
        args.sample_size,
        args.seed,
        verify_checksums=not args.no_verify_checksums,
        retries=args.retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
