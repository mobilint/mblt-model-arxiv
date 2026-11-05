#!/usr/bin/env python3
"""Prepare ImageNet-1k validation data for calibration and evaluation.

The official validation image archive has no class directories. This script
uses the official validation bounding-box XML archive to place each image in
its ImageNet synset directory, then selects one image from each class for
calibration. The resulting directories support a full 50,000-image evaluation
and a reproducible 1,000-image calibration run.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree
from pathlib import Path

from download import resolve_source

# The dataset's own directory: default preparation location and home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

DEFAULT_IMAGE_SOURCE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEFAULT_ANNOTATION_SOURCE = (
    "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
)
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def _synsets_path() -> Path:
    return DATASET_HOME / "imagenet_synsets.txt"


def load_synsets() -> tuple[str, ...]:
    """Return the canonical ImageNet-1k synsets in model-output order."""

    synsets = tuple(
        line.strip()
        for line in _synsets_path().read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(synsets) != 1000 or len(set(synsets)) != 1000:
        raise ValueError(
            "configs/imagenet_synsets.txt must contain 1,000 unique synsets."
        )
    return synsets


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a tar archive without permitting path traversal or links."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (root / member.name).resolve()
            if not member_path.is_relative_to(root):
                raise ValueError(
                    f"Archive member escapes extraction directory: {member.name}"
                )
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(
                    f"Archive contains unsupported link or device: {member.name}"
                )
        # Extract members individually for Python 3.10 compatibility. The
        # validation above rejects every member type that could create a link
        # or escape ``root``.
        for member in tar.getmembers():
            tar.extract(member, root)


def _image_directory(source: Path, work_dir: Path) -> Path:
    if source.is_dir():
        return source
    extract_dir = work_dir / "images"
    safe_extract_tar(source, extract_dir)
    return extract_dir


def _annotation_directory(source: Path, work_dir: Path) -> Path:
    if source.is_dir():
        candidates = (source / "val", source)
        return next(
            (candidate for candidate in candidates if candidate.is_dir()), source
        )
    extract_dir = work_dir / "annotations"
    safe_extract_tar(source, extract_dir)
    candidates = sorted(extract_dir.rglob("val"))
    if not candidates:
        raise ValueError(
            "ImageNet annotation archive does not contain a val directory."
        )
    return candidates[0]


def _synset_from_xml(annotation: Path) -> str:
    root = xml.etree.ElementTree.parse(annotation).getroot()
    names = {
        name.text.strip()
        for name in root.findall("object/name")
        if name.text and name.text.strip()
    }
    if len(names) != 1:
        raise ValueError(
            f"Expected exactly one ImageNet synset in {annotation.name}; found {sorted(names)}."
        )
    synset = names.pop()
    if len(synset) != 9 or not synset.startswith("n") or not synset[1:].isdigit():
        raise ValueError(f"Invalid ImageNet synset in {annotation.name}: {synset!r}")
    return synset


def _find_image(image_dir: Path, annotation: Path) -> Path:
    stem = annotation.stem
    matches = [
        image_dir / f"{stem}{suffix}" for suffix in (".JPEG", ".jpeg", ".JPG", ".jpg")
    ]
    image = next((candidate for candidate in matches if candidate.is_file()), None)
    if image is None:
        raise FileNotFoundError(
            f"No validation image matches annotation: {annotation.name}"
        )
    return image


def validate_organized_dataset(
    dataset_dir: Path, require_full_validation: bool = True
) -> None:
    """Validate the organized ImageNet layout and, by default, all 50,000 images."""

    synsets = load_synsets()
    found_dirs = {path.name for path in dataset_dir.iterdir() if path.is_dir()}
    expected_dirs = set(synsets)
    if require_full_validation and found_dirs != expected_dirs:
        missing = sorted(expected_dirs - found_dirs)
        extra = sorted(found_dirs - expected_dirs)
        raise ValueError(
            "Organized ImageNet classes do not match ImageNet-1k "
            f"(missing={missing[:5]}, extra={extra[:5]})."
        )
    for synset in synsets:
        class_dir = dataset_dir / synset
        if not class_dir.is_dir():
            continue
        image_count = sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in class_dir.iterdir()
        )
        if require_full_validation and image_count != 50:
            raise ValueError(f"Expected 50 images in {class_dir}, found {image_count}.")


def organize_imagenet(
    image_source: str,
    annotation_source: str,
    output_dir: Path,
    download_dir: Path,
    require_full_validation: bool = True,
    retries: int = 3,
) -> None:
    """Create an atomic ImageNet validation directory organized by synset."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    download_dir = download_dir.expanduser().resolve()
    image_source_path = resolve_source(image_source, download_dir, retries)
    annotation_source_path = resolve_source(annotation_source, download_dir, retries)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".imagenet-work-") as work_name:
        work_dir = Path(work_name)
        image_dir = _image_directory(image_source_path, work_dir)
        annotation_dir = _annotation_directory(annotation_source_path, work_dir)
        annotations = sorted(annotation_dir.glob("*.xml"))
        if not annotations:
            raise ValueError(f"No XML annotations found in {annotation_dir}")
        canonical_synsets = set(load_synsets())

        staging_dir = Path(
            tempfile.mkdtemp(prefix=".imagenet-staging-", dir=output_dir.parent)
        )
        try:
            for annotation in annotations:
                synset = _synset_from_xml(annotation)
                if require_full_validation and synset not in canonical_synsets:
                    raise ValueError(
                        f"Unknown ImageNet-1k synset in {annotation.name}: {synset}"
                    )
                destination = staging_dir / synset
                destination.mkdir(exist_ok=True)
                shutil.copy2(_find_image(image_dir, annotation), destination)
            validate_organized_dataset(staging_dir, require_full_validation)

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
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    print(f"Organized ImageNet validation set at {output_dir}")


def _images_in_class(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def make_calibration_dataset(
    evaluation_dir: Path, calibration_dir: Path, seed: int = 0
) -> list[dict[str, int | str]]:
    """Create a flat one-image-per-class calibration directory atomically."""

    evaluation_dir = evaluation_dir.expanduser().resolve()
    calibration_dir = calibration_dir.expanduser().resolve()
    if (
        calibration_dir == evaluation_dir
        or calibration_dir.is_relative_to(evaluation_dir)
        or evaluation_dir.is_relative_to(calibration_dir)
    ):
        raise ValueError("calibration_dir must not overlap evaluation_dir.")
    validate_organized_dataset(evaluation_dir)
    calibration_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{calibration_dir.name}-staging-", dir=calibration_dir.parent
        )
    )
    generator = random.Random(seed)
    manifest: list[dict[str, int | str]] = []
    try:
        for index, synset in enumerate(load_synsets()):
            selected = generator.choice(_images_in_class(evaluation_dir / synset))
            destination_name = f"{index:04d}_{synset}_{selected.name}"
            shutil.copy2(selected, staging_dir / destination_name)
            manifest.append(
                {
                    "class_index": index,
                    "synset": synset,
                    "source": str(selected.relative_to(evaluation_dir)),
                    "file": destination_name,
                }
            )
        (staging_dir / "manifest.json").write_text(
            json.dumps(
                {"seed": seed, "images_per_class": 1, "samples": manifest},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        backup_dir: Path | None = None
        if calibration_dir.exists():
            backup_dir = calibration_dir.with_name(f".{calibration_dir.name}.backup")
            if backup_dir.exists():
                raise FileExistsError(
                    f"Refusing to replace existing backup directory: {backup_dir}"
                )
            calibration_dir.replace(backup_dir)
        try:
            staging_dir.replace(calibration_dir)
        except BaseException:
            if backup_dir is not None and not calibration_dir.exists():
                backup_dir.replace(calibration_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return manifest


def prepare_imagenet(
    dataset_dir: Path,
    image_source: str,
    annotation_source: str,
    download_dir: Path | None = None,
    seed: int = 0,
    require_full_validation: bool = True,
    retries: int = 3,
) -> None:
    """Download, organize, and split ImageNet validation data for this repo.

    The evaluation split is written to ``<dataset_dir>/eval`` and the 1,000
    selected calibration images plus their manifest are written to
    ``<dataset_dir>/calib``.
    """

    dataset_dir = dataset_dir.expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    organize_imagenet(
        image_source=image_source,
        annotation_source=annotation_source,
        output_dir=dataset_dir / "eval",
        download_dir=download_dir or dataset_dir / "downloads",
        require_full_validation=require_full_validation,
        retries=retries,
    )
    samples = make_calibration_dataset(
        dataset_dir / "eval", dataset_dir / "calib", seed
    )
    print(
        f"Created {len(samples)} ImageNet calibration images at {dataset_dir / 'calib'}"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    parser.add_argument(
        "--image-source",
        default=DEFAULT_IMAGE_SOURCE,
        help="Image archive, image directory, or authenticated download URL.",
    )
    parser.add_argument(
        "--annotation-source",
        default=DEFAULT_ANNOTATION_SOURCE,
        help="XML archive, XML directory, or authenticated download URL.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_HOME,
        help="ImageNet directory containing eval/, calib/, and downloads/.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory for downloaded archives (default: <dataset-dir>/downloads).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Calibration selection seed."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for interrupted downloads (default: 3).",
    )
    return parser.parse_args(argv)


def run(argv) -> int:
    args = parse_args(argv)
    prepare_imagenet(
        args.dataset_dir,
        args.image_source,
        args.annotation_source,
        args.download_dir,
        args.seed,
        retries=args.retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
