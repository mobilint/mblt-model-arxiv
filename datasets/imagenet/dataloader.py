"""ImageNet-1k validation dataloader owned by the ImageNet dataset workflow."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The dataset's own directory: home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class ImageNetSample:
    """One organized ImageNet validation image and its canonical class ID."""

    path: Path
    label: int


def canonical_synsets(dataset_dir: Path) -> tuple[str, ...]:
    """Read the canonical ImageNet-1k class order bundled with this dataset."""
    synsets = tuple(
        line.strip()
        for line in (dataset_dir / "imagenet_synsets.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if len(synsets) != 1000 or len(set(synsets)) != 1000:
        raise ValueError("imagenet_synsets.txt must contain 1,000 unique synsets.")
    return synsets


def iter_validation_samples(data_path: Path) -> Iterator[ImageNetSample]:
    """Yield organized validation images in deterministic class/path order."""
    data_path = data_path.resolve()
    dataset_dir = DATASET_HOME
    synsets = canonical_synsets(dataset_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"ImageNet evaluation directory is missing: {data_path}"
        )
    for label, synset in enumerate(synsets):
        class_dir = data_path / synset
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet validation class directory is missing: {class_dir}"
            )
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"ImageNet validation class has no images: {class_dir}")
        yield from (ImageNetSample(path, label) for path in paths)


def iter_validation_batches(
    data_path: Path, batch_size: int
) -> Iterator[tuple[ImageNetSample, ...]]:
    """Yield deterministic ImageNet validation batches without a torch dependency."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[ImageNetSample] = []
    for sample in iter_validation_samples(data_path):
        batch.append(sample)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def samples(task: str, data_path: Path) -> list[dict[str, Any]]:
    """The validation samples in scoring order, as `{"image", "image_id"}` records."""
    if task != "image_classification":
        raise ValueError(f"ImageNet does not support task {task!r}")
    return [
        {"image": str(sample.path), "image_id": image_id}
        for image_id, sample in enumerate(iter_validation_samples(data_path))
    ]
