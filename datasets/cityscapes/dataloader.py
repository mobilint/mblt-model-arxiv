"""Deterministic Cityscapes validation pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import image_files


@dataclass(frozen=True)
class CityscapesSample:
    image: Path
    target: Path
    key: str


# The store holds the annotations under ``masks/`` and names each one after the
# Cityscapes archive it came from, so a validation pair is
# ``images/<id>_leftImg8bit.png`` / ``masks/<id>_gtFine_labelIds.png`` rather than the
# same relative path on both sides. ``prepare.py`` writes the flattened
# ``annotations/<id>.png`` layout instead. Resolve both, so a checkout works against
# either -- this is the same split the WiderFace loader resolves for its metadata.
_TARGET_DIRS = ("annotations", "masks")
_IMAGE_SUFFIX = "_leftImg8bit"
_TARGET_SUFFIX = "_gtFine_labelIds"


def _target_for(data_path: Path, relative: Path) -> Path:
    """The annotation that pairs with ``images/<relative>``, in whichever layout is present."""
    stem = relative.stem
    candidates = [relative]
    if stem.endswith(_IMAGE_SUFFIX):
        base = stem[: -len(_IMAGE_SUFFIX)]
        candidates.append(
            relative.with_name(f"{base}{_TARGET_SUFFIX}{relative.suffix}")
        )
        candidates.append(relative.with_name(f"{base}{relative.suffix}"))
    for directory in _TARGET_DIRS:
        for candidate in candidates:
            target = data_path / directory / candidate
            if target.is_file():
                return target
    raise FileNotFoundError(
        f"Cityscapes target is missing for images/{relative}: looked for "
        + ", ".join(f"{d}/{c}" for d in _TARGET_DIRS for c in candidates)
    )


def validation_samples(data_path: Path) -> tuple[CityscapesSample, ...]:
    image_root = data_path / "images"
    samples: list[CityscapesSample] = []
    for image in image_files(image_root, recursive=True):
        relative = image.relative_to(image_root)
        target = _target_for(data_path, relative)
        key = relative.stem
        key = key.removesuffix(_IMAGE_SUFFIX)
        samples.append(
            CityscapesSample(
                image, target, str(relative.with_name(key).with_suffix(""))
            )
        )
    return tuple(samples)


def samples(task: str, data_path: Path) -> list[dict[str, Any]]:
    """The validation samples in scoring order, as `{"image", "image_id"}` records."""
    if task != "semantic_segmentation":
        raise ValueError(f"Cityscapes does not support task {task!r}")
    return [
        {"image": str(sample.image), "image_id": image_id}
        for image_id, sample in enumerate(validation_samples(data_path))
    ]
