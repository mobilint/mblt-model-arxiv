"""Deterministic NYU Depth V2 validation pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import IMAGE_SUFFIXES, files_by_stem


@dataclass(frozen=True)
class NYUDepthSample:
    image: Path
    target: Path
    stem: str


def validation_samples(data_path: Path) -> tuple[NYUDepthSample, ...]:
    images = files_by_stem(data_path / "images", IMAGE_SUFFIXES)
    targets = files_by_stem(data_path / "depth", frozenset({".npy"}))
    if images.keys() != targets.keys():
        raise ValueError(
            "NYU Depth validation images and depth maps must have matching stems."
        )
    return tuple(
        NYUDepthSample(images[stem], targets[stem], stem) for stem in sorted(images)
    )


def samples(task: str, data_path: Path) -> list[dict[str, Any]]:
    """The validation samples in scoring order, as `{"image", "image_id"}` records."""
    if task != "depth_estimation":
        raise ValueError(f"NYU Depth does not support task {task!r}")
    return [
        {"image": str(sample.image), "image_id": image_id}
        for image_id, sample in enumerate(validation_samples(data_path))
    ]
