"""DOTAv1 validation image and label loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import image_files


@dataclass(frozen=True)
class DOTASample:
    image: Path
    original_label: Path


def validation_samples(data_path: Path) -> tuple[DOTASample, ...]:
    original_root = data_path / "labels" / "val_original"
    samples: list[DOTASample] = []
    for image in image_files(data_path / "images"):
        original = original_root / f"{image.stem}.txt"
        if not original.is_file():
            raise FileNotFoundError(
                f"DOTAv1 official label is missing for {image.name}: {original}"
            )
        samples.append(DOTASample(image, original))
    return tuple(samples)


def samples(task: str, data_path: Path) -> list[dict[str, Any]]:
    """The validation samples in scoring order, as `{"image", "image_id"}` records."""
    if task != "obb":
        raise ValueError(f"DOTAv1 does not support task {task!r}")
    return [
        {"image": str(sample.image), "image_id": image_id}
        for image_id, sample in enumerate(validation_samples(data_path))
    ]
