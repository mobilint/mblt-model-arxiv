"""COCO 2017 validation metadata loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class COCOSample:
    image: Path
    image_id: int
    height: int
    width: int


def _annotation_name(task: str) -> str:
    return (
        "person_keypoints_val2017.json"
        if task == "pose_estimation"
        else "instances_val2017.json"
    )


def validation_samples(data_path: Path, task: str) -> tuple[COCOSample, ...]:
    annotation_path = data_path / _annotation_name(task)
    try:
        raw: Any = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read COCO annotations {annotation_path}: {exc}"
        ) from exc
    images = raw.get("images") if isinstance(raw, dict) else None
    if not isinstance(images, list):
        raise TypeError(f"COCO annotations have no image table: {annotation_path}")
    samples: list[COCOSample] = []
    image_root = (data_path / "val2017").resolve()
    for record in sorted(images, key=lambda value: int(value["id"])):
        relative = Path(str(record["file_name"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe COCO image path: {relative}")
        image = (image_root / relative).resolve()
        image.relative_to(image_root)
        if not image.is_file():
            raise FileNotFoundError(f"COCO validation image is missing: {image}")
        samples.append(
            COCOSample(
                image, int(record["id"]), int(record["height"]), int(record["width"])
            )
        )
    if not samples:
        raise ValueError(f"COCO validation set is empty: {annotation_path}")
    return tuple(samples)


def samples(task: str, data_path: Path) -> list[dict[str, Any]]:
    """The validation samples in scoring order, as `{"image", "image_id"}` records."""
    return [
        {"image": str(sample.image), "image_id": sample.image_id}
        for sample in validation_samples(data_path, task)
    ]
