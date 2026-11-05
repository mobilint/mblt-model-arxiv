"""LFW pair protocol loader: the aligned images the 6,000 official pairs reference."""

from __future__ import annotations

from pathlib import Path

from face_pairs import PairImage, pair_images, read_pair_file
from face_verification import Pair

EXPECTED_PAIRS = 6_000


def read_pairs(data_path: Path) -> tuple[Pair, ...]:
    return read_pair_file(Path(data_path) / "pairs.csv", EXPECTED_PAIRS)


def validation_samples(data_path: Path) -> tuple[PairImage, ...]:
    """Every image named by a pair, once, in sorted file-name order."""
    return pair_images(Path(data_path) / "aligned", read_pairs(data_path))


def samples(task: str, data_path: Path) -> list[dict[str, str | int]]:
    """Expose each aligned pair image once in the scorer's canonical order."""
    if task != "face_recognition":
        raise ValueError(f"LFW supports face_recognition, not {task!r}")
    return [
        {"image": str(sample.image), "image_id": image_id}
        for image_id, sample in enumerate(validation_samples(data_path))
    ]
