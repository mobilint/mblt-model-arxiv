"""DukeMTMC-reID query/gallery sample loader."""

from __future__ import annotations

from pathlib import Path

from reid import ReidSample, split_samples

CAMERAS = 8
EXPECTED_QUERY = 2228
EXPECTED_GALLERY = 17661


def validation_samples(data_path: Path) -> tuple[ReidSample, ...]:
    """Query crops first, then gallery crops, each in sorted file-name order."""
    data_path = Path(data_path)
    query = split_samples(
        data_path / "query", "query", cameras=CAMERAS, skip_junk=False
    )
    gallery = split_samples(
        data_path / "bounding_box_test",
        "gallery",
        cameras=CAMERAS,
        skip_junk=False,
    )
    if len(query) != EXPECTED_QUERY or len(gallery) != EXPECTED_GALLERY:
        raise ValueError(
            f"DukeMTMC-reID evaluation split holds {len(query)} query and {len(gallery)} "
            f"gallery crops; expected {EXPECTED_QUERY} and {EXPECTED_GALLERY}"
        )
    return query + gallery


def samples(task: str, data_path: Path) -> list[dict[str, str | int]]:
    """Expose the benchmark crop order to an independent Model evaluator."""
    if task != "person_reid":
        raise ValueError(f"This Dataset supports person_reid, not {task!r}")
    return [
        {"image": str(sample.image), "image_id": image_id}
        for image_id, sample in enumerate(validation_samples(data_path))
    ]
