"""WiderFace validation sample loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import image_files


@dataclass(frozen=True)
class WiderFaceSample:
    image: Path
    event: str
    stem: str


EVALUATION_METADATA = (
    "wider_face_val.mat",
    "wider_easy_val.mat",
    "wider_medium_val.mat",
    "wider_hard_val.mat",
)

# A directory built by this dataset's prepare step holds all four
# .mat files flat at its root.
# The upstream WIDER FACE distribution does not: it ships wider_face_val.mat at the root and the
# three difficulty lists inside the eval_tools archive, and an evaluation directory unpacked that
# way is complete, just laid out differently. Look in both places rather than declaring the data missing.
_METADATA_DIRS = (Path("."), Path("eval_tools") / "ground_truth")


def metadata_file(data_path: Path, name: str) -> Path:
    for relative in _METADATA_DIRS:
        candidate = data_path / relative / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"WiderFace evaluation metadata is missing: {name} under {data_path} "
        f"(looked in {', '.join(str(d) for d in _METADATA_DIRS)})"
    )


def validation_samples(data_path: Path) -> tuple[WiderFaceSample, ...]:
    root = data_path / "images"
    samples = tuple(
        WiderFaceSample(path, path.parent.name, path.stem)
        for path in image_files(root, recursive=True)
    )
    for name in EVALUATION_METADATA:
        metadata_file(data_path, name)
    return samples


def samples(task: str, data_path: Path) -> list[dict[str, Any]]:
    """The validation samples in scoring order, as `{"image", "image_id"}` records."""
    if task != "face_detection":
        raise ValueError(f"WIDER FACE does not support task {task!r}")
    return [
        {"image": str(sample.image), "image_id": image_id}
        for image_id, sample in enumerate(validation_samples(data_path))
    ]
