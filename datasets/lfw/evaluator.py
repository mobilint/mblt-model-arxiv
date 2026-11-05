"""LFW scoring: 10-fold verification accuracy over face_recognition embedding artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_pairs import score_pairs

from dataloader import read_pairs, validation_samples


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(data_path)
    return score_pairs(
        validation_samples(data_path),
        read_pairs(data_path),
        artifact,
        benchmark="LFW",
        metric="lfw_accuracy",
    )
