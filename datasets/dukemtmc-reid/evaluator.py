"""DukeMTMC-reID scoring for normalized person_reid embedding artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reid import score

from dataloader import validation_samples

TASK = "person_reid"


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """mAP (primary) and CMC rank-1/5/10 (secondary) under the Market-1501 protocol."""
    return score(TASK, validation_samples(Path(data_path)), artifact)
