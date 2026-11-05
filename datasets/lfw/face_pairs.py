"""Pair-list face verification benchmarks: pair file, image list, and scoring.

LFW, CFP-FP and AgeDB-30 share one layout once prepared: an image directory and a
``pairs.csv`` whose rows name two image files, a same/different label and a fold. The
manifest lists every image a pair references once, in sorted file-name order; the score
looks each pair's two embeddings up by file name and applies the verification protocol
of :mod:`datasets._shared.face_verification`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import embeddings_by_image_id
from face_verification import (
    Pair,
    cosine_scores,
    verification_result,
)

PAIR_COLUMNS = ("pair_id", "fold_id", "is_same", "image_a", "image_b")
TASK = "face_recognition"


@dataclass(frozen=True)
class PairImage:
    image: Path  # aligned 112x112 crop
    name: str  # file name, the pair key


def read_pair_file(path: Path, expected: int) -> tuple[Pair, ...]:
    """Every pair of ``path`` in file order; the count must equal ``expected``."""
    if not path.is_file():
        raise FileNotFoundError(f"Pair file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(PAIR_COLUMNS) <= set(reader.fieldnames):
            raise ValueError(f"{path} lacks the pair columns {list(PAIR_COLUMNS)}")
        pairs = []
        ids: set[int] = set()
        for row in reader:
            for key in ("image_a", "image_b"):
                name = row[key]
                if not name or name in {".", ".."} or any(c in name for c in "/\\:"):
                    raise ValueError(f"{path} has an invalid {key} basename: {name!r}")
            if row["is_same"] not in {"0", "1"}:
                raise ValueError(f"{path} has an invalid is_same: {row['is_same']!r}")
            try:
                fold = int(row["fold_id"])
                pair_id = int(row["pair_id"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path} has a noninteger fold_id or pair_id") from exc
            if not 1 <= fold <= 10:
                raise ValueError(f"{path} fold_id must be in 1..10: {fold}")
            if pair_id in ids:
                raise ValueError(f"{path} has duplicate pair_id: {pair_id}")
            ids.add(pair_id)
            pairs.append(
                Pair(row["image_a"], row["image_b"], row["is_same"] == "1", fold)
            )
    if len(pairs) != expected:
        raise ValueError(f"{path} holds {len(pairs)} pairs, expected {expected}")
    return tuple(pairs)


def pair_images(image_dir: Path, pairs: tuple[Pair, ...]) -> tuple[PairImage, ...]:
    """Every image a pair names, once, in sorted file-name order."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory is missing: {image_dir}")
    names = sorted({p.image_a for p in pairs} | {p.image_b for p in pairs})
    images = []
    for name in names:
        path = image_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Pair image is missing: {path}")
        images.append(PairImage(path, name))
    return tuple(images)


def score_pairs(
    images: tuple[PairImage, ...],
    pairs: tuple[Pair, ...],
    artifact: dict[str, Any],
    *,
    benchmark: str,
    metric: str,
) -> dict[str, Any]:
    """Score a face_recognition embedding artifact made from the ``images`` manifest."""
    if artifact.get("task") != TASK:
        raise ValueError(f"{benchmark} accepts {TASK} prediction artifacts only")
    if artifact.get("image_ids") != list(range(len(images))):
        raise ValueError(
            f"Prediction artifact image_ids do not match the {benchmark} manifest order"
        )
    embeddings = embeddings_by_image_id(artifact)
    missing = [i for i in range(len(images)) if i not in embeddings]
    if missing:
        raise ValueError(f"Prediction artifact is missing image_ids {missing[:5]}")
    by_name = {image.name: embeddings[i] for i, image in enumerate(images)}
    a = np.stack([by_name[p.image_a] for p in pairs])
    b = np.stack([by_name[p.image_b] for p in pairs])
    result = verification_result(
        cosine_scores(a, b),
        np.array([p.is_same for p in pairs]),
        np.array([p.fold for p in pairs]),
    )
    return {
        "primary": {metric: round(result["accuracy"], 5)},
        "secondary": {f"{metric}_std": round(result["accuracy_std"], 5)},
        "sample_count": len(images),
    }
