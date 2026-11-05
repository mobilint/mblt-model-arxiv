"""Reader for the dense embedding matrix an embedding-task prediction artifact references.

Retrieval and verification benchmarks (person re-identification, face verification) receive
one vector per image. The artifact keeps ``records`` of ``{"image_id", "row"}`` and one
``embeddings`` block naming a float32 ``.npy`` matrix with its sha256 and shape; the matrix
is verified before any row is trusted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embeddings_by_image_id(artifact: dict[str, Any]) -> dict[int, np.ndarray]:
    """Return ``image_id -> embedding row`` after verifying the referenced matrix."""
    block = artifact.get("embeddings")
    if not isinstance(block, dict):
        raise TypeError("embedding artifact must reference an embeddings matrix")
    path = Path(str(block.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Embedding matrix is missing: {path}")
    actual = _sha256(path)
    if actual != block.get("sha256"):
        raise ValueError(f"Embedding matrix changed: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2 or list(matrix.shape) != list(block.get("shape", [])):
        raise ValueError(
            f"Embedding matrix shape {matrix.shape} differs from the recorded {block.get('shape')}"
        )
    records = artifact.get("records")
    if not isinstance(records, list):
        raise TypeError("Prediction artifact records must be an array")
    if len(records) != len(matrix):
        raise ValueError(
            f"Prediction artifact has {len(records)} records for {len(matrix)} embedding rows"
        )
    result: dict[int, np.ndarray] = {}
    rows_seen: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction artifact records must be objects")
        image_id, row = record.get("image_id"), record.get("row")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise TypeError("embedding record image_id must be an integer")
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or not 0 <= row < len(matrix)
        ):
            raise TypeError(f"embedding record row is not a matrix row: {row!r}")
        if image_id in result:
            raise ValueError(f"Prediction artifact has duplicate image_id: {image_id}")
        if row in rows_seen:
            raise ValueError(
                f"Prediction artifact maps two images to embedding row {row}"
            )
        rows_seen.add(row)
        result[image_id] = matrix[row]
    return result
