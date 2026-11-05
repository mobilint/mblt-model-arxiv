"""One L2-normalized appearance vector per input crop."""

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    """Embedding decoding has no runtime options."""


def embedding(
    outputs: list[np.ndarray], customer_contract: dict[str, Any]
) -> np.ndarray:
    if len(outputs) != 1:
        raise PackageError("embedding model must return one output")
    vector = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    expected = customer_contract["outputs"][0]["shape"]
    if vector.size != int(np.prod(expected)) or not np.isfinite(vector).all():
        raise PackageError(f"embedding output does not match finite shape {expected}")
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def summarize(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    output_dir: Path,
    source: Path,
) -> dict[str, Any]:
    vector = embedding(outputs, customer_contract)
    return {
        "task": customer_contract["task"],
        "embedding": vector.tolist(),
        "dimensions": int(vector.size),
    }
