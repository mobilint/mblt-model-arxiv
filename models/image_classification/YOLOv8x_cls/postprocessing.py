"""Classification outputs: softmax over the class logits and the top-k classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    argument_parser.add_argument(
        "--topk",
        type=int,
        default=int(customer_contract["inference"]["params"].get("topk", 5)),
        help="number of classes to report (default: %(default)s)",
    )


def softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32).reshape(-1)
    shifted = logits - logits.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def declared_logits(
    outputs: list[np.ndarray], customer_contract: dict[str, Any]
) -> np.ndarray:
    """The class logits, refused unless they match the contract's declared shape."""
    declared = customer_contract["outputs"][0].get("shape")
    logits = np.asarray(outputs[0], dtype=np.float32)
    if declared and logits.size != int(np.prod(declared)):
        raise PackageError(
            f"model output has {logits.size} values, the contract declares shape {declared}"
        )
    return logits


def summarize(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    output_dir: Path,
    source: Path,
) -> dict[str, Any]:
    if args.topk <= 0:
        raise PackageError("--topk must be positive")
    probabilities = softmax(declared_logits(outputs, customer_contract))
    topk = min(args.topk, probabilities.size)
    indices = np.argsort(-probabilities, kind="stable")[:topk]
    return {
        "task": customer_contract["task"],
        "classes": int(probabilities.size),
        "topk": [
            {"class_index": int(index), "probability": float(probabilities[index])}
            for index in indices
        ],
    }


def prediction_records(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    image_id: int,
) -> list[dict[str, Any]]:
    """One benchmark record: the five highest-scoring class indexes, best first."""
    logits = declared_logits(outputs, customer_contract).reshape(-1)
    top = np.argsort(-logits, kind="stable")[:5]
    return [{"image_id": image_id, "topk": [int(index) for index in top]}]
