"""Decode semantic outputs and write benchmark-neutral row-major label RLE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from package_common import PackageError

OutputLayout = Literal["channels_first", "channels_last"]


def class_map(output: Any, layout: OutputLayout) -> np.ndarray:
    """Reduce one backend-native semantic output to a two-dimensional label map.

    ONNX exports deliver ``NCHW`` logits while the MXQ runtime exposes the same
    dense head as ``HWC``.  That distinction belongs to the Model adapter;
    neither branch needs a benchmark class count to decide the reduction axis.
    """

    value = np.asarray(output)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise PackageError(
                f"Semantic output requires a singleton batch axis: {value.shape}"
            )
        value = value[0]
    if value.ndim == 2:
        labels = value
    elif value.ndim == 3:
        axis = 0 if layout == "channels_first" else -1
        labels = (
            np.squeeze(value, axis=axis)
            if value.shape[axis] == 1
            else value.argmax(axis=axis)
        )
    else:
        raise PackageError(
            f"Semantic model output has unsupported shape: {value.shape}"
        )
    if labels.ndim != 2:
        raise PackageError(f"Semantic class map has unsupported shape: {labels.shape}")
    if np.issubdtype(labels.dtype, np.floating) and (
        not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
    ):
        raise PackageError(
            "Semantic class-map output must contain finite integer labels"
        )
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0 or labels.min() < 0:
        raise PackageError("Semantic class-map output must contain non-negative labels")
    return labels


def encode_label_mask(labels: np.ndarray) -> dict[str, list[int]]:
    """Serialize opaque class indices as a compact, JSON-only row-major RLE."""

    value = np.asarray(labels)
    if value.ndim != 2 or value.size == 0:
        raise PackageError(
            f"Semantic label mask must be a non-empty 2D array: {value.shape}"
        )
    if np.issubdtype(value.dtype, np.floating) and (
        not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all()
    ):
        raise PackageError("Semantic label mask must contain finite integer labels")
    flat = np.asarray(value, dtype=np.int64).reshape(-1)
    if flat.min() < 0:
        raise PackageError("Semantic label mask must contain non-negative labels")
    starts = np.r_[True, flat[1:] != flat[:-1]]
    offsets = np.flatnonzero(starts)
    counts = np.diff(np.r_[offsets, flat.size])
    return {
        "size": [int(value.shape[0]), int(value.shape[1])],
        "values": [int(label) for label in flat[offsets]],
        "counts": [int(count) for count in counts],
    }


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    pass


def prediction_records(outputs, customer_contract, geometry, args, image_id):
    layout = "channels_last" if args.framework == "mxq" else "channels_first"
    return [
        {"image_id": image_id, "mask": encode_label_mask(class_map(outputs[0], layout))}
    ]


def summarize(
    outputs, customer_contract, geometry, args, output_dir: Path, source: Path
):
    layout = "channels_last" if args.framework == "mxq" else "channels_first"
    labels = class_map(outputs[0], layout)
    target = output_dir / f"{source.stem}.labels.npy"
    np.save(target, labels)
    return {
        "task": customer_contract["task"],
        "labels_npy": str(target),
        "shape": list(labels.shape),
    }
