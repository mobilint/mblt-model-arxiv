"""Dataset-side decoding for depth-estimation prediction artifacts.

Depth Model adapters serialize a two-dimensional float32 map plus the
LetterBox geometry that produced it. This module owns the inverse JSON wire;
the benchmark that consumes it remains responsible for target transforms and
metrics. Models deliberately do not import this module.
"""

from __future__ import annotations

import base64
import binascii
import math
import zlib
from typing import Any

import numpy as np

_ENCODING = "f32le-zlib-base64"
_MAX_PIXELS = 100_000_000


def decode_depth_map(payload: Any) -> np.ndarray:
    """Decode one opaque float32 depth map emitted by a Model adapter."""

    if not isinstance(payload, dict):
        raise TypeError("Depth prediction must be an object")
    encoding, size, data = (
        payload.get("encoding"),
        payload.get("size"),
        payload.get("data"),
    )
    if encoding != _ENCODING:
        raise ValueError(f"Unsupported depth prediction encoding: {encoding!r}")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in size)
    ):
        raise TypeError("Depth prediction size must be a two-item integer array")
    height, width = size
    if height <= 0 or width <= 0 or height * width > _MAX_PIXELS:
        raise ValueError("Depth prediction size is outside the supported range")
    if not isinstance(data, str):
        raise TypeError("Depth prediction data must be a base64 string")
    try:
        compressed = base64.b64decode(data.encode("ascii"), validate=True)
        raw = zlib.decompress(compressed)
    except (UnicodeEncodeError, binascii.Error, zlib.error) as error:
        raise ValueError(
            "Depth prediction data is not valid compressed base64"
        ) from error
    expected_bytes = height * width * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        raise ValueError("Depth prediction byte count does not match its declared size")
    return np.frombuffer(raw, dtype="<f4").reshape(height, width)


def decode_geometry(payload: Any) -> tuple[float, tuple[float, float]]:
    """Decode the Model's generic LetterBox geometry record."""

    if not isinstance(payload, dict):
        raise TypeError("Depth prediction geometry must be an object")
    ratio, pad = payload.get("ratio"), payload.get("pad")
    values = (ratio, *(pad if isinstance(pad, list) and len(pad) == 2 else ()))
    if len(values) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise TypeError("Depth prediction geometry requires numeric ratio and pad")
    ratio, left, top = (float(value) for value in values)
    if (
        not math.isfinite(ratio)
        or not math.isfinite(left)
        or not math.isfinite(top)
        or ratio <= 0
        or left < 0
        or top < 0
    ):
        raise ValueError("Depth prediction geometry is outside the supported range")
    return ratio, (left, top)


def artifact_records(
    artifact: dict[str, Any], benchmark: str
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Validate the task-neutral outer wire and index one record per image."""

    task = "depth_estimation"
    if artifact.get("task") != task:
        raise ValueError(
            f"{benchmark} requires task {task!r}, got {artifact.get('task')!r}"
        )
    image_ids, records = artifact.get("image_ids"), artifact.get("records")
    if not isinstance(image_ids, list) or any(
        not isinstance(image_id, int) or isinstance(image_id, bool)
        for image_id in image_ids
    ):
        raise TypeError("Prediction artifact image_ids must be an array of integers")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Prediction artifact image_ids must be unique")
    if not isinstance(records, list):
        raise TypeError("Prediction artifact records must be an array")
    by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction artifact records must be objects")
        image_id = record.get("image_id")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise TypeError("Depth prediction record image_id must be an integer")
        if image_id in by_id:
            raise ValueError(
                f"Prediction artifact has duplicate record for image {image_id}"
            )
        by_id[image_id] = record
    if set(by_id) != set(image_ids):
        raise ValueError(
            "Prediction artifact needs exactly one depth record per image_id"
        )
    return image_ids, by_id
