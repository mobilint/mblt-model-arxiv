"""Dataset-side primitives for scoring semantic-segmentation artifacts.

The Model wire contains a compact run-length representation of an opaque
integer label map.  This module deliberately owns the inverse operation and
the metric: a Model can emit labels without importing a benchmark, while each
benchmark retains control of its target encoding and valid label range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SemanticResult:
    """The two standard dense-semantic benchmark scores."""

    miou: float
    pixel_accuracy: float

    @property
    def primary_score(self) -> float:
        return self.miou

    @property
    def secondary_score(self) -> float:
        return self.pixel_accuracy


class SemanticMetric:
    """Accumulate a benchmark's confusion matrix over valid target pixels."""

    def __init__(self, class_count: int, ignore_label: int = 255) -> None:
        if class_count <= 0:
            raise ValueError("class_count must be positive")
        self.class_count = class_count
        self.ignore_label = ignore_label
        self.matrix = np.zeros((class_count, class_count), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Semantic shape mismatch: {prediction.shape} != {target.shape}"
            )
        valid = target != self.ignore_label
        valid_target = target[valid].astype(np.int64, copy=False)
        valid_prediction = prediction[valid].astype(np.int64, copy=False)
        if valid_target.size:
            if valid_target.min() < 0 or valid_target.max() >= self.class_count:
                raise ValueError("Semantic target contains an invalid class ID.")
            if valid_prediction.min() < 0 or valid_prediction.max() >= self.class_count:
                raise ValueError("Semantic prediction contains an invalid class ID.")
        histogram = np.bincount(
            self.class_count * valid_target + valid_prediction,
            minlength=self.class_count**2,
        )
        self.matrix += histogram.reshape(self.class_count, self.class_count)

    def result(self) -> SemanticResult:
        ground_truth, predicted = self.matrix.sum(1), self.matrix.sum(0)
        intersection = np.diag(self.matrix)
        union = ground_truth + predicted - intersection
        present = ground_truth > 0
        if not present.any():
            raise ValueError("Semantic evaluation received no valid target pixels.")
        iou = np.divide(
            intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0
        )
        return SemanticResult(
            miou=float(iou[present].mean()),
            pixel_accuracy=float(intersection.sum() / self.matrix.sum()),
        )


def letterbox_target(
    target: np.ndarray, shape: tuple[int, int], ignore_label: int = 255
) -> np.ndarray:
    """Place a benchmark target into the Model prediction's letterbox geometry."""

    target_height, target_width = shape
    height, width = target.shape
    ratio = min(target_height / height, target_width / width)
    resized = cv2.resize(
        target,
        (round(width * ratio), round(height * ratio)),
        interpolation=cv2.INTER_NEAREST,
    )
    pad_width, pad_height = (
        target_width - resized.shape[1],
        target_height - resized.shape[0],
    )
    left, top = pad_width // 2, pad_height // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        pad_height - top,
        left,
        pad_width - left,
        cv2.BORDER_CONSTANT,
        value=ignore_label,
    )


def decode_label_mask(payload: Any, class_count: int) -> np.ndarray:
    """Decode the generic JSON run-length label-map wire into a dense mask."""

    if not isinstance(payload, dict):
        raise TypeError("Semantic prediction mask must be an object")
    size, values, counts = (
        payload.get("size"),
        payload.get("values"),
        payload.get("counts"),
    )
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in size)
        or not isinstance(values, list)
        or not isinstance(counts, list)
        or len(values) != len(counts)
        or not values
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        )
        or any(
            not isinstance(count, int) or isinstance(count, bool) for count in counts
        )
    ):
        raise TypeError(
            "Semantic mask must contain integer size, values, and counts arrays"
        )
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError("Semantic mask size must be positive")
    if any(value < 0 or value >= class_count for value in values):
        raise ValueError("Semantic prediction contains an invalid class ID")
    if any(count <= 0 for count in counts) or sum(counts) != height * width:
        raise ValueError("Semantic mask runs do not cover its declared size")
    return np.repeat(
        np.asarray(values, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    ).reshape(height, width)


def artifact_records(
    artifact: dict[str, Any], benchmark: str
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Validate the task-neutral outer wire and index one mask per image id."""

    task = "semantic_segmentation"
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
            raise TypeError("Semantic prediction record image_id must be an integer")
        if image_id in by_id:
            raise ValueError(
                f"Prediction artifact has duplicate record for image {image_id}"
            )
        by_id[image_id] = record
    if set(by_id) != set(image_ids):
        raise ValueError(
            "Prediction artifact needs exactly one mask record per image_id"
        )
    return image_ids, by_id
