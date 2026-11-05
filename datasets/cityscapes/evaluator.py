# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Cityscapes scoring for generic semantic label-mask prediction artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from semantic import (
    SemanticMetric,
    SemanticResult,
    artifact_records,
    decode_label_mask,
    letterbox_target,
)

from dataloader import CityscapesSample, validation_samples

CLASS_COUNT = 19

# Official Cityscapes ``labelIds`` corresponding to contiguous train IDs 0--18.
# All remaining source IDs are void labels and must stay at the semantic
# evaluator's ignore value (255).
SOURCE_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
SOURCE_TO_TRAIN_ID[[7, 8, 11, 12, 13, 17, *range(19, 29), 31, 32, 33]] = np.arange(
    19, dtype=np.uint8
)


def load_target(sample: CityscapesSample) -> np.ndarray:
    """Load a raw Cityscapes labelIds mask as a 19-class train-ID mask."""
    target = cv2.imread(str(sample.target), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(f"Unable to read Cityscapes target: {sample.target}")
    # OpenCV's ``MatLike`` type permits several array dtypes, while the LUT is
    # explicitly indexed by uint8 Cityscapes label IDs.
    label_ids = np.asarray(target, dtype=np.uint8)
    return SOURCE_TO_TRAIN_ID[label_ids]


def evaluate(
    data_path: Path, predict: Callable[[CityscapesSample], np.ndarray]
) -> SemanticResult:
    metric = SemanticMetric(CLASS_COUNT)
    for sample in validation_samples(data_path):
        prediction = predict(sample)
        metric.update(
            prediction, letterbox_target(load_target(sample), prediction.shape)
        )
    return metric.result()


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Score opaque Model labels after Cityscapes maps its own source labels."""

    image_ids, records = artifact_records(artifact, "Cityscapes")
    samples = validation_samples(Path(data_path))
    selected: list[tuple[int, CityscapesSample]] = []
    for image_id in image_ids:
        if image_id < 0 or image_id >= len(samples):
            raise ValueError(
                f"Prediction artifact image_id is outside Cityscapes: {image_id}"
            )
        selected.append((image_id, samples[image_id]))
    metric = SemanticMetric(CLASS_COUNT)
    for image_id, sample in selected:
        prediction = decode_label_mask(records[image_id].get("mask"), CLASS_COUNT)
        target = letterbox_target(load_target(sample), prediction.shape)
        metric.update(prediction, target)
    result = metric.result()
    return {
        "primary": {"miou": round(result.miou, 5)},
        "secondary": {"pixel_accuracy": round(result.pixel_accuracy, 5)},
        "sample_count": len(selected),
    }
