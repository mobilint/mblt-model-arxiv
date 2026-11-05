# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""NYU Depth V2 scoring for generic depth prediction artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from depth import artifact_records, decode_depth_map, decode_geometry

from dataloader import NYUDepthSample, validation_samples


@dataclass(frozen=True)
class NYUDepthResult:
    delta1: float
    abs_rel: float
    rmse: float

    @property
    def primary_score(self) -> float:
        """Return threshold accuracy delta1."""
        return self.delta1

    @property
    def secondary_score(self) -> float:
        """Return absolute relative error for singular-score compatibility."""
        return self.abs_rel

    @property
    def secondary_scores(self) -> tuple[float, float]:
        """Return absolute-relative error and RMSE in report order."""
        return self.abs_rel, self.rmse


class NYUDepthMetric:
    MIN_DEPTH, MAX_DEPTH = 0.001, 100.0

    def __init__(self) -> None:
        self.delta1_sum = self.abs_rel_sum = self.rmse_sum = 0.0
        self.valid_sample_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction, target = (
            np.asarray(prediction, dtype=np.float32),
            np.asarray(target, dtype=np.float32),
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"NYU Depth shape mismatch: {prediction.shape} != {target.shape}"
            )
        valid = (
            np.isfinite(target) & (target > self.MIN_DEPTH) & (target < self.MAX_DEPTH)
        )
        if not valid.any():
            raise ValueError("NYU Depth sample contains no valid target pixels.")
        predicted, actual = prediction[valid], target[valid]
        if not np.isfinite(predicted).all():
            raise ValueError("NYU Depth prediction contains non-finite values.")
        median_prediction = float(np.median(np.maximum(predicted, self.MIN_DEPTH)))
        if median_prediction <= 0:
            raise ValueError("NYU Depth prediction has a non-positive median.")
        aligned = np.clip(
            predicted * (np.median(actual) / median_prediction),
            self.MIN_DEPTH,
            self.MAX_DEPTH,
        )
        ratio = np.maximum(actual / aligned, aligned / actual)
        # Keep each validation image equally weighted, matching Ultralytics'
        # depth validator after its per-image median alignment.
        self.delta1_sum += float(np.mean(ratio < 1.25))
        self.abs_rel_sum += float(np.mean(np.abs(actual - aligned) / actual))
        self.rmse_sum += float(np.sqrt(np.mean((actual - aligned) ** 2)))
        self.valid_sample_count += 1

    def result(self) -> NYUDepthResult:
        if not self.valid_sample_count:
            raise ValueError("NYU Depth evaluation received no valid pixels.")
        return NYUDepthResult(
            self.delta1_sum / self.valid_sample_count,
            self.abs_rel_sum / self.valid_sample_count,
            self.rmse_sum / self.valid_sample_count,
        )


def evaluate(
    data_path: Path, predict: Callable[[NYUDepthSample], np.ndarray]
) -> NYUDepthResult:
    """Score a direct benchmark callback without constructing a Model runtime."""

    metric = NYUDepthMetric()
    for sample in validation_samples(data_path):
        target = np.load(sample.target, allow_pickle=False)
        metric.update(predict(sample), target)
    return metric.result()


def _letterbox_target(
    target: np.ndarray,
    shape: tuple[int, int],
    geometry: tuple[float, tuple[float, float]],
) -> np.ndarray:
    """Map an NYU target through the Model-provided LetterBox transform.

    Padding is not an image pixel. It remains NaN so ``NYUDepthMetric``
    excludes it from every per-image calculation.
    """

    value = np.asarray(target, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"NYU Depth target has unsupported shape: {value.shape}")
    ratio, (left, top) = geometry
    resized_width = round(value.shape[1] * ratio)
    resized_height = round(value.shape[0] * ratio)
    left_int, top_int = round(left), round(top)
    right, bottom = left_int + resized_width, top_int + resized_height
    if (
        ratio <= 0
        or left_int < 0
        or top_int < 0
        or right > shape[1]
        or bottom > shape[0]
    ):
        raise ValueError(
            "NYU Depth LetterBox geometry is incompatible with the model input: "
            f"shape={shape}, geometry={geometry}, target={value.shape}"
        )
    result = np.full(shape, np.nan, dtype=np.float32)
    result[top_int:bottom, left_int:right] = cv2.resize(
        value,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return result


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Score depth maps and preprocessing geometry against NYU's own targets."""

    image_ids, records = artifact_records(artifact, "NYU Depth")
    samples = validation_samples(Path(data_path))
    selected: list[tuple[int, NYUDepthSample]] = []
    for image_id in image_ids:
        if image_id < 0 or image_id >= len(samples):
            raise ValueError(
                f"Prediction artifact image_id is outside NYU Depth: {image_id}"
            )
        selected.append((image_id, samples[image_id]))
    metric = NYUDepthMetric()
    for image_id, sample in selected:
        prediction = decode_depth_map(records[image_id].get("depth"))
        geometry = decode_geometry(records[image_id].get("geometry"))
        target = np.load(sample.target, allow_pickle=False).astype(np.float32)
        metric.update(prediction, _letterbox_target(target, prediction.shape, geometry))
    result = metric.result()
    return {
        "primary": {"delta1": round(result.delta1, 5)},
        "secondary": {
            "abs_rel": round(result.abs_rel, 5),
            "rmse": round(result.rmse, 5),
        },
        "sample_count": len(selected),
    }
