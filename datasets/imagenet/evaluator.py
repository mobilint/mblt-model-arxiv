"""ImageNet-1k benchmark scoring for normalized classification artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataloader import iter_validation_samples


@dataclass(frozen=True)
class ImageNetResult:
    top1_accuracy: float
    top5_accuracy: float

    @property
    def primary_score(self) -> float:
        """Return Top-1 classification accuracy."""
        return self.top1_accuracy

    @property
    def secondary_score(self) -> float:
        """Return Top-5 classification accuracy."""
        return self.top5_accuracy


def _topk_by_image_id(records: list[dict[str, Any]]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction artifact records must be objects")
        image_id, topk = record.get("image_id"), record.get("topk")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise TypeError("Classification prediction image_id must be an integer")
        if (
            not isinstance(topk, list)
            or len(topk) != 5
            or any(
                not isinstance(label, int) or isinstance(label, bool) for label in topk
            )
            or any(not 0 <= label < 1000 for label in topk)
            or len(set(topk)) != len(topk)
        ):
            raise TypeError(
                "Classification prediction topk must contain five unique class indexes"
            )
        if image_id in result:
            raise ValueError(f"Prediction artifact has duplicate image_id: {image_id}")
        result[image_id] = topk
    return result


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Score Model top-k records against ImageNet's benchmark-owned labels."""
    if artifact.get("task") != "image_classification":
        raise ValueError(
            "ImageNet accepts image_classification prediction artifacts only"
        )
    image_ids = artifact.get("image_ids")
    records = artifact.get("records")
    if not isinstance(image_ids, list) or any(
        not isinstance(image_id, int) or isinstance(image_id, bool)
        for image_id in image_ids
    ):
        raise TypeError("Prediction artifact image_ids must be an array of integers")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Prediction artifact image_ids must be unique")
    if not isinstance(records, list):
        raise TypeError("Prediction artifact records must be an array")
    predictions = _topk_by_image_id(records)
    top1 = top5 = count = 0
    for image_id, sample in enumerate(iter_validation_samples(Path(data_path))):
        if image_id not in predictions:
            raise ValueError(
                f"Prediction artifact is missing ImageNet image_id {image_id}"
            )
        topk = predictions.pop(image_id)
        top1 += int(topk[0] == sample.label)
        top5 += int(sample.label in topk)
        count += 1
    if predictions:
        raise ValueError(
            f"Prediction artifact has unknown ImageNet image_ids: {sorted(predictions)[:5]}"
        )
    if image_ids != list(range(count)):
        raise ValueError(
            "Prediction artifact image_ids do not match ImageNet manifest order"
        )
    if not count:
        raise ValueError(f"ImageNet validation set is empty: {data_path}")
    result = ImageNetResult(top1 / count, top5 / count)
    return {
        "primary": {"top1_accuracy": round(result.top1_accuracy, 5)},
        "secondary": {"top5_accuracy": round(result.top5_accuracy, 5)},
        "sample_count": count,
    }
