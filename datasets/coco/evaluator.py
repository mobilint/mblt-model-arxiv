# /// script
# requires-python = ">=3.10"
# dependencies = ["faster-coco-eval", "pyyaml"]
# ///
"""COCO 2017 benchmark scoring for normalized prediction artifacts.

This module owns the COCO annotation convention, class-index mapping, and
metric. It deliberately accepts only JSON-compatible artifact data, never a
Model module or Model-specific Python type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faster_coco_eval
import yaml
from faster_coco_eval import COCO, COCOeval_faster

# The dataset's own directory: home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

COCOIoUType = Literal["bbox", "segm", "keypoints"]
_SUPPORTED_TASKS = {"object_detection", "instance_segmentation", "pose_estimation"}


@dataclass(frozen=True)
class COCOResult:
    map50_95: float
    map50: float

    @property
    def primary_score(self) -> float:
        """Return COCO mAP averaged over IoU 0.50 through 0.95."""
        return self.map50_95

    @property
    def secondary_score(self) -> float:
        """Return COCO mAP at IoU 0.50."""
        return self.map50


def annotation_path(data_path: Path, task: str) -> Path:
    if task not in _SUPPORTED_TASKS:
        raise ValueError(f"Unsupported COCO task: {task!r}")
    name = (
        "person_keypoints_val2017.json"
        if task == "pose_estimation"
        else "instances_val2017.json"
    )
    path = data_path / name
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation file is missing: {path}")
    return path


def evaluate_predictions(
    data_path: Path,
    task: str,
    predictions: list[dict[str, Any]],
    image_ids: list[int] | None = None,
) -> COCOResult:
    """Score standard COCO bbox, segmentation, or keypoint result dictionaries."""
    ground_truth = COCO(str(annotation_path(data_path, task)))
    detections = ground_truth.loadRes(predictions) if predictions else COCO()
    if task == "object_detection":
        iou_type: COCOIoUType = "bbox"
    elif task == "instance_segmentation":
        iou_type = "segm"
    else:
        iou_type = "keypoints"
    evaluator = COCOeval_faster(ground_truth, detections, iou_type)
    if image_ids is not None:
        evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return COCOResult(
        map50_95=float(evaluator.stats[0]), map50=float(evaluator.stats[1])
    )


def coco_category_ids() -> list[int]:
    config_path = DATASET_HOME / "coco.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    category_ids = config.get("category_ids") if isinstance(config, dict) else None
    if (
        not isinstance(category_ids, list)
        or not category_ids
        or any(not isinstance(category_id, int) for category_id in category_ids)
    ):
        raise ValueError(f"COCO category_ids are invalid: {config_path}")
    return category_ids


def _coco_mask(mask: Any) -> dict[str, Any]:
    """Convert the artifact's JSON RLE into the format COCOeval consumes."""
    if not isinstance(mask, dict):
        raise TypeError("Prediction mask must be an object")
    size, counts = mask.get("size"), mask.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(not isinstance(value, int) for value in size)
        or not isinstance(counts, list)
        or any(not isinstance(value, int) for value in counts)
    ):
        raise TypeError("Prediction mask must be an uncompressed COCO RLE object")
    encoded = faster_coco_eval.mask.frPyObjects(
        {"size": size, "counts": counts}, size[0], size[1]
    )
    if isinstance(encoded, list):
        if len(encoded) != 1:
            raise ValueError("Prediction mask must encode exactly one instance")
        encoded = encoded[0]
    if not isinstance(encoded, dict):
        raise TypeError("COCO RLE conversion did not produce an object")
    if isinstance(encoded.get("counts"), bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def standard_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Model class indexes and generic masks into COCO result dictionaries."""
    configured_ids = coco_category_ids()
    results: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction artifact records must be objects")
        label = record.get("label")
        if not isinstance(label, int) or isinstance(label, bool):
            raise TypeError("Prediction artifact record label must be an integer")
        if not 0 <= label < len(configured_ids):
            raise ValueError(
                f"Prediction artifact label is outside COCO's classes: {label}"
            )
        result = {
            key: value for key, value in record.items() if key not in {"label", "mask"}
        }
        result["category_id"] = configured_ids[label]
        if "mask" in record:
            result["segmentation"] = _coco_mask(record["mask"])
        results.append(result)
    return results


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Score a JSON-wire prediction artifact and return a JSON-wire result."""
    task = artifact.get("task")
    image_ids = artifact.get("image_ids")
    records = artifact.get("records")
    if not isinstance(task, str):
        raise TypeError("Prediction artifact task must be a string")
    if not isinstance(image_ids, list) or any(
        not isinstance(image_id, int) or isinstance(image_id, bool)
        for image_id in image_ids
    ):
        raise TypeError("Prediction artifact image_ids must be an array of integers")
    if not isinstance(records, list):
        raise TypeError("Prediction artifact records must be an array")
    result = evaluate_predictions(
        Path(data_path), task, standard_predictions(records), image_ids
    )
    return {
        "primary": {"map50_95": round(result.map50_95, 5)},
        "secondary": {"map50": round(result.map50, 5)},
        "sample_count": len(image_ids),
    }
