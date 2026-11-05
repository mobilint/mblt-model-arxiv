# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""DOTAv1 evaluator contracts and average-precision integration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataloader import DOTASample, validation_samples


@dataclass(frozen=True)
class DOTAResult:
    map50_95: float
    map50: float

    @property
    def primary_score(self) -> float:
        """Return rotated mAP at IoU 0.50, the metric Ultralytics publishes for DOTAv1."""
        return self.map50

    @property
    def secondary_score(self) -> float:
        """Return rotated mAP averaged over IoU 0.50 through 0.95."""
        return self.map50_95


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Integrate a monotonic precision envelope on the 101-point COCO grid."""
    recall, precision = (
        np.asarray(recall, dtype=float),
        np.asarray(precision, dtype=float),
    )
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0.0, 1.0, 101)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(np.interp(grid, mrec, mpre), grid))


def summarize_average_precision(ap: np.ndarray) -> DOTAResult:
    """Summarize a [classes, IoU thresholds] rotated-box AP matrix."""
    values = np.asarray(ap, dtype=float)
    if values.ndim != 2 or values.shape[1] != 10 or values.shape[0] == 0:
        raise ValueError(f"DOTA AP must have shape [classes, 10], got {values.shape}")
    return DOTAResult(map50_95=float(values.mean()), map50=float(values[:, 0].mean()))


def _probiou(first: np.ndarray, second: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pairwise probabilistic IoU for xywhr boxes (Ultralytics probIoU).

    This independently implements the same rotated-IoU definition as the
    Model decoder and mblt-vision's ``batch_probiou``. Only working precision
    differs, because a scorer can afford float64 for the ``a * b - c ** 2``
    cancellation. Three details are load-bearing and were each wrong here
    before:

    * the mean-difference cross term pairs ``x2 - x1`` with ``y1 - y2``, so it
      *subtracts* from the Bhattacharyya quadratic form. Writing both
      differences the same way round flips its sign, which moved this IoU by up
      to 0.7 on the nearby elongated boxes DOTA is full of and carried matches
      across the 0.5 line;
    * the log ratio takes the covariance determinant itself, not the
      ``eps``-padded ``denominator`` the two quadratic terms divide by;
    * each per-box determinant is clamped at zero on its own. Clamping their
      product instead lets two rounding-negative determinants multiply back to
      a positive and shrink the term.

    Rows come from ``first`` (targets), columns from ``second`` (predictions).
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)

    def covariance(boxes):
        w2, h2, angle = boxes[:, 2] ** 2 / 12, boxes[:, 3] ** 2 / 12, boxes[:, 4]
        cosine, sine = np.cos(angle), np.sin(angle)
        return (
            w2 * cosine**2 + h2 * sine**2,
            w2 * sine**2 + h2 * cosine**2,
            (w2 - h2) * cosine * sine,
        )

    x1, y1 = first[:, 0][:, None], first[:, 1][:, None]
    x2, y2 = second[:, 0][None, :], second[:, 1][None, :]
    a1, b1, c1 = (value[:, None] for value in covariance(first))
    a2, b2, c2 = (value[None, :] for value in covariance(second))
    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denominator * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator * 0.5
    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
            / (
                4.0
                * np.sqrt(
                    np.clip(a1 * b1 - c1**2, 0.0, None)
                    * np.clip(a2 * b2 - c2**2, 0.0, None)
                )
                + eps
            )
            + eps
        )
        * 0.5
    )
    distance = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1.0 - np.sqrt(1.0 - np.exp(-distance) + eps)


CLASS_IDS = {
    name: index
    for index, name in enumerate(
        (
            "plane",
            "ship",
            "storage-tank",
            "baseball-diamond",
            "tennis-court",
            "basketball-court",
            "ground-track-field",
            "harbor",
            "bridge",
            "large-vehicle",
            "small-vehicle",
            "helicopter",
            "roundabout",
            "soccer-ball-field",
            "swimming-pool",
        )
    )
}

# The Model wire carries its own opaque label index. DOTAv1 alone defines how
# that index aligns with benchmark classes; it happens to be identity today.
ARTIFACT_LABEL_TO_CLASS_ID = {
    label: CLASS_IDS[name] for label, name in enumerate(CLASS_IDS)
}


def _ground_truth(sample: DOTASample) -> tuple[np.ndarray, np.ndarray]:
    """Read every labelled object, the difficult ones included, as a target.

    The difficulty flag (field 9) is validated but not used: Ultralytics, whose
    checkpoints these are, converts DOTA labels with `convert_dota_to_yolo_obb`,
    which keeps the eight coordinates and the class name and drops the flag, so
    its validation scores difficult objects as ordinary targets. The DOTA devkit
    behind the test server ignores them instead. mblt-vision-python counts them
    too from 0.0.7 (mobilint/mblt-vision-python#25).
    """
    boxes, classes = [], []
    for line_number, line in enumerate(
        sample.original_label.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.split()
        if not fields or fields[0].lower().startswith(("imagesource", "gsd")):
            continue
        if (
            len(fields) < 10
            or fields[8] not in CLASS_IDS
            or fields[9]
            not in {
                "0",
                "1",
                "2",
            }
        ):
            raise ValueError(
                f"Malformed DOTAv1 target at {sample.original_label}:{line_number}"
            )
        points = np.asarray(
            [float(value) for value in fields[:8]], dtype=np.float32
        ).reshape(4, 2)
        (center_x, center_y), (width, height), angle = cv2.minAreaRect(points)
        box = [center_x, center_y, width, height, np.deg2rad(angle)]
        boxes.append(box)
        classes.append(CLASS_IDS[fields[8]])
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
        np.asarray(classes, dtype=np.int64),
    )


def _match_thresholds(
    ious: np.ndarray,
    target_classes: np.ndarray,
    pred_classes: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Greedily match predictions to targets once for all ten IoU thresholds.

    Greedy matching over IoU-sorted same-class pairs is prefix-determined: the
    candidate set at a higher threshold is a prefix of the sorted candidate set
    at the lowest one, and greedy over a prefix makes exactly the matches the
    full pass makes before it leaves that prefix. So one pass that records the
    IoU each prediction matched at answers every threshold, instead of ten
    passes that re-sort and re-match.

    `kind="stable"` is what makes that argument true, and it is load bearing
    rather than tidy. With an unstable sort, equal IoUs can be ordered one way
    in the full candidate set and another once the lower-IoU entries are gone,
    so the prefix stops being the prefix and greedy picks different pairs.
    Measured over 4000 tie-heavy matrices: stable agrees with a stable
    per-threshold pass on all of them, an unstable sort disagrees on 20.

    Stability also removes a defect of the per-threshold form this replaces.
    Re-sorting each threshold independently let a prediction come out a true
    positive at IoU >= 0.75 while not being one at IoU >= 0.50 — 21
    non-monotone prediction rows in that same sweep. Correctness here is
    agreement with the stable ordering, not with a particular unstable run.
    """
    matched_iou = np.zeros(ious.shape[1])
    candidates = np.argwhere(
        (ious >= thresholds[0]) & (target_classes[:, None] == pred_classes)
    )
    if len(candidates):
        values = ious[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(values, kind="stable")[::-1]
        used_targets = np.zeros(ious.shape[0], dtype=bool)
        used_predictions = np.zeros(ious.shape[1], dtype=bool)
        for target_index, prediction_index in candidates[order]:
            if used_targets[target_index] or used_predictions[prediction_index]:
                continue
            used_targets[target_index] = True
            used_predictions[prediction_index] = True
            matched_iou[prediction_index] = ious[target_index, prediction_index]
    return matched_iou[:, None] >= thresholds


def evaluate_predictions(
    samples: tuple[DOTASample, ...],
    predictions: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> DOTAResult:
    thresholds = np.linspace(0.5, 0.95, 10)
    all_tp, all_scores, all_pred_classes, all_target_classes = [], [], [], []
    # Reading and parsing the original DOTA labels is pure per-sample file work,
    # so it is prefetched off the scoring loop instead of stalling it.
    with ThreadPoolExecutor(max_workers=8) as pool:
        ground_truth = list(pool.map(_ground_truth, samples))
    for (pred_boxes, scores, pred_classes), (target_boxes, target_classes) in zip(
        predictions, ground_truth, strict=True
    ):
        true_positive = np.zeros((len(pred_boxes), 10), dtype=bool)
        if len(pred_boxes) and len(target_boxes):
            true_positive = _match_thresholds(
                _probiou(target_boxes, pred_boxes),
                target_classes,
                pred_classes,
                thresholds,
            )
        all_tp.append(true_positive)
        all_scores.append(scores)
        all_pred_classes.append(pred_classes)
        all_target_classes.append(target_classes)
    target_classes = (
        np.concatenate(all_target_classes) if all_target_classes else np.zeros(0)
    )
    if not len(target_classes):
        raise ValueError("DOTAv1 evaluation contains no targets.")
    tp, scores, pred_classes = map(
        np.concatenate, (all_tp, all_scores, all_pred_classes)
    )
    unique_classes = np.unique(target_classes)
    ap = np.zeros((len(unique_classes), 10), dtype=float)
    for class_index, class_id in enumerate(unique_classes):
        order = np.flatnonzero(pred_classes == class_id)
        order = order[np.argsort(scores[order])[::-1]]
        target_count = int(np.sum(target_classes == class_id))
        for threshold_index in range(10):
            class_tp = tp[order, threshold_index]
            if len(class_tp):
                cumulative_tp = class_tp.cumsum()
                cumulative_fp = (~class_tp).cumsum()
                ap[class_index, threshold_index] = compute_ap(
                    cumulative_tp / target_count,
                    cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1),
                )
    return summarize_average_precision(ap)


def standard_predictions(
    data_path: Path, artifact: dict[str, Any]
) -> tuple[
    tuple[DOTASample, ...],
    tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
]:
    """Attach generic OBB records to DOTAv1's validation sample order and classes."""
    if artifact.get("task") != "obb":
        raise ValueError("DOTAv1 accepts obb prediction artifacts only")
    image_ids = artifact.get("image_ids")
    records = artifact.get("records")
    if not isinstance(image_ids, list) or any(
        not isinstance(image_id, int) or isinstance(image_id, bool)
        for image_id in image_ids
    ):
        raise TypeError("Prediction artifact image_ids must be an array of integers")
    if not isinstance(records, list):
        raise TypeError("Prediction artifact records must be an array")

    samples = validation_samples(Path(data_path))
    if image_ids != list(range(len(samples))):
        raise ValueError(
            "Prediction artifact image_ids do not match DOTAv1 manifest order"
        )
    boxes_by_image: list[list[list[float]]] = [[] for _sample in samples]
    scores_by_image: list[list[float]] = [[] for _sample in samples]
    classes_by_image: list[list[int]] = [[] for _sample in samples]
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction artifact records must be objects")
        image_id, bbox, label, score = (
            record.get("image_id"),
            record.get("bbox"),
            record.get("label"),
            record.get("score"),
        )
        if (
            not isinstance(image_id, int)
            or isinstance(image_id, bool)
            or not 0 <= image_id < len(samples)
        ):
            raise ValueError(
                "Prediction artifact record image_id is outside the manifest"
            )
        if (
            not isinstance(bbox, list)
            or len(bbox) != 5
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in bbox
            )
        ):
            raise TypeError("OBB prediction bbox must contain five numbers")
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in ARTIFACT_LABEL_TO_CLASS_ID
        ):
            raise ValueError("OBB prediction label is outside the DOTAv1 class mapping")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TypeError("OBB prediction score must be a number")
        boxes_by_image[image_id].append([float(value) for value in bbox])
        scores_by_image[image_id].append(float(score))
        classes_by_image[image_id].append(ARTIFACT_LABEL_TO_CLASS_ID[label])

    predictions = tuple(
        (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
            np.asarray(scores, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
        )
        for boxes, scores, labels in zip(
            boxes_by_image, scores_by_image, classes_by_image, strict=True
        )
    )
    return samples, predictions


def _legacy_ground_truth(sample: DOTASample):
    boxes, classes, ignored_boxes, ignored_classes = [], [], [], []
    for line_number, line in enumerate(
        sample.original_label.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.split()
        if not fields or fields[0].lower().startswith(("imagesource", "gsd")):
            continue
        if len(fields) < 10 or fields[8] not in CLASS_IDS:
            raise ValueError(
                f"Malformed DOTAv1 target at {sample.original_label}:{line_number}"
            )
        points = np.asarray(
            [float(value) for value in fields[:8]], dtype=np.float32
        ).reshape(4, 2)
        (center_x, center_y), (width, height), angle = cv2.minAreaRect(points)
        box = [center_x, center_y, width, height, np.deg2rad(angle)]
        target_boxes, target_classes = (
            (ignored_boxes, ignored_classes)
            if fields[9] in {"1", "2"}
            else (boxes, classes)
        )
        target_boxes.append(box)
        target_classes.append(CLASS_IDS[fields[8]])
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
        np.asarray(classes, dtype=np.int64),
        np.asarray(ignored_boxes, dtype=np.float32).reshape(-1, 5),
        np.asarray(ignored_classes, dtype=np.int64),
    )


def evaluate_legacy_predictions(samples, predictions) -> DOTAResult:
    thresholds = np.linspace(0.5, 0.95, 10)
    all_tp, all_ignore, all_scores, all_pred_classes, all_target_classes = (
        [],
        [],
        [],
        [],
        [],
    )
    # Reading and parsing the original DOTA labels is pure per-sample file work,
    # so it is prefetched off the scoring loop instead of stalling it.
    with ThreadPoolExecutor(max_workers=8) as pool:
        ground_truth = list(pool.map(_legacy_ground_truth, samples))
    for (pred_boxes, scores, pred_classes), (
        target_boxes,
        target_classes,
        ignored_boxes,
        ignored_classes,
    ) in zip(predictions, ground_truth, strict=True):
        true_positive = np.zeros((len(pred_boxes), 10), dtype=bool)
        ignored = np.zeros_like(true_positive)
        if len(pred_boxes) and len(target_boxes):
            true_positive = _match_thresholds(
                _probiou(target_boxes, pred_boxes),
                target_classes,
                pred_classes,
                thresholds,
            )
        if len(pred_boxes) and len(ignored_boxes):
            ignored_iou = _probiou(ignored_boxes, pred_boxes)
            # A prediction is ignored at a threshold when some same-class
            # ignored target reaches it, which is the per-prediction maximum
            # over those targets -- one reduction rather than ten masked ones.
            best = np.where(
                ignored_classes[:, None] == pred_classes, ignored_iou, -1.0
            ).max(0)
            ignored = (best[:, None] >= thresholds) & ~true_positive
        all_tp.append(true_positive)
        all_ignore.append(ignored)
        all_scores.append(scores)
        all_pred_classes.append(pred_classes)
        all_target_classes.append(target_classes)
    target_classes = (
        np.concatenate(all_target_classes) if all_target_classes else np.zeros(0)
    )
    if not len(target_classes):
        raise ValueError("DOTAv1 evaluation contains no non-ignored targets.")
    tp, ignored, scores, pred_classes = map(
        np.concatenate, (all_tp, all_ignore, all_scores, all_pred_classes)
    )
    unique_classes = np.unique(target_classes)
    ap = np.zeros((len(unique_classes), 10), dtype=float)
    for class_index, class_id in enumerate(unique_classes):
        order = np.flatnonzero(pred_classes == class_id)
        order = order[np.argsort(scores[order])[::-1]]
        target_count = int(np.sum(target_classes == class_id))
        for threshold_index in range(10):
            keep = ~ignored[order, threshold_index]
            class_tp = tp[order, threshold_index][keep]
            if len(class_tp):
                cumulative_tp = class_tp.cumsum()
                cumulative_fp = (~class_tp).cumsum()
                ap[class_index, threshold_index] = compute_ap(
                    cumulative_tp / target_count,
                    cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1),
                )
    return summarize_average_precision(ap)


def score_artifact(
    data_path: Path, artifact: dict[str, Any], *, protocol: str | None = None
) -> dict[str, Any]:
    """Score a JSON-wire OBB artifact with DOTAv1's rotated mAP metric."""
    samples, predictions = standard_predictions(Path(data_path), artifact)
    if protocol is None:
        result = evaluate_predictions(samples, predictions)
    elif protocol == "dotav1-ignore-difficult.v1":
        result = evaluate_legacy_predictions(samples, predictions)
    else:
        raise ValueError(f"Unknown DOTAv1 evaluation protocol: {protocol!r}")
    return {
        "primary": {"map50": round(result.map50, 5)},
        "secondary": {"map50_95": round(result.map50_95, 5)},
        "sample_count": len(samples),
    }
