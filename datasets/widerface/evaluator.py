# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""WIDER FACE benchmark scoring for normalized face-detection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from dataloader import metadata_file, validation_samples


@dataclass(frozen=True)
class WiderFaceResult:
    easy_ap: float
    medium_ap: float
    hard_ap: float

    @property
    def primary_score(self) -> float:
        """Return AP on the Hard split."""
        return self.hard_ap

    @property
    def secondary_score(self) -> float:
        """Return AP on the Medium split for singular-score compatibility."""
        return self.medium_ap

    @property
    def secondary_scores(self) -> tuple[float, float]:
        """Return Medium and Easy AP in secondary-metric order."""
        return self.medium_ap, self.easy_ap


def bbox_overlaps(boxes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Calculate pairwise IoU for xyxy boxes."""
    boxes, targets = np.asarray(boxes, dtype=float), np.asarray(targets, dtype=float)
    if (
        boxes.ndim != 2
        or targets.ndim != 2
        or boxes.shape[1] < 4
        or targets.shape[1] < 4
    ):
        raise ValueError("WiderFace boxes must have shape [N, >=4].")
    left_top = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    right_bottom = np.minimum(boxes[:, None, 2:4], targets[None, :, 2:4])
    intersection = np.maximum(right_bottom - left_top + 1.0, 0.0).prod(2)
    box_area = np.maximum(boxes[:, 2] - boxes[:, 0] + 1.0, 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1] + 1.0, 0.0
    )
    target_area = np.maximum(targets[:, 2] - targets[:, 0] + 1.0, 0.0) * np.maximum(
        targets[:, 3] - targets[:, 1] + 1.0, 0.0
    )
    return intersection / np.maximum(
        box_area[:, None] + target_area[None] - intersection, 1e-12
    )


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute the official WiderFace VOC-style area under the PR curve."""
    mrec = np.concatenate(([0.0], np.asarray(recall, dtype=float), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precision, dtype=float), [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _ground_truth(data_path: Path) -> tuple[Any, ...]:
    validation = loadmat(metadata_file(data_path, "wider_face_val.mat"))
    return (
        validation["face_bbx_list"],
        validation["event_list"],
        validation["file_list"],
        loadmat(metadata_file(data_path, "wider_easy_val.mat"))["gt_list"],
        loadmat(metadata_file(data_path, "wider_medium_val.mat"))["gt_list"],
        loadmat(metadata_file(data_path, "wider_hard_val.mat"))["gt_list"],
    )


def _normalize_scores(predictions: dict[str, dict[str, np.ndarray]]) -> None:
    populated = [
        value
        for event in predictions.values()
        for value in event.values()
        if len(value)
    ]
    if not populated:
        return
    minimum = min(float(value[:, 4].min()) for value in populated)
    maximum = max(float(value[:, 4].max()) for value in populated)
    if maximum <= minimum:
        return
    for value in populated:
        value[:, 4] = (value[:, 4] - minimum) / (maximum - minimum)


@dataclass(frozen=True)
class _ImageMatches:
    """Per-image matching state that every difficulty split shares.

    Which faces a split ignores is the only thing that changes between Easy,
    Medium and Hard, so the IoU matrix, each prediction's best-matching face
    and the score-threshold bookkeeping are computed once per image here and
    reused for all three splits instead of three times over.
    """

    best: np.ndarray
    """Index of the highest-IoU ground-truth face for each prediction."""
    matched: np.ndarray
    """Whether that best IoU reaches the official 0.5 match threshold."""
    target_count: int
    """Number of annotated faces in the image."""
    last: np.ndarray
    """Last prediction index at or above each score threshold, ``-1`` if none."""
    above: np.ndarray
    """Prediction count at or above each score threshold."""


def _score_threshold_index(
    scores: np.ndarray, threshold_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Locate every score threshold in one pass instead of scanning per cut-off.

    WiderFace walks ``threshold_count`` evenly spaced score cut-offs. The
    official curve needs two things per cut-off: the index of the *last*
    prediction that reaches it, which the running suffix maximum turns into a
    single ``searchsorted`` because the qualifying indices then form a prefix,
    and the plain count above it, which a sorted-score ``searchsorted`` gives.
    Predictions are not assumed to be sorted, so both are derived rather than
    read off the ends of the array.

    The cut-offs are cast to the score dtype, and that is the whole comparison,
    not housekeeping. The loop this replaces compared a float32 score array
    against a Python float, which numpy resolves in float32 by value-based
    casting, so a score stored as ``float32(0.9)`` *reaches* the 0.9 cut-off.
    Widening the scores to float64 instead makes the same score
    0.899999976... and drops it, which moves the proposal and recall counts at
    every cut-off a score lands on exactly -- and `_normalize_scores` maps the
    highest score to exactly 1.0, so landing there is routine rather than
    hypothetical. Integer scores are widened to float64, which is what the
    original comparison promoted them to.
    """
    values = np.asarray(scores)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    thresholds = (1.0 - np.arange(1, threshold_count + 1) / threshold_count).astype(
        values.dtype
    )
    suffix_maximum = np.maximum.accumulate(values[::-1])[::-1]
    last = np.searchsorted(-suffix_maximum, -thresholds, side="right") - 1
    above = len(values) - np.searchsorted(np.sort(values), thresholds, side="left")
    return last, above


def _image_matches(
    prediction: np.ndarray, target: np.ndarray, threshold_count: int
) -> _ImageMatches:
    """Match one image's predictions against its annotated faces."""
    target = np.asarray(target, dtype=np.float32)
    if target.shape == (1, 4) and not target.any():
        target = np.empty((0, 4), dtype=np.float32)
    last, above = _score_threshold_index(prediction[:, 4], threshold_count)
    if not len(target):
        return _ImageMatches(
            best=np.zeros(len(prediction), dtype=np.int64),
            matched=np.zeros(len(prediction), dtype=bool),
            target_count=0,
            last=last,
            above=above,
        )
    converted_prediction, converted_target = prediction.copy(), target.copy()
    converted_prediction[:, 2:4] += converted_prediction[:, :2]
    converted_target[:, 2:4] += converted_target[:, :2]
    overlaps = bbox_overlaps(converted_prediction[:, :4], converted_target)
    best = overlaps.argmax(1)
    return _ImageMatches(
        best=best,
        matched=overlaps[np.arange(len(prediction)), best] >= 0.5,
        target_count=len(target),
        last=last,
        above=above,
    )


def _image_counts(
    matches: _ImageMatches, keep: np.ndarray, threshold_count: int
) -> np.ndarray:
    """Per-threshold ``(proposal, recall)`` counts for one difficulty split."""
    output = np.zeros((threshold_count, 2), dtype=np.float32)
    if matches.target_count == 0:
        output[:, 0] = matches.above
        return output
    ignored = np.ones(matches.target_count, dtype=bool)
    if len(keep):
        ignored[np.asarray(keep, dtype=np.int64).reshape(-1) - 1] = False
    ignored_best = ignored[matches.best]
    # A prediction is only dropped from the proposal count when it matched a
    # face this split ignores; recall counts each non-ignored face once, at the
    # first prediction that matched it, which is what the official greedy loop
    # accumulates one prediction at a time.
    proposals = np.cumsum(~(matches.matched & ignored_best))
    hits = np.flatnonzero(matches.matched & ~ignored_best)
    first = np.zeros(len(matches.best), dtype=np.int64)
    if hits.size:
        first[hits[np.unique(matches.best[hits], return_index=True)[1]]] = 1
    recall = np.cumsum(first)
    reached = matches.last >= 0
    index = matches.last[reached]
    output[reached, 0] = proposals[index]
    output[reached, 1] = recall[index]
    return output


def evaluate_predictions(
    data_path: Path, predictions: dict[str, dict[str, np.ndarray]]
) -> WiderFaceResult:
    """Evaluate event/stem keyed xywh-score predictions with official difficulty metadata."""
    _normalize_scores(predictions)
    face_boxes, events, files, easy, medium, hard = _ground_truth(data_path)
    threshold_count = 1000
    difficulties = (easy, medium, hard)
    curves = [np.zeros((threshold_count, 2), dtype=np.float64) for _ in difficulties]
    face_counts = [0, 0, 0]
    # One pass over the images, three splits scored from the same matches: the
    # IoU matrix is the expensive part and it does not depend on which faces a
    # split ignores.
    for event_index in range(len(events)):
        event = str(events[event_index][0][0])
        if event not in predictions:
            raise ValueError(f"WiderFace predictions are missing event {event!r}.")
        for image_index, file_record in enumerate(files[event_index][0]):
            stem = str(file_record[0][0])
            if stem not in predictions[event]:
                raise ValueError(f"WiderFace predictions are missing {event}/{stem}.")
            matches = _image_matches(
                np.asarray(predictions[event][stem], dtype=np.float32),
                np.asarray(
                    face_boxes[event_index][0][image_index][0], dtype=np.float32
                ),
                threshold_count,
            )
            for position, difficulty in enumerate(difficulties):
                keep = np.asarray(
                    difficulty[event_index][0][image_index][0], dtype=np.int64
                )
                face_counts[position] += len(keep)
                curves[position] += _image_counts(matches, keep, threshold_count)
    results = []
    for curve, face_count in zip(curves, face_counts, strict=True):
        precision = np.divide(
            curve[:, 1],
            curve[:, 0],
            out=np.zeros(threshold_count),
            where=curve[:, 0] > 0,
        )
        recall = curve[:, 1] / max(face_count, 1)
        results.append(voc_ap(recall, precision))
    return WiderFaceResult(*results)


def standard_predictions(
    data_path: Path, artifact: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    """Attach generic image-ID records to WIDER FACE's benchmark sample identity."""
    if artifact.get("task") != "face_detection":
        raise ValueError("WIDER FACE accepts face_detection prediction artifacts only")
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
    expected_ids = list(range(len(samples)))
    if image_ids != expected_ids:
        raise ValueError(
            "Prediction artifact image_ids do not match WIDER FACE manifest order"
        )
    rows_by_image = [[] for _sample in samples]
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction artifact records must be objects")
        image_id, bbox, score = (
            record.get("image_id"),
            record.get("bbox"),
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
            or len(bbox) != 4
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in bbox
            )
        ):
            raise TypeError("Face-detection prediction bbox must contain four numbers")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TypeError("Face-detection prediction score must be a number")
        rows_by_image[image_id].append([*bbox, score])

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for image_id, sample in enumerate(samples):
        predictions.setdefault(sample.event, {})[sample.stem] = np.asarray(
            rows_by_image[image_id], dtype=np.float32
        ).reshape(-1, 5)
    return predictions


def score_artifact(data_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Score a JSON-wire face-detection artifact with WIDER FACE's metric."""
    predictions = standard_predictions(Path(data_path), artifact)
    result = evaluate_predictions(Path(data_path), predictions)
    image_ids = artifact["image_ids"]
    return {
        "primary": {"hard_ap": round(result.hard_ap, 5)},
        "secondary": {
            "medium_ap": round(result.medium_ap, 5),
            "easy_ap": round(result.easy_ap, 5),
        },
        "sample_count": len(image_ids),
    }
