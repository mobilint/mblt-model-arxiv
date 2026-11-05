"""Model-owned OBB tensor decoding and rotated suppression.

This module turns a Model runtime's raw tensors into label-indexed rotated boxes.
It deliberately has no dataset imports or benchmark terminology: a benchmark maps
the normalized label index and scores the resulting artifact at the root seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError


def _topk(values: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` largest values, ties in ascending index order."""
    return np.argsort(-values, kind="stable")[:k]


def end2end_topk(
    scores: np.ndarray, max_det: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select candidates the way Ultralytics' ``Detect.get_topk_index`` does.

    This follows the class-aware, ungrouped path that validation takes. It first keeps
    the ``max_det`` anchors with the highest best-class score, then keeps the ``max_det``
    highest (anchor, class) scores among them. One anchor may therefore appear with
    several classes.

    Args:
        scores: ``(anchors, classes)`` class probabilities.
        max_det: Maximum number of selected rows.

    Returns:
        Anchor indices, class indices and scores of the selected rows, by descending score.
        The caller applies the confidence threshold, as the Ultralytics validator does.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise PackageError(
            f"end2end_topk expects (anchors, classes) scores, got {scores.shape}"
        )
    anchors, classes = scores.shape
    k = min(max_det, anchors)
    if k == 0 or classes == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), np.empty(0, dtype=scores.dtype)
    top_anchors = _topk(scores.max(axis=1), k)
    flat = scores[top_anchors].reshape(-1)
    top = _topk(flat, k)
    return top_anchors[top // classes], (top % classes).astype(np.int64), flat[top]


Geometry = tuple[float, tuple[float, float]]
DEFAULT_CLASS_COUNT = 15
_NMS_BLOCK = 512


def _covariance(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = boxes[:, 2] ** 2 / 12.0
    b = boxes[:, 3] ** 2 / 12.0
    cosine, sine = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    return (
        a * cosine**2 + b * sine**2,
        a * sine**2 + b * cosine**2,
        (a - b) * cosine * sine,
    )


def probiou(
    boxes: np.ndarray, other: np.ndarray | None = None, eps: float = 1e-7
) -> np.ndarray:
    """Return pairwise probabilistic IoU for rotated ``xywhr`` boxes."""
    if other is None:
        other = boxes
    x1, y1 = boxes[:, 0][:, None], boxes[:, 1][:, None]
    a1, b1, c1 = (value[:, None] for value in _covariance(boxes))
    x2, y2 = other[:, 0][None, :], other[:, 1][None, :]
    a2, b2, c2 = (value[None, :] for value in _covariance(other))
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


def rotated_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Return Fast-NMS survivors for rotated boxes, preserving score order."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    if labels is not None:
        offset = labels.astype(boxes.dtype)[:, None] * 7680.0
        boxes = np.concatenate((boxes[:, :2] + offset, boxes[:, 2:]), axis=-1)
    order = np.argsort(-scores, kind="stable")
    ordered = boxes[order]
    suppressed = np.zeros(len(ordered), dtype=bool)
    for start in range(0, len(ordered), _NMS_BLOCK):
        stop = min(start + _NMS_BLOCK, len(ordered))
        pending = np.arange(start, stop)
        for row_start in range(0, start, _NMS_BLOCK):
            row_stop = min(row_start + _NMS_BLOCK, start)
            overlaps = probiou(ordered[row_start:row_stop], ordered[pending])
            fallen = (overlaps >= threshold).any(axis=0)
            suppressed[pending[fallen]] = True
            pending = pending[~fallen]
            if not pending.size:
                break
        if not pending.size:
            continue
        overlaps = probiou(ordered[start:stop], ordered[pending])
        rows = np.arange(start, stop)[:, None]
        suppressed[pending] |= (
            np.where(rows < pending[None, :], overlaps, 0.0) >= threshold
        ).any(axis=0)
    return order[(~suppressed).nonzero()[0]]


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-value))


def has_split_head(outputs: list[np.ndarray], channels: int) -> bool:
    """Return whether runtime outputs include an HWC split head of ``channels``."""
    return any(
        np.asarray(output).ndim == 3 and np.asarray(output).shape[-1] == channels
        for output in outputs
    )


def decode_dfl_heads(
    outputs: list[np.ndarray], reg_max: int, input_shape: tuple[int, int]
) -> np.ndarray:
    """Decode split DFL box, class, and angle heads into one candidate tensor."""
    boxes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    classes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    angles_by_shape: dict[tuple[int, int], np.ndarray] = {}
    for head in outputs:
        value = np.asarray(head, dtype=np.float32)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise PackageError(f"Unsupported DFL OBB head shape: {value.shape}")
        shape = (int(value.shape[0]), int(value.shape[1]))
        if value.shape[-1] == reg_max * 4:
            boxes_by_shape[shape] = value
        elif value.shape[-1] == 1:
            angles_by_shape[shape] = value
        else:
            classes_by_shape[shape] = value
    if boxes_by_shape.keys() != classes_by_shape.keys() or not boxes_by_shape:
        raise PackageError(
            "DFL OBB outputs must have one box and class head per scale."
        )

    decoded: list[np.ndarray] = []
    bins = np.arange(reg_max, dtype=np.float32)
    for (height, width), box_logits in sorted(boxes_by_shape.items(), reverse=True):
        class_logits = classes_by_shape[(height, width)]
        angle_logits = angles_by_shape.get((height, width))
        if angle_logits is None:
            raise PackageError("DFL OBB outputs must have one angle head per scale.")
        logits = box_logits.reshape(height, width, 4, reg_max)
        logits = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        distances = (
            probabilities / probabilities.sum(axis=-1, keepdims=True) * bins
        ).sum(axis=-1)
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        stride_x, stride_y = input_shape[1] / width, input_shape[0] / height
        anchor_x, anchor_y = (grid_x + 0.5) * stride_x, (grid_y + 0.5) * stride_y
        left, top, right, bottom = np.moveaxis(distances, -1, 0)
        angle = (sigmoid(angle_logits[..., 0]) - 0.25) * np.pi
        half_x = (right - left) * stride_x / 2
        half_y = (bottom - top) * stride_y / 2
        cosine, sine = np.cos(angle), np.sin(angle)
        boxes = np.stack(
            (
                anchor_x + half_x * cosine - half_y * sine,
                anchor_y + half_x * sine + half_y * cosine,
                (left + right) * stride_x,
                (top + bottom) * stride_y,
            ),
            axis=-1,
        )
        decoded.append(
            np.concatenate(
                (boxes, sigmoid(class_logits), angle[..., None]), axis=-1
            ).reshape(-1, 5 + class_logits.shape[-1])
        )
    return np.concatenate(decoded, axis=0)[None]


def decode_dflfree_heads(
    outputs: list[np.ndarray], input_shape: tuple[int, int]
) -> np.ndarray:
    """Decode direct-distance OBB box, class, and angle heads."""
    boxes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    classes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    angles_by_shape: dict[tuple[int, int], np.ndarray] = {}
    for head in outputs:
        value = np.asarray(head, dtype=np.float32)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise PackageError(
                f"Unsupported direct-distance OBB head shape: {value.shape}"
            )
        shape = (int(value.shape[0]), int(value.shape[1]))
        if value.shape[-1] == 4:
            boxes_by_shape[shape] = value
        elif value.shape[-1] == 1:
            angles_by_shape[shape] = value
        else:
            classes_by_shape[shape] = value
    if boxes_by_shape.keys() != classes_by_shape.keys() or not boxes_by_shape:
        raise PackageError(
            "Direct-distance OBB outputs need one box and class head per scale."
        )

    decoded: list[np.ndarray] = []
    for (height, width), distances in sorted(boxes_by_shape.items(), reverse=True):
        class_logits = classes_by_shape[(height, width)]
        angle = angles_by_shape.get((height, width))
        if angle is None:
            raise PackageError(
                "Direct-distance OBB outputs need one angle head per scale."
            )
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        stride_x, stride_y = input_shape[1] / width, input_shape[0] / height
        left, top, right, bottom = np.moveaxis(distances, -1, 0)
        angle_value = angle[..., 0]
        offset_x, offset_y = (right - left) / 2, (bottom - top) / 2
        center_x = (
            grid_x
            + 0.5
            + offset_x * np.cos(angle_value)
            - offset_y * np.sin(angle_value)
        ) * stride_x
        center_y = (
            grid_y
            + 0.5
            + offset_x * np.sin(angle_value)
            + offset_y * np.cos(angle_value)
        ) * stride_y
        boxes = np.stack(
            (
                center_x,
                center_y,
                (left + right) * stride_x,
                (top + bottom) * stride_y,
            ),
            axis=-1,
        )
        decoded.append(
            np.concatenate((boxes, sigmoid(class_logits), angle), axis=-1).reshape(
                -1, 5 + class_logits.shape[-1]
            )
        )
    return np.concatenate(decoded, axis=0)[None]


def detections(
    raw: np.ndarray,
    threshold: float,
    iou: float,
    geometry: Geometry | None,
    class_count: int = DEFAULT_CLASS_COUNT,
    *,
    one_to_one: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Threshold, select, and unletterbox a Model's decoded OBB candidates.

    Rows an end-to-end head already selected, and candidates decoded from its raw
    one-to-one heads (``one_to_one``, YOLO26-obb), follow Ultralytics' NMS-free path:
    top-k and a confidence threshold, never rotated IoU suppression. Every other output
    goes through Fast-NMS.
    """
    data = np.asarray(raw)[0]
    extra = 1
    widths = (6 + extra, 4 + class_count + extra)
    if (
        data.ndim == 2
        and data.shape[-1] not in widths
        and (data.shape[0] in widths or data.shape[0] < data.shape[1])
    ):
        data = data.T
    if data.shape[-1] == 6 + extra:
        # YOLO26-obb's exported end-to-end head: (max_det, 4 + 1 + 1 + 1) rows the head
        # already selected. Ultralytics only thresholds them.
        selected = data[:, 4] > threshold
        centers, scores, labels, extras = (
            data[selected, :4],
            data[selected, 4],
            data[selected, 5].astype(int),
            data[selected, 6:],
        )
        order = np.argsort(-scores, kind="stable")[:300]
        centers, scores, labels, extras = (
            centers[order],
            scores[order],
            labels[order],
            extras[order],
        )
    else:
        start = 4
        end = data.shape[1] - extra if data.shape[1] > start + extra else data.shape[1]
        classes = data[:, start:end]
        all_extras = (
            data[:, end:]
            if data.shape[1] > end
            else np.empty((len(data), 0), dtype=data.dtype)
        )
        if one_to_one:
            box_index, labels, scores = end2end_topk(classes, max_det=300)
            selected = scores > threshold
            box_index = box_index[selected]
            centers, scores, labels, extras = (
                data[box_index, :4],
                scores[selected],
                labels[selected],
                all_extras[box_index],
            )
        else:
            box_index, label_index = np.nonzero(classes > threshold)
            centers, scores, labels, extras = (
                data[box_index, :4],
                classes[box_index, label_index],
                label_index,
                all_extras[box_index],
            )
            # The stable sort is Ultralytics' order: on CUDA, where its validation runs,
            # the argsort it uses above 30000 candidates and inside Fast-NMS is stable.
            order = np.argsort(-scores, kind="stable")[:30000]
            centers, scores, labels, extras = (
                centers[order],
                scores[order],
                labels[order],
                extras[order],
            )
            angles = (
                extras[:, 0] if extras.shape[1] else np.zeros(len(scores), data.dtype)
            )
            keep = rotated_nms(np.column_stack((centers, angles)), scores, iou, labels)[
                :300
            ]
            centers, scores, labels, extras = (
                centers[keep],
                scores[keep],
                labels[keep],
                extras[keep],
            )
    boxes = np.concatenate(
        (centers[:, :2] - centers[:, 2:] / 2, centers[:, :2] + centers[:, 2:] / 2),
        axis=-1,
    )
    if geometry:
        ratio, (left, top) = geometry
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / ratio
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / ratio
    return boxes, scores, labels, extras


def decode(
    outputs: list[np.ndarray],
    threshold: float,
    iou: float,
    geometry: Geometry | None,
    input_shape: tuple[int, int],
    *,
    reg_max: int = 16,
    dflfree: bool = False,
    class_count: int = DEFAULT_CLASS_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one OBB runtime output into xyxy boxes, scores, labels, and angles."""
    raw = outputs[0]
    # ``dflfree`` marks YOLO26-obb, which is end-to-end: its split heads and its
    # decode-enabled tensor both come from the one-to-one heads.
    one_to_one = dflfree
    if dflfree and has_split_head(outputs, 4):
        raw = decode_dflfree_heads(outputs, input_shape)
    elif not dflfree and has_split_head(outputs, reg_max * 4):
        raw = decode_dfl_heads(outputs, reg_max, input_shape)
    return detections(raw, threshold, iou, geometry, class_count, one_to_one=one_to_one)


def _records(
    image_id: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    angles: np.ndarray,
) -> list[dict[str, Any]]:
    """Serialize Model-native boxes without assigning them benchmark semantics."""
    xywhr = np.column_stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            angles[:, 0] if angles.shape[1] else np.zeros(len(boxes)),
        )
    )
    return [
        {
            "image_id": image_id,
            "bbox": [float(value) for value in box],
            "label": int(label),
            "score": float(score),
        }
        for box, score, label in zip(xywhr, scores, labels, strict=True)
    ]


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    pass


def prediction_records(outputs, customer_contract, geometry, args, image_id):
    params = customer_contract["inference"]["params"]
    boxes, scores, labels, angles = decode(
        outputs,
        params["conf_thres"],
        params["iou_thres"],
        (geometry["ratio"], tuple(geometry["pad"])),
        tuple(geometry["input_size"]),
        reg_max=params["reg_max"],
        class_count=params["class_count"],
        dflfree=params.get("dflfree", False),
    )
    return _records(image_id, boxes, scores, labels, angles)


def summarize(
    outputs, customer_contract, geometry, args, output_dir: Path, source: Path
):
    return {
        "task": customer_contract["task"],
        "detections": prediction_records(outputs, customer_contract, geometry, args, 0),
    }
