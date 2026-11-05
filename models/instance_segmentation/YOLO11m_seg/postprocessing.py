"""Default instance-segmentation tensor decoding without presentation drawing.

The evaluation adapter calls this module to obtain boxes, labels, and mask
coefficients. It never selects benchmark samples or computes a benchmark score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
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


def is_end2end_model(decode_cfg: dict) -> bool:
    """Return whether source.yaml's ``decode_cfg`` describes an end-to-end model.

    ``nmsfree`` marks YOLOv10, and ``dflfree`` marks YOLO26, the only DFL-free family.
    Both are end-to-end. Their decoded candidate tensor, and YOLOv10's DFL heads, have the
    same shape as YOLOv8's, so the decoders cannot tell them apart without this flag.
    """
    return bool(decode_cfg.get("nmsfree", False) or decode_cfg.get("dflfree", False))


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray | None = None,
    max_det: int = 300,
) -> np.ndarray:
    """Greedy IoU suppression. ``labels`` offsets each box into a disjoint
    coordinate region per class (Ultralytics-style class-aware NMS) so
    candidates from different classes never suppress one another.

    Candidates are materialised once in descending-score order and the
    surviving set is compacted after each suppression pass, so every pass
    reads contiguous memory and the box areas are computed once instead of
    over the whole candidate array on every iteration -- the same hoist
    mblt-vision's ``non_max_suppression`` applies. The kept indices are
    identical to re-indexing the original arrays per iteration.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    if labels is not None:
        boxes = boxes + labels.astype(boxes.dtype)[:, None] * 7680.0
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = np.ascontiguousarray(boxes[order, :4].T)
    area = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        keep.append(int(order[0]))
        head_x1, head_y1, head_x2, head_y2, head_area = (
            x1[0],
            y1[0],
            x2[0],
            y2[0],
            area[0],
        )
        order, x1, y1, x2, y2, area = (
            order[1:],
            x1[1:],
            y1[1:],
            x2[1:],
            y2[1:],
            area[1:],
        )
        if not order.size:
            break
        overlap = np.maximum(0, np.minimum(head_x2, x2) - np.maximum(head_x1, x1)) * (
            np.maximum(0, np.minimum(head_y2, y2) - np.maximum(head_y1, y1))
        )
        # torchvision.ops.nms, which Ultralytics calls, divides without an epsilon and
        # suppresses only an IoU above the threshold. Two zero-area boxes give 0 / 0 = NaN,
        # which compares False and so keeps the candidate.
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap_ratio = overlap / (head_area + area - overlap)
        selected = ~(overlap_ratio > threshold)
        order, x1, y1, x2, y2, area = (
            order[selected],
            x1[selected],
            y1[selected],
            x2[selected],
            y2[selected],
            area[selected],
        )
    return np.asarray(keep, dtype=np.int64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-value))


def _raw_anchor_heads(outputs: list[np.ndarray]) -> list[np.ndarray]:
    """Return raw anchor heads while excluding decoded and auxiliary outputs.

    Mirrors models/<Model>/common/local_inference.py::_raw_anchor_heads: heads are
    ``(1, 3, H, W, no)`` or ``(H, W, 3*no)``, sorted fine to coarse so they line up with
    source.yaml's ``decode_cfg.anchors`` rather than with the runtime's output sequence.
    """
    heads: list[np.ndarray] = []
    for output in outputs:
        value = np.asarray(output)
        if (
            value.ndim == 5
            and value.shape[0] == 1
            and value.shape[1] == 3
            or value.ndim == 3
            and value.shape[-1] % 3 == 0
            and value.shape[-1] // 3 >= 6
        ):
            heads.append(value)
    return sorted(
        heads, key=lambda value: int(np.prod(value.shape[-3:-1])), reverse=True
    )


def _has_split_head(outputs: list[np.ndarray], channels: int) -> bool:
    """Return whether outputs contain an HWC split head with ``channels``."""
    return any(
        np.asarray(output).ndim == 3 and np.asarray(output).shape[-1] == channels
        for output in outputs
    )


def _decode_anchor_heads(
    outputs: list[np.ndarray], anchors: list[list[int]], input_shape: tuple[int, int]
) -> np.ndarray:
    """Decode raw YOLO anchor heads into the concatenated ``(1, anchors, no)`` tensor.

    For instance segmentation (no > 85), mask coefficients are linear mask-projection weights,
    not probabilities. They are NOT sigmoided; instead they are multiplied by the objectness
    score before being used to project prototype masks. This mirrors
    models/<M>/common/local_inference.py::_decode_anchor_heads for instance segmentation.
    """
    decoded: list[np.ndarray] = []
    for head, anchor_values in zip(outputs, anchors, strict=True):
        value = np.asarray(head, dtype=np.float32)
        if value.ndim == 5 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 4 and value.shape[0] == 3:
            raw = value
            _, height, width, no = raw.shape
        elif value.ndim == 3 and value.shape[-1] % 3 == 0:
            height, width, channels = value.shape
            no = channels // 3
            raw = value.reshape(height, width, 3, no).transpose(2, 0, 1, 3)
        else:
            raise PackageError(f"Unsupported raw YOLO anchor head shape: {value.shape}")
        if no < 6 or len(anchor_values) != 6:
            raise PackageError(
                "YOLO anchor heads require three [width, height] anchors."
            )
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((grid_x, grid_y), axis=-1)[None]
        anchor_grid = np.asarray(anchor_values, dtype=np.float32).reshape(3, 1, 1, 2)
        stride = np.asarray(
            (input_shape[1] / width, input_shape[0] / height), dtype=np.float32
        )
        head_output = _sigmoid(raw)
        # Segmentation coefficients are linear mask-projection weights, not
        # probabilities. The upstream decoder preserves them and applies the
        # objectness score before multiplying them with prototype masks.
        if no > 85:
            head_output[..., 85:] = raw[..., 85:] * head_output[..., 4:5]
        head_output[..., :2] = (head_output[..., :2] * 2.0 - 0.5 + grid) * stride
        head_output[..., 2:4] = (head_output[..., 2:4] * 2.0) ** 2 * anchor_grid
        decoded.append(head_output.reshape(-1, no))
    return np.concatenate(decoded, axis=0)[None]


def _check_extra_channels(n_extra: int, box_channels: int, n_classes: int = 80) -> None:
    """Refuse a mask-coefficient count that a split head's width cannot identify.

    The split-head decoders tell box, class and mask-coefficient heads apart by channel
    count alone, because runtime output order is not a reliable role. A coefficient count
    equal to the box or class width would file the mask head as one of those and then
    fail, or worse, decode it, so it is rejected up front. No Ultralytics segmentation
    head collides: ``Segment``'s ``nm`` is 32.
    """
    if n_extra in (box_channels, n_classes):
        raise PackageError(
            f"decode_cfg.n_extra={n_extra} equals the {box_channels}-channel box or "
            f"{n_classes}-channel class head width, so split heads cannot be told apart."
        )


def _decode_dflfree_heads(
    outputs: list[np.ndarray],
    input_shape: tuple[int, int],
    n_extra: int = 0,
) -> np.ndarray:
    """Decode DFL-free YOLO split heads with direct distance outputs.

    Mirrors models/<Model>/common/local_inference.py::_decode_segmentation_heads.
    Split box heads emit (H, W, 4) distance tensors, class heads (H, W, 80),
    and optional extra heads (H, W, n_extra) for mask coefficients.
    """
    if n_extra:
        _check_extra_channels(n_extra, 4)
    boxes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    classes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    extras_by_shape: dict[tuple[int, int], np.ndarray] = {}
    for head in outputs:
        value = np.asarray(head, dtype=np.float32)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise PackageError(f"Unsupported DFL-free YOLO head shape: {value.shape}")
        shape = (int(value.shape[0]), int(value.shape[1]))
        if value.shape[-1] == 4:
            boxes_by_shape[shape] = value
        elif value.shape[-1] == 80:
            classes_by_shape[shape] = value
        elif n_extra and value.shape[-1] == n_extra:
            extras_by_shape[shape] = value
    if boxes_by_shape.keys() != classes_by_shape.keys() or not boxes_by_shape:
        raise PackageError(
            "DFL-free YOLO outputs must have one box and class head per scale."
        )

    decoded: list[np.ndarray] = []
    for (height, width), distances in sorted(boxes_by_shape.items(), reverse=True):
        class_logits = classes_by_shape[(height, width)]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        stride_x, stride_y = input_shape[1] / width, input_shape[0] / height
        anchor_x, anchor_y = (grid_x + 0.5) * stride_x, (grid_y + 0.5) * stride_y
        left, top, right, bottom = np.moveaxis(distances, -1, 0)
        x1, y1 = anchor_x - left * stride_x, anchor_y - top * stride_y
        x2, y2 = anchor_x + right * stride_x, anchor_y + bottom * stride_y
        boxes = np.stack(
            ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
            axis=-1,
        )
        parts = [boxes, _sigmoid(class_logits)]
        if n_extra:
            extra = extras_by_shape.get((height, width))
            if extra is None:
                raise PackageError(
                    "DFL-free segmentation outputs must have one mask-coefficient head per scale."
                )
            parts.append(extra)
        decoded.append(
            np.concatenate(parts, axis=-1).reshape(
                -1, 4 + class_logits.shape[-1] + n_extra
            )
        )
    return np.concatenate(decoded, axis=0)[None]


def _decode_dfl_heads(
    outputs: list[np.ndarray],
    reg_max: int,
    input_shape: tuple[int, int],
    n_extra: int = 0,
) -> np.ndarray:
    """Decode split DFL box and class heads emitted by anchorless YOLO MXQs."""
    if n_extra:
        _check_extra_channels(n_extra, reg_max * 4)
    boxes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    classes_by_shape: dict[tuple[int, int], np.ndarray] = {}
    extras_by_shape: dict[tuple[int, int], np.ndarray] = {}
    for head in outputs:
        value = np.asarray(head, dtype=np.float32)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise PackageError(f"Unsupported DFL YOLO head shape: {value.shape}")
        shape = (int(value.shape[0]), int(value.shape[1]))
        if value.shape[-1] == reg_max * 4:
            boxes_by_shape[shape] = value
        elif value.shape[-1] == 80:
            classes_by_shape[shape] = value
        elif n_extra and value.shape[-1] == n_extra:
            extras_by_shape[shape] = value
    if boxes_by_shape.keys() != classes_by_shape.keys() or not boxes_by_shape:
        raise PackageError(
            "DFL YOLO outputs must have one box and class head per scale."
        )

    decoded: list[np.ndarray] = []
    bins = np.arange(reg_max, dtype=np.float32)
    for (height, width), box_logits in sorted(boxes_by_shape.items(), reverse=True):
        class_logits = classes_by_shape[(height, width)]
        # ``box_logits`` is a view of the runtime output buffer, so the
        # max-subtraction that stabilises the softmax must not be in place:
        # decoding a head may not mutate the tensor its caller still holds.
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
        x1, y1 = anchor_x - left * stride_x, anchor_y - top * stride_y
        x2, y2 = anchor_x + right * stride_x, anchor_y + bottom * stride_y
        boxes = np.stack(
            ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
            axis=-1,
        )
        parts = [boxes, _sigmoid(class_logits)]
        if n_extra:
            extra = extras_by_shape.get((height, width))
            if extra is None:
                raise PackageError(
                    "DFL segmentation outputs must have one mask-coefficient head per scale."
                )
            parts.append(extra)
        decoded.append(
            np.concatenate(parts, axis=-1).reshape(
                -1, 4 + class_logits.shape[-1] + n_extra
            )
        )
    return np.concatenate(decoded, axis=0)[None]


def _detections(
    raw: np.ndarray,
    threshold: float,
    iou: float,
    geometry: tuple[float, tuple[float, float]] | None,
    *,
    end2end: bool = False,
    n_extra: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one image's detections: top-k for end-to-end heads, NMS otherwise.

    ``n_extra`` is the mask-coefficient count (source.yaml's ``decode_cfg.n_extra``);
    every shipped Ultralytics segmentation head has 32, ``Segment``'s default ``nm``.
    """
    data = np.asarray(raw)[0]
    # A decode-included artifact returns one final row per detection, so a
    # sparse image can produce fewer rows than the 38 channels beside them.
    # Dispatching that on "transpose whichever axis is shorter" reads ``(2, 38)``
    # as 38 detections of two channels; the known final-row width settles it
    # first, the way mblt-vision settles the same ambiguity from its own
    # expected channel count. Candidate tensors are far wider than 38 and are
    # unaffected.
    final_width = 6 + n_extra
    if (
        data.ndim == 2
        and data.shape[-1] != final_width
        and (data.shape[0] == final_width or data.shape[0] < data.shape[1])
    ):
        data = data.T
    if data.shape[-1] == final_width:
        # YOLO26-seg's exported end-to-end head: (max_det, 4 + 1 + 1 + n_extra) rows the head
        # already selected. Ultralytics only thresholds them; suppressing them again
        # would drop the second of two overlapping objects the head kept.
        boxes, scores, labels, coefficients = (
            data[:, :4],
            data[:, 4],
            data[:, 5].astype(int),
            data[:, 6:],
        )
        selected = scores > threshold
        boxes, scores, labels, coefficients = (
            boxes[selected],
            scores[selected],
            labels[selected],
            coefficients[selected],
        )
        order = np.argsort(-scores, kind="stable")[:300]
        boxes, scores, labels, coefficients = (
            boxes[order],
            scores[order],
            labels[order],
            coefficients[order],
        )
    else:
        all_boxes = data[:, :4].copy()
        all_boxes[:, :2] -= all_boxes[:, 2:] / 2
        all_boxes[:, 2:] += all_boxes[:, :2]
        all_coefficients = data[:, -n_extra:]
        # An anchor-based head carries objectness between the box and the classes, and the
        # reference folds it into the class score before NMS. Slicing from column 4 instead
        # reads objectness as class 0 and shifts every label by one, which does not raise
        # here -- it raises later in the scorer's category_ids lookup, or silently mis-scores.
        n_classes = data.shape[-1] - n_extra - 4
        if n_classes == 81:
            classes = data[:, 5:-n_extra] * data[:, 4:5]
        else:
            classes = data[:, 4:-n_extra]
        if end2end:
            # Raw one-to-one heads of an end-to-end model (YOLO26-seg), decoded here.
            # Ultralytics selects their rows by top-k without IoU suppression.
            box_index, labels, scores = end2end_topk(classes, max_det=300)
            selected = scores > threshold
            box_index = box_index[selected]
            boxes, scores, labels, coefficients = (
                all_boxes[box_index],
                scores[selected],
                labels[selected],
                all_coefficients[box_index],
            )
        else:
            # Ultralytics-style multi-label expansion: emit one candidate row per
            # class whose score clears the threshold, instead of collapsing each
            # anchor to a single argmax label before NMS.
            box_index, label_index = np.nonzero(classes > threshold)
            boxes, scores, labels, coefficients = (
                all_boxes[box_index],
                classes[box_index, label_index],
                label_index,
                all_coefficients[box_index],
            )
            # At most 30000 candidates enter NMS and at most 300 detections leave it,
            # the caps Ultralytics applies around its suppression (max_nms and max_det).
            # Without the candidate cap, multi-label expansion feeds every
            # above-threshold class of every anchor into an O(n^2) comparison: a dense
            # 640x640 output diverges from the reference, which truncates first, and
            # gets slow doing it. The stable sort is Ultralytics' order too:
            # torchvision.ops.nms sorts stably, and the argsort Ultralytics runs above
            # 30000 candidates is stable on CUDA.
            order = np.argsort(-scores, kind="stable")[:30000]
            boxes, scores, labels, coefficients = (
                boxes[order],
                scores[order],
                labels[order],
                coefficients[order],
            )
            keep = _nms(boxes, scores, iou, labels, max_det=300)[:300]
            boxes, scores, labels, coefficients = (
                boxes[keep],
                scores[keep],
                labels[keep],
                coefficients[keep],
            )
    if geometry:
        ratio, (left, top) = geometry
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / ratio
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / ratio
    return boxes, scores, labels, coefficients


def _prototype_output(
    outputs: list[np.ndarray], n_extra: int = 32
) -> np.ndarray | None:
    """Return the largest ``n_extra``-channel prototype map, as HWC, or None."""
    candidates: list[np.ndarray] = []
    for output in outputs:
        value = np.asarray(output, dtype=np.float32)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 3 and value.shape[-1] == n_extra:
            candidates.append(value)
        elif value.ndim == 3 and value.shape[0] == n_extra:
            candidates.append(value.transpose(1, 2, 0))
    return max(candidates, key=lambda value: value.size) if candidates else None


def _draw_detection(
    image: np.ndarray,
    outputs: list[np.ndarray],
    threshold: float,
    iou: float,
    geometry: tuple[float, tuple[float, float]] | None,
    *,
    reg_max: int,
    input_shape: tuple[int, int],
    anchors: list[list[int]] | None = None,
    nmsfree: bool = False,
    n_extra: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's outputs into boxes, scores, labels, and coefficients.

    An end-to-end model's tensors come from its one-to-one heads, so their rows are
    selected by top-k rather than IoU NMS. DFL-free split heads exist only on YOLO26-seg
    and say so by shape; its decode-enabled tensor looks like YOLOv8-seg's, so ``nmsfree``
    (``is_end2end_model(decode_cfg)``) names the model.
    """
    raw = outputs[0]
    end2end = nmsfree
    if anchors:
        # Anchor-based model: per-scale anchor heads
        heads = _raw_anchor_heads(outputs)
        if len(heads) == len(anchors):
            raw = _decode_anchor_heads(heads, anchors, input_shape)
    elif _has_split_head(outputs, 4) and _has_split_head(outputs, 80):
        # DFL-free model: split box (4 channels) and class (80 channels) heads
        raw = _decode_dflfree_heads(outputs, input_shape, n_extra=n_extra)
        end2end = True
    elif _has_split_head(outputs, reg_max * 4):
        # DFL model: split box (reg_max*4 channels) and class (80 channels) heads
        raw = _decode_dfl_heads(outputs, reg_max, input_shape, n_extra=n_extra)
    return _detections(raw, threshold, iou, geometry, end2end=end2end, n_extra=n_extra)


def _uncompressed_rle(mask: np.ndarray) -> dict[str, list[int]]:
    """Encode a binary mask as JSON-safe column-major run lengths.

    Runs alternate starting with background, so a mask whose first pixel is set starts with a
    zero-length run. The run boundaries are found with array operations: a Python loop over every
    pixel cost ~0.35 s per mask, and a COCO evaluate encodes up to 300 masks for each of 5000
    images.
    """
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F") != 0
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], edges, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


_ROI_PIXEL_COST = 80
_ROI_DETECTION_COST = 320_000


def _mask_taps(
    index: np.ndarray, target_length: int, source_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two source taps and weight used by linear image resizing."""
    scale = source_length / target_length
    coordinate = ((index + 0.5) * scale - 0.5).astype(np.float32)
    low = np.floor(coordinate)
    weight = (coordinate - low).astype(np.float32)
    low = low.astype(np.intp)
    return (
        np.clip(low, 0, source_length - 1),
        np.clip(low + 1, 0, source_length - 1),
        weight,
    )


def _resample_window(
    source: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    row_origin: int,
    column_origin: int,
) -> np.ndarray:
    """Bilinearly resample a window without rasterising an entire frame."""
    row_low, row_high, row_weight = _mask_taps(rows, target_shape[0], source_shape[0])
    column_low, column_high, column_weight = _mask_taps(
        columns, target_shape[1], source_shape[1]
    )
    horizontal = (
        source[:, column_low - column_origin] * (1.0 - column_weight).astype(np.float32)
        + source[:, column_high - column_origin] * column_weight
    )
    return (
        horizontal[row_low - row_origin]
        * (1.0 - row_weight).astype(np.float32)[:, None]
        + horizontal[row_high - row_origin] * row_weight[:, None]
    )


def _clipped_boxes(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    return np.clip(
        np.asarray(boxes).astype(int),
        [0, 0, 0, 0],
        [width, height, width, height],
    )


def _roi_binary_masks(
    prototype: np.ndarray,
    coefficients: np.ndarray,
    clipped: np.ndarray,
    geometry,
    image_shape: tuple[int, int],
    input_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Rasterise each mask only over its box while matching full-frame resize."""
    height, width = image_shape
    ratio, (left, top) = geometry
    crop_top, crop_left = round(top), round(left)
    crop_height = round(top + height * ratio) - crop_top
    crop_width = round(left + width * ratio) - crop_left
    proto_height, proto_width, channels = prototype.shape
    result: list[np.ndarray] = []
    for coefficient, (x1, y1, x2, y2) in zip(coefficients, clipped, strict=True):
        mask = np.zeros((height, width), dtype=np.uint8, order="F")
        result.append(mask)
        if x1 >= x2 or y1 >= y2 or crop_height <= 0 or crop_width <= 0:
            continue
        rows, columns = np.arange(y1, y2), np.arange(x1, x2)
        crop_rows_low, crop_rows_high, _ = _mask_taps(rows, height, crop_height)
        crop_columns_low, crop_columns_high, _ = _mask_taps(columns, width, crop_width)
        input_rows = np.arange(
            crop_top + crop_rows_low[0], crop_top + crop_rows_high[-1] + 1
        )
        input_columns = np.arange(
            crop_left + crop_columns_low[0], crop_left + crop_columns_high[-1] + 1
        )
        proto_rows_low, proto_rows_high, _ = _mask_taps(
            input_rows, input_shape[0], proto_height
        )
        proto_columns_low, proto_columns_high, _ = _mask_taps(
            input_columns, input_shape[1], proto_width
        )
        row_origin, column_origin = proto_rows_low[0], proto_columns_low[0]
        window = prototype[
            row_origin : proto_rows_high[-1] + 1,
            column_origin : proto_columns_high[-1] + 1,
        ]
        low_resolution = _sigmoid(window.reshape(-1, channels) @ coefficient).reshape(
            window.shape[0], window.shape[1]
        )
        on_input = _resample_window(
            low_resolution,
            input_rows,
            input_columns,
            (proto_height, proto_width),
            input_shape,
            row_origin,
            column_origin,
        )
        on_image = _resample_window(
            on_input,
            rows,
            columns,
            (crop_height, crop_width),
            (height, width),
            crop_rows_low[0],
            crop_columns_low[0],
        )
        mask[y1:y2, x1:x2] = on_image >= 0.5
    return result


def _binary_masks(
    outputs: list[np.ndarray],
    coefficients: np.ndarray,
    boxes: np.ndarray,
    geometry,
    image_shape: tuple[int, int],
    input_shape: tuple[int, int],
    n_extra: int = 32,
) -> list[np.ndarray]:
    """Project coefficients into masks, choosing a whole-frame or ROI path."""
    prototype = _prototype_output(outputs, n_extra)
    height, width = image_shape
    if prototype is None or len(coefficients) == 0:
        return [np.zeros(image_shape, dtype=np.uint8, order="F") for _ in boxes]
    clipped = _clipped_boxes(boxes, image_shape)
    box_pixels = int(
        (
            np.maximum(clipped[:, 2] - clipped[:, 0], 0)
            * np.maximum(clipped[:, 3] - clipped[:, 1], 0)
        ).sum()
    )
    frame_pixels = len(clipped) * (input_shape[0] * input_shape[1] + height * width)
    if _ROI_PIXEL_COST * box_pixels + _ROI_DETECTION_COST * len(clipped) < frame_pixels:
        return _roi_binary_masks(
            prototype, coefficients, clipped, geometry, image_shape, input_shape
        )
    masks = _sigmoid(
        coefficients @ prototype.reshape(-1, prototype.shape[-1]).T
    ).reshape(len(coefficients), *prototype.shape[:2])
    ratio, (left, top) = geometry
    right, bottom = round(left + width * ratio), round(top + height * ratio)
    result: list[np.ndarray] = []
    for mask, (x1, y1, x2, y2) in zip(masks, clipped, strict=True):
        output = np.zeros((height, width), dtype=np.uint8, order="F")
        result.append(output)
        if x1 >= x2 or y1 >= y2:
            continue
        mask = cv2.resize(
            mask, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask[round(top) : bottom, round(left) : right],
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        output[y1:y2, x1:x2] = mask[y1:y2, x1:x2] >= 0.5
    return result


def _records(
    image_id: int,
    image_shape: tuple[int, int],
    input_shape: tuple[int, int],
    outputs: list[np.ndarray],
    geometry,
    decode_cfg: dict,
    threshold: float,
    iou: float,
) -> list[dict]:
    # Mask coefficients per row; Ultralytics' Segment head default nm is 32.
    n_extra = int(decode_cfg.get("n_extra", 32))
    boxes, scores, labels, coefficients = _draw_detection(
        None,
        outputs,
        threshold,
        iou,
        geometry,
        reg_max=int(decode_cfg.get("reg_max", 16)),
        input_shape=input_shape,
        anchors=decode_cfg.get("anchors"),
        nmsfree=is_end2end_model(decode_cfg),
        n_extra=n_extra,
    )
    masks = _binary_masks(
        outputs,
        coefficients,
        boxes,
        geometry,
        image_shape,
        input_shape,
        n_extra,
    )
    height, width = image_shape
    boxes = np.clip(
        np.asarray(boxes, dtype=np.float32), 0.0, [width, height, width, height]
    )
    records: list[dict] = []
    for box, score, label, mask in zip(boxes, scores, labels, masks, strict=True):
        x1, y1, x2, y2 = box.tolist()
        records.append(
            {
                "image_id": image_id,
                "label": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "mask": _uncompressed_rle(mask),
            }
        )
    return records


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    pass


def prediction_records(outputs, customer_contract, geometry, args, image_id):
    params = customer_contract["inference"]["params"]
    return _records(
        image_id,
        tuple(geometry["original_size"]),
        tuple(geometry["input_size"]),
        outputs,
        (geometry["ratio"], tuple(geometry["pad"])),
        params,
        params["conf_thres"],
        params["iou_thres"],
    )


def summarize(
    outputs, customer_contract, geometry, args, output_dir: Path, source: Path
):
    return {
        "task": customer_contract["task"],
        "detections": prediction_records(outputs, customer_contract, geometry, args, 0),
    }
