"""YOLO detection outputs: box decoding and non-maximum suppression.

Decodes the output layouts an Ultralytics-family model can emit after export or NPU
compilation: one decoded (batch, anchors, 4 + classes) tensor, an end-to-end
(batch, max_det, 6) tensor, or split per-scale box and class heads (with or without
distribution focal loss bins). Applies confidence thresholding and class-aware NMS, maps
boxes back to the original image, and writes a JSON list of detections plus an optional
annotated image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError, import_optional

END2END_WIDTH = 6
MAX_CANDIDATES = 30000
CLASS_OFFSET = 7680.0


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    params = customer_contract["inference"]["params"]
    argument_parser.add_argument(
        "--conf-thres",
        type=float,
        default=float(params["conf_thres"]),
        help="confidence threshold (default: %(default)s, the benchmark value; 0.25 is a "
        "common display value)",
    )
    argument_parser.add_argument(
        "--iou-thres",
        type=float,
        default=float(params["iou_thres"]),
        help="NMS IoU threshold (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--save-image",
        action="store_true",
        help="also write an annotated copy of the input image beside the outputs",
    )


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-value))


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray | None = None,
    max_det: int = 300,
) -> np.ndarray:
    """Greedy IoU suppression, class-aware through a per-class coordinate offset."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    if labels is not None:
        boxes = boxes + labels.astype(boxes.dtype)[:, None] * CLASS_OFFSET
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = np.ascontiguousarray(boxes[order, :4].T)
    area = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        keep.append(int(order[0]))
        head = (x1[0], y1[0], x2[0], y2[0], area[0])
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
        overlap = np.maximum(0, np.minimum(head[2], x2) - np.maximum(head[0], x1)) * (
            np.maximum(0, np.minimum(head[3], y2) - np.maximum(head[1], y1))
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap_ratio = overlap / (head[4] + area - overlap)
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


def _has_split_head(outputs: list[np.ndarray], channels: int) -> bool:
    return any(
        np.asarray(o).ndim == 3 and np.asarray(o).shape[-1] == channels for o in outputs
    )


def _split_heads(
    outputs: list[np.ndarray], box_channels: int, n_classes: int
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    boxes: dict[tuple[int, int], np.ndarray] = {}
    classes: dict[tuple[int, int], np.ndarray] = {}
    for head in outputs:
        value = np.asarray(head, dtype=np.float32)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            continue
        shape = (int(value.shape[0]), int(value.shape[1]))
        if value.shape[-1] == box_channels:
            boxes[shape] = value
        elif value.shape[-1] == n_classes:
            classes[shape] = value
    if boxes.keys() != classes.keys() or not boxes:
        raise PackageError(
            "split YOLO outputs must have one box and one class head per scale"
        )
    return boxes, classes


def decode_dfl_heads(
    outputs: list[np.ndarray],
    reg_max: int,
    input_shape: tuple[int, int],
    n_classes: int,
) -> np.ndarray:
    boxes_by_shape, classes_by_shape = _split_heads(outputs, reg_max * 4, n_classes)
    decoded: list[np.ndarray] = []
    bins = np.arange(reg_max, dtype=np.float32)
    for (height, width), box_logits in sorted(boxes_by_shape.items(), reverse=True):
        class_logits = classes_by_shape[(height, width)]
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
        boxes = np.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), axis=-1)
        decoded.append(
            np.concatenate([boxes, sigmoid(class_logits)], axis=-1).reshape(
                -1, 4 + class_logits.shape[-1]
            )
        )
    return np.concatenate(decoded, axis=0)[None]


def decode_dflfree_heads(
    outputs: list[np.ndarray], input_shape: tuple[int, int], n_classes: int
) -> np.ndarray:
    boxes_by_shape, classes_by_shape = _split_heads(outputs, 4, n_classes)
    decoded: list[np.ndarray] = []
    for (height, width), distances in sorted(boxes_by_shape.items(), reverse=True):
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        stride_x, stride_y = input_shape[1] / width, input_shape[0] / height
        left, top, right, bottom = np.moveaxis(distances, -1, 0)
        x1, y1 = (grid_x + 0.5 - left) * stride_x, (grid_y + 0.5 - top) * stride_y
        x2, y2 = (grid_x + 0.5 + right) * stride_x, (grid_y + 0.5 + bottom) * stride_y
        xywh = np.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), axis=-1)
        decoded.append(
            np.concatenate(
                (xywh, sigmoid(classes_by_shape[(height, width)])), axis=-1
            ).reshape(-1, 4 + n_classes)
        )
    return np.concatenate(decoded)[None]


def _raw_anchor_heads(outputs: list[np.ndarray]) -> list[np.ndarray]:
    """Select raw anchor heads, ignoring a decode-enabled MXQ's first output.

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


def _decode_anchor_heads(
    outputs: list[np.ndarray], anchors: list[list[int]], input_shape: tuple[int, int]
) -> np.ndarray:
    """Decode raw YOLO anchor heads into the concatenated ``(1, anchors, no)`` tensor.

    A reshape alone is not a decode: the head emits logits, and the box halves are relative
    to the cell grid and the configured anchor. Skipping the sigmoid, the grid offset, the
    anchor multiply or the stride does not raise -- it publishes a plausible, wrong mAP.
    Kept behaviourally identical to models/<Model>/common/local_inference.py.
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
        head_output = sigmoid(raw)
        head_output[..., :2] = (head_output[..., :2] * 2.0 - 0.5 + grid) * stride
        head_output[..., 2:4] = (head_output[..., 2:4] * 2.0) ** 2 * anchor_grid
        decoded.append(head_output.reshape(-1, no))
    return np.concatenate(decoded, axis=0)[None]


def _map_back(boxes: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    ratio = float(geometry["ratio"])
    left, top = geometry["pad"]
    boxes = boxes.copy()
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / ratio
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / ratio
    height, width = geometry["original_size"]
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, width)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, height)
    return boxes


def detections(
    raw: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    geometry: dict[str, Any],
    *,
    n_classes: int,
    max_det: int,
    nmsfree: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(raw, dtype=np.float32)[0]
    if (
        data.ndim == 2
        and data.shape[0] < data.shape[1]
        and data.shape[-1] not in (4 + n_classes, 5 + n_classes, END2END_WIDTH)
    ):
        data = data.T
    if data.shape[-1] == END2END_WIDTH and END2END_WIDTH not in (
        4 + n_classes,
        5 + n_classes,
    ):
        boxes, scores, labels = (
            data[:, :4].copy(),
            data[:, 4],
            data[:, 5].astype(np.int64),
        )
        selected = scores > conf_thres
        boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
        order = np.argsort(-scores, kind="stable")[:max_det]
        return _map_back(boxes[order], geometry), scores[order], labels[order]
    if data.shape[-1] not in (4 + n_classes, 5 + n_classes):
        raise PackageError(
            f"unsupported detection output width {data.shape[-1]} (expected "
            f"{4 + n_classes}, {5 + n_classes} or {END2END_WIDTH} for {n_classes} classes)"
        )
    all_boxes = data[:, :4].copy()
    all_boxes[:, :2] -= all_boxes[:, 2:] / 2
    all_boxes[:, 2:] += all_boxes[:, :2]
    classes = (
        data[:, 5:] * data[:, 4:5] if data.shape[-1] == 5 + n_classes else data[:, 4:]
    )
    if nmsfree:
        k = min(max_det, len(classes))
        top_anchors = np.argsort(-classes.max(axis=1), kind="stable")[:k]
        flat = classes[top_anchors].reshape(-1)
        top = np.argsort(-flat, kind="stable")[:k]
        scores = flat[top]
        selected = scores > conf_thres
        indices = top_anchors[top // n_classes][selected]
        labels = (top % n_classes)[selected]
        return _map_back(all_boxes[indices], geometry), scores[selected], labels
    box_index, label_index = np.nonzero(classes > conf_thres)
    boxes, scores, labels = (
        all_boxes[box_index],
        classes[box_index, label_index],
        label_index,
    )
    order = np.argsort(-scores, kind="stable")[:MAX_CANDIDATES]
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    keep = nms(boxes, scores, iou_thres, labels, max_det=max_det)[:max_det]
    return _map_back(boxes[keep], geometry), scores[keep], labels[keep]


def decode(
    outputs: list[np.ndarray],
    params: dict[str, Any],
    conf_thres: float,
    iou_thres: float,
    geometry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_classes = int(params["n_classes"])
    reg_max = int(params["reg_max"])
    max_det = int(params.get("max_det", 300))
    input_shape = (int(geometry["input_size"][0]), int(geometry["input_size"][1]))
    raw = outputs[0]
    anchors = params.get("anchors")
    if _has_split_head(outputs, 4) and _has_split_head(outputs, n_classes):
        raw = decode_dflfree_heads(outputs, input_shape, n_classes)
    elif _has_split_head(outputs, reg_max * 4):
        raw = decode_dfl_heads(outputs, reg_max, input_shape, n_classes)
    elif anchors:
        heads = _raw_anchor_heads(outputs)
        if len(heads) == len(anchors):
            raw = _decode_anchor_heads(heads, anchors, input_shape)
    return detections(
        raw,
        conf_thres,
        iou_thres,
        geometry,
        n_classes=n_classes,
        max_det=max_det,
        nmsfree=bool(params.get("nmsfree", False)),
    )


def summarize(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    output_dir: Path,
    source: Path,
) -> dict[str, Any]:
    if not 0.0 <= args.conf_thres <= 1.0 or not 0.0 <= args.iou_thres <= 1.0:
        raise PackageError("--conf-thres and --iou-thres must be within [0, 1]")
    boxes, scores, labels = decode(
        outputs,
        customer_contract["inference"]["params"],
        args.conf_thres,
        args.iou_thres,
        geometry,
    )
    records = [
        {
            "class_index": int(label),
            "score": float(score),
            "bbox_xyxy": [float(v) for v in box],
        }
        for box, score, label in zip(boxes, scores, labels)
    ]
    summary: dict[str, Any] = {
        "task": customer_contract["task"],
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "detections": records,
        "detection_count": len(records),
        "note": "class_index follows the training dataset's class order; bbox_xyxy is in "
        "original image pixels",
    }
    if args.save_image:
        cv2 = import_optional("cv2", "install the packages in requirements.txt")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        for record in records:
            x1, y1, x2, y2 = (round(v) for v in record["bbox_xyxy"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{record['class_index']}:{record['score']:.2f}",
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        annotated = output_dir / f"{source.stem}.detections.png"
        cv2.imwrite(str(annotated), image)
        summary["annotated_image"] = str(annotated)
    return summary


def prediction_records(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    image_id: int,
) -> list[dict[str, Any]]:
    """Benchmark records: one `[x, y, width, height]` box in original pixels per detection."""
    boxes, scores, labels = decode(
        outputs,
        customer_contract["inference"]["params"],
        args.conf_thres,
        args.iou_thres,
        geometry,
    )
    return [
        {
            "image_id": image_id,
            "bbox": [
                float(box[0]),
                float(box[1]),
                float(box[2] - box[0]),
                float(box[3] - box[1]),
            ],
            "score": float(score),
            "label": int(label),
        }
        for box, score, label in zip(boxes, scores, labels)
    ]
