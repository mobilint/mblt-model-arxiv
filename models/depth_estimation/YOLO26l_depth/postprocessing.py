"""Depth outputs: the predicted depth map.

The model emits one relative depth map at the letterboxed input resolution. The map is
cropped to the un-padded region, resized to the original image size, saved as float32
`.npy`, and rendered as an 8-bit PNG for a quick visual check.
"""

from __future__ import annotations

import base64
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError, import_optional


def add_arguments(argument_parser: Any, customer_contract: dict[str, Any]) -> None:
    argument_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="do not write the 8-bit PNG preview of the depth map",
    )


def depth_map(output: Any, shape: tuple[int, int]) -> np.ndarray:
    cv2 = import_optional("cv2", "install the packages in requirements.txt")
    value = np.asarray(output, dtype=np.float32)
    while value.ndim > 2:
        value = (
            value[0]
            if value.shape[0] == 1
            else value[..., 0]
            if value.shape[-1] == 1
            else value[0]
        )
    if value.ndim != 2:
        raise PackageError(f"depth output has an unsupported shape: {value.shape}")
    if value.shape != shape:
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return value


def summarize(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    output_dir: Path,
    source: Path,
) -> dict[str, Any]:
    cv2 = import_optional("cv2", "install the packages in requirements.txt")
    input_height, input_width = geometry["input_size"]
    depth = depth_map(outputs[0], (int(input_height), int(input_width)))
    ratio = float(geometry["ratio"])
    left, top = (round(v) for v in geometry["pad"])
    original_height, original_width = geometry["original_size"]
    content_height = max(1, round(original_height * ratio))
    content_width = max(1, round(original_width * ratio))
    cropped = depth[top : top + content_height, left : left + content_width]
    restored = cv2.resize(
        cropped,
        (int(original_width), int(original_height)),
        interpolation=cv2.INTER_LINEAR,
    )
    stem = source.stem
    npy_path = output_dir / f"{stem}.depth.npy"
    np.save(npy_path, restored.astype(np.float32))
    summary: dict[str, Any] = {
        "task": customer_contract["task"],
        "depth_npy": str(npy_path),
        "depth_shape": [int(restored.shape[0]), int(restored.shape[1])],
        "depth_min": float(restored.min()),
        "depth_max": float(restored.max()),
        "note": "relative depth in model units at the original image resolution",
    }
    if not args.no_preview:
        span = float(restored.max() - restored.min())
        normalized = (
            (restored - restored.min()) / span if span > 0 else np.zeros_like(restored)
        )
        preview = (normalized * 255.0).astype(np.uint8)
        png_path = output_dir / f"{stem}.depth.png"
        cv2.imwrite(str(png_path), preview)
        summary["depth_png"] = str(png_path)
    return summary


def prediction_records(
    outputs: list[np.ndarray],
    customer_contract: dict[str, Any],
    geometry: dict[str, Any],
    args: Any,
    image_id: int,
) -> list[dict[str, Any]]:
    """Keep input-space depth and LetterBox geometry for Dataset-owned scoring."""
    height, width = geometry["input_size"]
    depth = depth_map(outputs[0], (int(height), int(width)))
    wire = np.ascontiguousarray(depth, dtype="<f4")
    return [
        {
            "image_id": image_id,
            "depth": {
                "encoding": "f32le-zlib-base64",
                "size": [int(height), int(width)],
                "data": base64.b64encode(zlib.compress(wire.tobytes())).decode("ascii"),
            },
            "geometry": {
                "ratio": float(geometry["ratio"]),
                "pad": [float(value) for value in geometry["pad"]],
            },
        }
    ]
