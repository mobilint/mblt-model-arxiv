"""Image preprocessing: letterbox resize to the model input and RGB scaling.

Read the image with OpenCV, convert BGR to RGB, resize keeping the aspect ratio, pad to the
model input size with the YOLO grey value, and scale to [0, 1]. Returns a float32 HWC
tensor and the letterbox geometry (ratio and left/top padding) needed to map results back
to the original image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError, import_optional


def letterbox(
    image: np.ndarray, shape: tuple[int, int], pad_value: int
) -> tuple[np.ndarray, float, tuple[float, float]]:
    cv2 = import_optional("cv2", "install the packages in requirements.txt")
    height, width = image.shape[:2]
    target_height, target_width = shape
    ratio = min(target_height / height, target_width / width)
    resized = cv2.resize(
        image,
        (round(width * ratio), round(height * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_width, pad_height = (
        target_width - resized.shape[1],
        target_height - resized.shape[0],
    )
    left, top = pad_width // 2, pad_height // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_height - top,
        left,
        pad_width - left,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return padded, ratio, (float(left), float(top))


def preprocess(
    source: Path, customer_contract: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    cv2 = import_optional("cv2", "install the packages in requirements.txt")
    params = customer_contract["preprocessing"]["params"]
    original = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if original is None:
        raise PackageError(f"cannot read the input image {source}")
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width = params["size"]
    padded, ratio, (left, top) = letterbox(
        rgb, (int(height), int(width)), int(params["pad_value"])
    )
    tensor = np.ascontiguousarray(padded.astype(np.float32) / float(params["scale"]))
    geometry = {
        "original_size": [int(original.shape[0]), int(original.shape[1])],
        "ratio": ratio,
        "pad": [left, top],
        "input_size": [int(height), int(width)],
    }
    return tensor, geometry
