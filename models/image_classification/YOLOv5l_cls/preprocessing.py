"""Image preprocessing: resize, center crop, and ImageNet normalization.

Resize the shorter side, center-crop, scale to [0, 1], and normalize with the ImageNet
mean and standard deviation. Returns a float32 HWC tensor and the geometry record that
inference.py stores beside the outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError, import_optional


def preprocess(
    source: Path, customer_contract: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    pil = import_optional("PIL.Image", "install the packages in requirements.txt")
    image_module = pil.Image
    params = customer_contract["preprocessing"]["params"]
    try:
        image = image_module.open(source).convert("RGB")
    except OSError as error:
        raise PackageError(f"cannot read the input image {source}: {error}") from error
    original = (image.height, image.width)
    size = int(params["resize_shorter_side"])
    width, height = image.size
    if width <= height:
        width, height = size, int(size * height / width)
    else:
        width, height = int(size * width / height), size
    resampling = (
        image_module.Resampling.BICUBIC
        if params.get("interpolation") == "bicubic"
        else image_module.Resampling.BILINEAR
    )
    image = image.resize((width, height), resampling)
    crop_height, crop_width = params["crop"]
    left = round((image.width - crop_width) / 2.0)
    top = round((image.height - crop_height) / 2.0)
    image = image.crop((left, top, left + crop_width, top + crop_height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(params["mean"], dtype=np.float32)) / np.asarray(
        params["std"], dtype=np.float32
    )
    tensor = np.ascontiguousarray(array, dtype=np.float32)
    return tensor, {"original_size": list(original), "crop": [crop_height, crop_width]}
