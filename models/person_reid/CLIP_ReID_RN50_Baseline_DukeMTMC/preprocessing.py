"""RGB resize to a fixed spatial shape and explicit channel normalization."""

from pathlib import Path
from typing import Any

import numpy as np
from package_common import PackageError, import_optional


def preprocess(
    source: Path, customer_contract: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    pil = import_optional("PIL.Image", "install the packages in requirements.txt").Image
    params = customer_contract["preprocessing"]["params"]
    try:
        image = pil.open(source).convert("RGB")
    except OSError as error:
        raise PackageError(f"cannot read {source}: {error}") from error
    original = [image.height, image.width]
    height, width = params["size"]
    modes = {"bilinear": pil.Resampling.BILINEAR, "bicubic": pil.Resampling.BICUBIC}
    image = image.resize(
        (width, height), modes[params.get("interpolation", "bilinear")]
    )
    tensor = (
        np.asarray(image, dtype=np.float32) / 255
        - np.asarray(params["mean"], dtype=np.float32)
    ) / np.asarray(params["std"], dtype=np.float32)
    return np.ascontiguousarray(tensor), {"original_size": original}
