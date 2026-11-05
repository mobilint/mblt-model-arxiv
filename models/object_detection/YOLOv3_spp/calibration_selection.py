"""Calibration selection: a seeded random sample from one flat image directory.

The source directory holds images directly. The images are sorted by name and a fixed
number of them is drawn with one seeded generator, so the same source directory always
yields the same calibration set.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from package_common import PackageError

LAYOUT = "flat-directory.v1"
SELECTION = "seeded-sample.v1"


def describe_source(spec: dict[str, Any]) -> str:
    return (
        f"one flat directory of images (at least {spec['selection']['sample_size']} files "
        f"with suffixes {', '.join(spec['layout']['image_suffixes'])})"
    )


def select(spec: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    suffixes = {suffix.lower() for suffix in spec["layout"]["image_suffixes"]}
    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    size = int(spec["selection"]["sample_size"])
    if len(images) < size:
        raise PackageError(
            f"{source} holds {len(images)} supported images; the calibration profile needs "
            f"at least {size}"
        )
    chosen = random.Random(int(spec["selection"]["seed"])).sample(images, size)
    return [
        {
            "sample_index": index,
            "source": str(image.relative_to(source)),
            "file": f"{index:04d}_{image.name}",
            "path": image,
        }
        for index, image in enumerate(chosen)
    ]
