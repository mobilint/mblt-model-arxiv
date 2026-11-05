"""Calibration selection: one image per class from a directory per class.

The source directory holds one sub-directory per class, named as listed in
calibration_spec.json. One image per class is chosen with a single seeded generator in
class order, so the same source directory always yields the same calibration set.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from package_common import PackageError

LAYOUT = "class-directories.v1"
SELECTION = "one-per-class.v1"


def describe_source(spec: dict[str, Any]) -> str:
    classes = spec["layout"]["classes"]
    return (
        f"a directory with {len(classes)} class sub-directories "
        f"({classes[0]} ... {classes[-1]}), each holding that class's images"
    )


def _images(directory: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def select(spec: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    classes = spec["layout"]["classes"]
    suffixes = {suffix.lower() for suffix in spec["layout"]["image_suffixes"]}
    missing = [name for name in classes if not (source / name).is_dir()]
    if missing:
        raise PackageError(
            f"{len(missing)} class directories are missing under {source} (first: "
            f"{missing[0]}); the source must contain one directory per class as listed in "
            "calibration_spec.json"
        )
    generator = random.Random(int(spec["selection"]["seed"]))
    selected: list[dict[str, Any]] = []
    for index, name in enumerate(classes):
        images = _images(source / name, suffixes)
        if not images:
            raise PackageError(
                f"class directory has no supported images: {source / name}"
            )
        image = generator.choice(images)
        selected.append(
            {
                "class_index": index,
                "class": name,
                "source": str(image.relative_to(source)),
                "file": f"{index:04d}_{name}_{image.name}",
                "path": image,
            }
        )
    return selected
