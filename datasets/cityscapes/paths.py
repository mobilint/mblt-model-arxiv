"""Small path helpers shared by dataset-owned validation loaders."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


def image_files(directory: Path, *, recursive: bool = False) -> tuple[Path, ...]:
    """Return deterministic supported image files from an existing directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory is missing: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    paths = tuple(
        sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"Image directory contains no supported images: {directory}")
    return paths


def files_by_stem(directory: Path, suffixes: frozenset[str]) -> dict[str, Path]:
    """Return direct-child files keyed by unique stem."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory is missing: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate filename stem {path.stem!r} in {directory}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"Directory contains no matching files: {directory}")
    return result
