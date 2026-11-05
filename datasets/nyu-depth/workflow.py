"""Shared safe staging and calibration helpers for dataset preparation."""

from __future__ import annotations

import json
import random
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (root / item.filename).resolve().is_relative_to(
                root
            ) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {item.filename}")
        z.extractall(root)


def replace(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup") if target.exists() else None
    if backup:
        if backup.exists():
            raise FileExistsError(f"Existing backup: {backup}")
        target.replace(backup)
    try:
        staging.replace(target)
    except BaseException:
        if backup and not target.exists():
            backup.replace(target)
        raise
    if backup:
        shutil.rmtree(backup)


def publish(*moves: tuple[Path, Path]) -> None:
    """Replace every ``(staging, target)`` pair, or none of them.

    An evaluation split and its calibration set must land together: publishing ``eval/``
    first and failing on ``calib/`` would pair a new split with a stale calibration set.
    On failure every target already touched is restored from its backup and every moved
    staging tree is put back where it was, so the caller can inspect or retry.
    """
    done: list[tuple[Path, Path, Path | None, bool]] = []
    try:
        for staging, target in moves:
            backup = (
                target.with_name(f".{target.name}.backup") if target.exists() else None
            )
            if backup is not None:
                if backup.exists():
                    raise FileExistsError(f"Existing backup: {backup}")
                target.replace(backup)
            done.append((staging, target, backup, False))
            staging.replace(target)
            done[-1] = (staging, target, backup, True)
    except BaseException:
        for staging, target, backup, moved in reversed(done):
            if moved:
                target.replace(staging)
            if backup is not None:
                backup.replace(target)
        raise
    for _staging, _target, backup, _moved in done:
        if backup is not None:
            shutil.rmtree(backup)


def calibration(
    images: list[Path],
    output: Path,
    size: int,
    seed: int,
    relative_to: Path | None = None,
) -> None:
    """Copy a seeded sample of ``images`` into ``output`` with a manifest.

    ``relative_to`` makes the recorded ``source`` paths relative to that root, so a set
    built inside a temporary staging directory records the same manifest as one built in
    place.
    """
    if size <= 0 or size > len(images):
        raise ValueError(f"Calibration size must be 1..{len(images)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        records = []
        for index, image in enumerate(random.Random(seed).sample(sorted(images), size)):
            name = f"{index:04d}_{image.name}"
            shutil.copy2(image, stage / name)
            source = image.relative_to(relative_to) if relative_to else image
            records.append({"sample_index": index, "source": str(source), "file": name})
        (stage / "manifest.json").write_text(
            json.dumps(
                {"seed": seed, "sample_size": size, "samples": records}, indent=2
            )
            + "\n"
        )
        replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
