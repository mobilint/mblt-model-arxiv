#!/usr/bin/env python3
"""Download and prepare NYU Depth V2 validation data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

from download import resolve_source
from workflow import IMAGE_SUFFIXES, extract_zip, replace

# The dataset's own directory: default preparation location and home of its data files.
DATASET_HOME = Path(__file__).resolve().parent

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip"
EXPECTED_VALIDATION_PAIRS = 654
DEFAULT_CALIBRATION_IMAGES = 100


def validate_evaluation_dataset(evaluation_dir: Path) -> None:
    """Validate the prepared NYU Depth V2 validation layout."""

    for name in ("images", "depth"):
        if not (evaluation_dir / name).is_dir():
            raise FileNotFoundError(
                f"NYU Depth evaluation directory is missing {name}/: {evaluation_dir}"
            )


def _evaluation_images(evaluation_dir: Path) -> list[Path]:
    """Return the validation images that have a matching depth target, sorted."""

    depth_dir = evaluation_dir / "depth"
    return sorted(
        path
        for path in (evaluation_dir / "images").iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and (depth_dir / (path.stem + ".npy")).is_file()
    )


def make_calibration_dataset(
    evaluation_dir: Path,
    calibration_dir: Path,
    sample_size: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 0,
) -> list[dict[str, int | str]]:
    """Create a deterministic flat NYU Depth calibration directory atomically.

    Every Model of this dataset calibrates on the set one (prepare.py, seed) pair
    selects, so the sample must depend on nothing but the
    sorted evaluation images, ``sample_size`` and ``seed``. A manifest recording
    all three is written inside the directory; ``prepare`` treats its presence as
    proof the set already exists and does no work on a second call.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    evaluation_dir = Path(evaluation_dir).expanduser().resolve()
    calibration_dir = Path(calibration_dir).expanduser().resolve()
    if (
        calibration_dir == evaluation_dir
        or calibration_dir.is_relative_to(evaluation_dir)
        or evaluation_dir.is_relative_to(calibration_dir)
    ):
        raise ValueError("calibration_dir must not overlap evaluation_dir.")
    validate_evaluation_dataset(evaluation_dir)
    images = _evaluation_images(evaluation_dir)
    if sample_size > len(images):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds the {len(images)} available images."
        )

    calibration_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{calibration_dir.name}-staging-", dir=calibration_dir.parent
        )
    )
    selected = random.Random(seed).sample(images, sample_size)
    manifest: list[dict[str, int | str]] = []
    try:
        for index, image in enumerate(selected):
            destination_name = f"{index:04d}_{image.name}"
            shutil.copy2(image, staging_dir / destination_name)
            manifest.append(
                {
                    "sample_index": index,
                    "source": str(image.relative_to(evaluation_dir)),
                    "file": destination_name,
                }
            )
        (staging_dir / "manifest.json").write_text(
            json.dumps(
                {"seed": seed, "sample_size": sample_size, "samples": manifest},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        replace(staging_dir, calibration_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return manifest


def main(a):
    root = a.dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = resolve_source(a.source, a.download_dir or root / "downloads", a.retries)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        extract_zip(source, work) if source.is_file() else None
        base = work if source.is_file() else source
        candidates = list(base.rglob("images/val"))
        image_dir = next(
            (
                path
                for path in candidates
                if (path.parent.parent / "depth" / "val").is_dir()
            ),
            None,
        )
        if image_dir is None:
            raise ValueError(
                "NYU source must contain matching images/val and depth/val directories"
            )
        depth_dir = image_dir.parent.parent / "depth" / "val"
        images = sorted(
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (depth_dir / (p.stem + ".npy")).is_file()
        )
        if len(images) != EXPECTED_VALIDATION_PAIRS:
            raise ValueError(
                f"Expected {EXPECTED_VALIDATION_PAIRS} matched NYU validation pairs, "
                f"found {len(images)}."
            )
        stage = Path(tempfile.mkdtemp(prefix=".nyu-eval-", dir=root))
        (stage / "images").mkdir()
        (stage / "depth").mkdir()
        for image in images:
            shutil.copy2(image, stage / "images" / image.name)
            shutil.copy2(
                depth_dir / (image.stem + ".npy"),
                stage / "depth" / (image.stem + ".npy"),
            )
        replace(stage, root / "eval")
    make_calibration_dataset(root / "eval", root / "calib", a.sample_size, a.seed)


def parse(argv=None):
    p = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    p.add_argument("--source", default=URL)
    p.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    p.add_argument("--download-dir", type=Path)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args(argv)


def run(argv) -> int:
    main(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
