#!/usr/bin/env python3
"""Prepare Market-1501: download or accept the official archive, stage the splits, pick calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from reid import (
    SPLIT_DIRS,
    check_counts,
    jpg_files,
    locate_split,
    split_samples,
)
from source_identity import source_content_sha256
from workflow import calibration, extract_zip, publish

from dataloader import CAMERAS, EXPECTED_GALLERY, EXPECTED_QUERY

DATASET_HOME = Path(__file__).resolve().parent

DRIVE_ID = "0B8-rUzbwVRk0c054eEozWG9COHM"
ARCHIVE = "Market-1501-v15.09.15.zip"
SHA256 = "416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96"
EXPECTED = {"query": 3368, "bounding_box_test": 19732, "bounding_box_train": 12936}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(download_dir: Path) -> Path:
    """Fetch the official archive from its Google Drive file through gdown under uv."""
    target = download_dir / ARCHIVE
    if target.is_file() and _sha256(target) == SHA256:
        print(f"Using existing download: {target}")
        return target
    if shutil.which("uv") is None:
        raise FileNotFoundError(
            "Provide --source Market-1501-v15.09.15.zip, or install uv for automatic download."
        )
    subprocess.run(
        ["uv", "run", "--with", "gdown", "gdown", DRIVE_ID, "--output", str(target)],
        check=True,
    )
    return target


def stage(
    source: Path, root: Path, *, verify: bool, seed: int, sample_size: int
) -> None:
    if source.is_file():
        if verify and _sha256(source) != SHA256:
            raise ValueError(f"{source} does not match the recorded Market-1501 sha256")
        with tempfile.TemporaryDirectory(dir=root) as work:
            extract_zip(source, Path(work) / "archive")
            _stage_extracted(
                Path(work) / "archive",
                root,
                seed=seed,
                sample_size=sample_size,
                source_sha256=_sha256(source),
            )
    elif source.is_dir():
        _stage_extracted(source, root, seed=seed, sample_size=sample_size)
    else:
        raise FileNotFoundError(f"Market-1501 source does not exist: {source}")


def _stage_extracted(
    extracted: Path,
    root: Path,
    *,
    seed: int = 0,
    sample_size: int = 100,
    source_sha256: str | None = None,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".market1501-eval-", dir=root))
    calib_staging = Path(tempfile.mkdtemp(prefix=".market1501-calib-", dir=root))
    try:
        for name in SPLIT_DIRS:
            shutil.copytree(
                locate_split(extracted, name),
                staging / name,
                ignore=shutil.ignore_patterns("Thumbs.db"),
            )
        check_counts(staging, EXPECTED)
        query = split_samples(
            staging / "query", "query", cameras=CAMERAS, skip_junk=True
        )
        gallery = split_samples(
            staging / "bounding_box_test", "gallery", cameras=CAMERAS, skip_junk=True
        )
        if len(query) != EXPECTED_QUERY or len(gallery) != EXPECTED_GALLERY:
            raise ValueError(
                f"Staged Market-1501 validation holds {len(query)} query and {len(gallery)} gallery crops; expected {EXPECTED_QUERY} and {EXPECTED_GALLERY}"
            )
        content_sha256 = (
            source_content_sha256(
                staging,
                [path for split in SPLIT_DIRS for path in jpg_files(staging / split)],
            )
            if source_sha256 is None
            else None
        )
        (staging / "source.json").write_text(
            json.dumps(
                {
                    "archive": ARCHIVE if source_sha256 is not None else None,
                    "sha256": source_sha256,
                    "content_sha256": content_sha256,
                    "verified": source_sha256 == SHA256,
                    "source_type": "archive"
                    if source_sha256 is not None
                    else "extracted_directory",
                },
                indent=2,
            )
            + "\n"
        )
        calibration(
            list(jpg_files(staging / "bounding_box_train")),
            calib_staging / "calib",
            sample_size,
            seed,
            relative_to=staging,
        )
        publish((staging, root / "eval"), (calib_staging / "calib", root / "calib"))
    finally:
        for tree in (staging, calib_staging):
            if tree.exists():
                shutil.rmtree(tree)


def parse(argv=None):
    parser = argparse.ArgumentParser(
        prog="python prepare.py", description=__doc__
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Local Market-1501-v15.09.15.zip or an extracted directory (default: download).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    parser.add_argument("--no-verify-checksums", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(args) -> None:
    root = args.dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.source is None:
        download = root / "downloads"
        download.mkdir(exist_ok=True)
        source = _download(download)
    else:
        source = Path(args.source).expanduser().resolve()
    stage(
        source,
        root,
        verify=not args.no_verify_checksums,
        seed=args.seed,
        sample_size=args.sample_size,
    )


def run(argv) -> int:
    main(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(None))
