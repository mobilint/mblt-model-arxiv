#!/usr/bin/env python3
"""Prepare Occluded-Duke by re-splitting a supplied DukeMTMC-reID archive with the published lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from reid import check_counts, jpg_files, locate_split, split_samples
from source_identity import source_content_sha256
from workflow import calibration, extract_zip, publish

from dataloader import CAMERAS, EXPECTED_GALLERY, EXPECTED_QUERY

SHA256 = "932ae18937b6a77bc59846d4fb00da4ee02cdda93329ca0537ad899a569e3505"
EXPECTED = {"query": 2210, "bounding_box_test": 17661, "bounding_box_train": 15618}
# Published by lightas/Occluded-DukeMTMC-Dataset (commit dcba185bb20cbd53d3da2c8a4bfc25aa6971ce1d):
# list name -> the DukeMTMC-reID split it is copied into.
LISTS = {
    "query": "query",
    "gallery": "bounding_box_test",
    "train": "bounding_box_train",
}
DATASET_HOME = Path(__file__).resolve().parent
SPLITS_DIR = DATASET_HOME


def read_list(name: str) -> tuple[str, ...]:
    lines = (SPLITS_DIR / f"{name}.list").read_text(encoding="utf-8").splitlines()
    names = tuple(line.strip() for line in lines if line.strip())
    if len(names) != len(set(names)):
        raise ValueError(f"{name}.list holds duplicate file names")
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(
    source: Path, root: Path, *, verify: bool, seed: int, sample_size: int
) -> None:
    if source.is_file():
        if verify and _sha256(source) != SHA256:
            raise ValueError(
                f"{source} does not match the recorded DukeMTMC-reID sha256; pass "
                "--no-verify-checksums for a differently packaged archive"
            )
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
        raise FileNotFoundError(f"DukeMTMC-reID source does not exist: {source}")


def _stage_extracted(
    extracted: Path,
    root: Path,
    *,
    seed: int = 0,
    sample_size: int = 100,
    source_sha256: str | None = None,
) -> None:
    """Copy each listed crop from its Duke split; the reference script falls back to the
    Duke gallery because Occluded-Duke moves occluded gallery crops into its query split."""
    duke = {name: locate_split(extracted, name) for name in LISTS.values()}
    staging = Path(tempfile.mkdtemp(prefix=".occluded-duke-eval-", dir=root))
    calib_staging = Path(tempfile.mkdtemp(prefix=".occluded-duke-calib-", dir=root))
    try:
        for list_name, split in LISTS.items():
            target = staging / split
            target.mkdir()
            for file_name in read_list(list_name):
                candidate = duke[split] / file_name
                if not candidate.is_file():
                    candidate = duke["bounding_box_test"] / file_name
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"{file_name} from {list_name}.list is not in the DukeMTMC-reID archive"
                    )
                shutil.copy2(candidate, target / file_name)
        check_counts(staging, EXPECTED)
        query = split_samples(
            staging / "query", "query", cameras=CAMERAS, skip_junk=False
        )
        gallery = split_samples(
            staging / "bounding_box_test", "gallery", cameras=CAMERAS, skip_junk=False
        )
        if len(query) != EXPECTED_QUERY or len(gallery) != EXPECTED_GALLERY:
            raise ValueError(
                f"Staged Occluded-Duke validation holds {len(query)} query and {len(gallery)} gallery crops; expected {EXPECTED_QUERY} and {EXPECTED_GALLERY}"
            )
        content_sha256 = (
            source_content_sha256(
                staging,
                [path for split in EXPECTED for path in jpg_files(staging / split)],
            )
            if source_sha256 is None
            else None
        )
        (staging / "source.json").write_text(
            json.dumps(
                {
                    "archive": "DukeMTMC-reID.zip"
                    if source_sha256 is not None
                    else None,
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
        "--source", required=True, help="DukeMTMC-reID.zip or an extracted directory."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    parser.add_argument("--no-verify-checksums", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(args) -> None:
    root = args.dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stage(
        Path(args.source).expanduser().resolve(),
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
