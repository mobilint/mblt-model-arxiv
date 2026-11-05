#!/usr/bin/env python3
"""Prepare LFW: fetch the Hugging Face drop, align every face once, stage pairs and calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from face_pairs import read_pair_file
from source_identity import source_content_sha256
from workflow import calibration, publish

DATASET_HOME = Path(__file__).resolve().parent

HF_DATASET = "marcelohaps/lfw"
# Dataset repository commit and the digests of its protocol files; both are also recorded in
# lfw.yaml. A different revision or digest means a different benchmark, not a retry.
REVISION = "12a61458b56d0433d07269dc1d64368abf4f6b4d"
DIGESTS = {
    "pairs.csv": "7f540157be42f57ab5bb1d7ef53b7b379e0331b4842cbd32f4d1b625243e54fe",
    "train/metadata.csv": "ebe9dce66e8c751647086731dc87b329b5cfd2de21d79ae570bb464b8a51709a",
}
EXPECTED_IMAGES = 13_233
EXPECTED_PAIRS = 6_000
EXPECTED_FOLDS = 10
PAIR_COLUMNS = {"pair_id", "fold_id", "is_same", "image_a", "image_b"}


def download(download_dir: Path) -> Path:
    target = download_dir / "lfw"
    if not (target / "pairs.csv").is_file():
        subprocess.run(
            [
                "hf",
                "download",
                HF_DATASET,
                "--repo-type",
                "dataset",
                "--revision",
                REVISION,
                "--local-dir",
                str(target),
            ],
            check=True,
        )
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_digests(source: Path) -> None:
    """Fail unless the protocol files match the recorded revision's digests."""
    for relative, expected in DIGESTS.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"LFW source lacks {relative}: {source}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{path} sha256 {actual} differs from the recorded {expected}; the source "
                f"is not revision {REVISION} of {HF_DATASET} (pass --no-verify-checksums "
                "to prepare a different drop deliberately)"
            )


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not PAIR_COLUMNS <= set(reader.fieldnames):
            raise ValueError(
                f"{path} lacks the LFW pair columns {sorted(PAIR_COLUMNS)}"
            )
        rows = list(reader)
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"{path} holds {len(rows)} pairs, expected {EXPECTED_PAIRS}")
    if len({row["fold_id"] for row in rows}) != EXPECTED_FOLDS:
        raise ValueError(f"{path} does not hold {EXPECTED_FOLDS} folds")
    read_pair_file(path, EXPECTED_PAIRS)
    return rows


def align(images: Path, output: Path, checkout: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError(
            "uv is required to run the isolated LFW alignment script"
        )
    script = DATASET_HOME / "_private" / "align.py"
    if not script.is_file():
        script = DATASET_HOME / "align.py"
    lock = script.with_suffix(script.suffix + ".lock")
    if not lock.is_file():
        raise FileNotFoundError(f"required alignment lockfile is missing: {lock}")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--script",
            str(script),
            "--images",
            str(images.resolve()),
            "--output",
            str(output.resolve()),
            "--checkout",
            str(checkout.resolve()),
        ],
        check=True,
        cwd=script.parent,
        env=environment,
    )


def stage(
    source: Path, root: Path, *, verify: bool, seed: int, sample_size: int
) -> None:
    if verify:
        verify_digests(source)
    pairs = read_pairs(source / "pairs.csv")
    files = sorted(source.glob("train/images/*/*.jpg"))
    if len(files) != EXPECTED_IMAGES:
        raise ValueError(
            f"{source} holds {len(files)} images, expected {EXPECTED_IMAGES}"
        )
    if len({f.name for f in files}) != len(files):
        raise ValueError("LFW image names must be unique")
    staging = Path(tempfile.mkdtemp(prefix=".lfw-eval-", dir=root))
    calib_staging = Path(tempfile.mkdtemp(prefix=".lfw-calib-", dir=root))
    try:
        images = staging / "images"
        images.mkdir()
        for path in files:
            shutil.copy2(path, images / path.name)
        image_sha256 = source_content_sha256(staging, sorted(images.glob("*.jpg")))
        shutil.copy2(source / "pairs.csv", staging / "pairs.csv")
        in_pairs = {row["image_a"] for row in pairs} | {row["image_b"] for row in pairs}
        missing = sorted(name for name in in_pairs if not (images / name).is_file())
        if missing:
            raise ValueError(f"pairs.csv names images that are absent: {missing[:5]}")
        align(images, staging / "aligned", root / "downloads" / "edgeface")
        aligned = sorted((staging / "aligned").glob("*.jpg"))
        if len(aligned) != EXPECTED_IMAGES or {p.name for p in aligned} != {
            p.name for p in files
        }:
            raise ValueError(
                "LFW alignment did not produce exactly one crop for every source image"
            )
        alignment_record = staging / "aligned" / "alignment.json"
        if not alignment_record.is_file():
            raise FileNotFoundError(
                f"LFW alignment provenance missing: {alignment_record}"
            )
        (staging / "source.json").write_text(
            json.dumps(
                {
                    "repository": HF_DATASET,
                    "protocol_revision": REVISION if verify else None,
                    "pairs_sha256": _sha256(source / "pairs.csv"),
                    "metadata_sha256": _sha256(source / "train" / "metadata.csv")
                    if (source / "train" / "metadata.csv").is_file()
                    else None,
                    "source_images_sha256": image_sha256,
                    "alignment": json.loads(alignment_record.read_text()),
                    "protocol_verified": verify,
                    "images_verified_at_revision": False,
                },
                indent=2,
            )
            + "\n"
        )
        pool = staging / "calibration_pool"
        pool.mkdir()
        for path in files:
            if path.name not in in_pairs:
                shutil.copy2(staging / "aligned" / path.name, pool / path.name)
        calibration(
            sorted(pool.glob("*.jpg")),
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
        help="An existing download of the Hugging Face dataset (default: download it).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_HOME)
    parser.add_argument("--no-verify-checksums", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(args) -> None:
    root = args.dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "downloads").mkdir(exist_ok=True)
    source = (
        download(root / "downloads")
        if args.source is None
        else Path(args.source).expanduser().resolve()
    )
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
