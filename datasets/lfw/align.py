# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "torch==2.14.0",
#   "numpy==2.5.3",
#   "pillow==12.3.0",
#   "opencv-python-headless==5.0.0.93",
# ]
# ///
"""Align LFW faces to 112x112 with the MTCNN that the EdgeFace repository ships.

Runs in its own environment: it imports ``face_alignment`` from a clone of
otroshi/edgeface at a pinned commit (AdaFace's MTCNN with bundled P/R/O-net weights) and
writes one aligned crop per input image plus ``alignment.json`` naming the images where
detection failed and a center crop was used instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

DATASET_HOME = Path(__file__).resolve().parent

REVISION = "ce86851cfc37979a9cd2558598d0e9bc592cbba3"
FALLBACK_BOX = (55, 55, 195, 195)  # 140x140 center of the 250x250 LFW frame


def checkout(repo: str, target: Path) -> Path:
    """A clone of ``repo`` at REVISION with no local edits or stray files."""
    if not (target / ".git").is_dir():
        subprocess.run(["git", "clone", "--quiet", repo, str(target)], check=True)
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--quiet", "--detach", REVISION],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "reset", "--hard", "--quiet", REVISION], check=True
    )
    subprocess.run(["git", "-C", str(target), "clean", "-fdxq"], check=True)
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(target), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != REVISION or dirty:
        raise RuntimeError(
            f"{target} is not a clean checkout of {REVISION} (HEAD {head}, status {dirty[:200]!r})"
        )
    return target


def alignment_environment() -> dict:
    """Record resolved runtime versions, not just requested dependencies."""
    script = DATASET_HOME / "_private" / "align.py"
    if not script.is_file():
        script = DATASET_HOME / "align.py"
    lock = script.with_suffix(script.suffix + ".lock")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
            )
        ),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="flat directory of jpgs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default="https://github.com/otroshi/edgeface")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--single-sample",
        action="store_true",
        help="Allow a single sample fallback; aggregate health checks belong to batch preparation",
    )
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    environment = alignment_environment()
    clone = checkout(args.repo, args.checkout)
    sys.path.insert(0, str(clone))
    from face_alignment import mtcnn

    # CPU only: 0.02 s per LFW frame, and the result does not depend on the host's GPU.
    device = "cpu"
    detector = mtcnn.MTCNN(device=device, crop_size=(112, 112))
    args.output.mkdir(parents=True, exist_ok=True)
    fallbacks: list[str] = []
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() == ".jpg")
    for index, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        # The pinned detector answers a no-face frame with ([], []); any exception is a
        # real failure and must surface rather than degrade the frame to a center crop.
        boxes, faces = detector.align_multi(image)
        if faces:
            # LFW frames are centered on their subject; bystanders' faces sit off-center,
            # and MTCNN's first detection is not always the subject.
            center = np.array(image.size, dtype=np.float64) / 2.0
            centers = (np.asarray(boxes)[:, :2] + np.asarray(boxes)[:, 2:4]) / 2.0
            face = faces[int(np.argmin(np.linalg.norm(centers - center, axis=1)))]
        else:
            face = image.crop(FALLBACK_BOX).resize(
                (112, 112), Image.Resampling.BILINEAR
            )
            fallbacks.append(path.name)
        face.save(args.output / path.name, quality=95)
        if index % 1000 == 0:
            print(f"aligned {index}/{len(images)}", flush=True)
    (args.output / "alignment.json").write_text(
        json.dumps(
            {
                "repository": args.repo,
                "revision": REVISION,
                "device": device,
                "environment": environment,
                "crop": [112, 112],
                "fallback_box": list(FALLBACK_BOX),
                "fallbacks": fallbacks,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"aligned {len(images)} images, {len(fallbacks)} fallbacks")
    if not args.single_sample and len(fallbacks) > len(images) // 20:
        raise RuntimeError(
            f"{len(fallbacks)} of {len(images)} images fell back to a center crop; "
            "detection is failing systematically rather than on a few frames"
        )


if __name__ == "__main__":
    main()
