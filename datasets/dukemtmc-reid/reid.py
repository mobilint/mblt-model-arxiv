"""Person re-identification benchmark primitives shared by the ReID dataset nodes.

Market-1501, DukeMTMC-reID and Occluded-Duke share one protocol: a query split and a
gallery split of person crops whose file names carry the identity and camera
(``<pid>_c<cam>_...jpg``), a distance matrix between query and gallery embeddings,
and the Market-1501 evaluation rule that discards gallery images taken by the query's
own camera of the same identity. CMC rank-k and mAP follow the reference
implementation (CLIP-ReID / TransReID ``utils/metrics.py::eval_func``).

The Model side owns the embedding: this module only consumes the dense matrix a
``person_reid`` prediction artifact references, so no Model code is imported here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from embeddings import embeddings_by_image_id

FILENAME = re.compile(r"^(-?\d+)_c(\d+)(?:s\d+)?(?:_|\.)")
Split = Literal["query", "gallery"]
MAX_RANK = 50
RANKS = (1, 5, 10)


@dataclass(frozen=True)
class ReidSample:
    """One query or gallery crop with the benchmark-owned identity and camera."""

    image: Path
    pid: int
    camid: int
    split: Split


def parse_name(path: Path) -> tuple[int, int]:
    """Return ``(pid, camid)`` from a ReID file name; camid is the 1-based upstream index."""
    match = FILENAME.match(path.name)
    if match is None:
        raise ValueError(f"ReID file name carries no <pid>_c<cam> token: {path}")
    return int(match.group(1)), int(match.group(2))


def split_samples(
    directory: Path,
    split: Split,
    *,
    cameras: int,
    skip_junk: bool,
) -> tuple[ReidSample, ...]:
    """Enumerate one split directory deterministically.

    ``skip_junk`` drops the ``-1`` distractor identity that Market-1501 ships inside its
    query and gallery folders, exactly as the reference loader does; identity ``0``
    (background crops) stays in the gallery.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"ReID split directory is missing: {directory}")
    samples: list[ReidSample] = []
    for path in sorted(directory.glob("*.jpg")):
        pid, camid = parse_name(path)
        if pid == -1:
            if skip_junk:
                continue
            raise ValueError(f"Unexpected junk identity in {directory}: {path.name}")
        if not 1 <= camid <= cameras:
            raise ValueError(
                f"Camera {camid} outside 1..{cameras} in {directory}: {path.name}"
            )
        samples.append(ReidSample(path, pid, camid - 1, split))
    if not samples:
        raise ValueError(f"ReID split directory holds no images: {directory}")
    return tuple(samples)


def sample_manifest(task: str, samples: tuple[ReidSample, ...]) -> dict[str, Any]:
    return {
        "task": task,
        "samples": [
            {"image": str(sample.image), "image_id": image_id}
            for image_id, sample in enumerate(samples)
        ],
    }


def euclidean_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Squared euclidean distance, the reference ``euclidean_distance`` without the sqrt.

    Ranking is monotone in the square, so CMC and mAP are unchanged.
    """
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    q2 = np.sum(query**2, axis=1, keepdims=True)
    g2 = np.sum(gallery**2, axis=1, keepdims=True).T
    return q2 + g2 - 2.0 * (query @ gallery.T)


def evaluate(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = MAX_RANK,
) -> tuple[np.ndarray, float]:
    """Market-1501 protocol CMC curve and mAP (reference ``eval_func``)."""
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1, kind="stable")
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_ap = []
    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        orig_cmc = matches[q_idx][~remove]
        if not np.any(orig_cmc):
            continue  # the query identity never appears in the gallery
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        # Same-identity/same-camera removal leaves different gallery lengths.
        # Saturate missing ranks with the final CMC value for this query.
        all_cmc.append(
            np.pad(cmc[:max_rank], (0, max_rank - min(len(cmc), max_rank)), mode="edge")
        )
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum() / np.arange(1, orig_cmc.shape[0] + 1)
        all_ap.append(float((tmp_cmc * orig_cmc).sum() / num_rel))
    if not all_cmc:
        raise ValueError("No query identity appears in the gallery")
    cmc_curve = np.asarray(all_cmc, dtype=np.float64).sum(0) / len(all_cmc)
    return cmc_curve, float(np.mean(all_ap))


def score(
    task: str, samples: tuple[ReidSample, ...], artifact: dict[str, Any]
) -> dict[str, Any]:
    """Score a ``person_reid`` artifact against benchmark-owned identities and cameras."""
    if artifact.get("task") != task:
        raise ValueError(f"This benchmark accepts {task} prediction artifacts only")
    image_ids = artifact.get("image_ids")
    if not isinstance(image_ids, list) or any(
        not isinstance(image_id, int) or isinstance(image_id, bool)
        for image_id in image_ids
    ):
        raise TypeError("Prediction artifact image_ids must be an array of integers")
    if image_ids != list(range(len(samples))):
        raise ValueError(
            "Prediction artifact image_ids do not match the manifest order"
        )
    embeddings = embeddings_by_image_id(artifact)
    missing = [i for i in range(len(samples)) if i not in embeddings]
    if missing:
        raise ValueError(f"Prediction artifact is missing image_ids {missing[:5]}")
    unknown = sorted(set(embeddings) - set(range(len(samples))))
    if unknown:
        raise ValueError(f"Prediction artifact has unknown image_ids: {unknown[:5]}")
    query = [i for i, s in enumerate(samples) if s.split == "query"]
    gallery = [i for i, s in enumerate(samples) if s.split == "gallery"]
    if not query or not gallery:
        raise ValueError("Benchmark needs both a query and a gallery split")
    qf = np.stack([embeddings[i] for i in query])
    gf = np.stack([embeddings[i] for i in gallery])
    cmc, mean_ap = evaluate(
        euclidean_distance(qf, gf),
        np.asarray([samples[i].pid for i in query]),
        np.asarray([samples[i].pid for i in gallery]),
        np.asarray([samples[i].camid for i in query]),
        np.asarray([samples[i].camid for i in gallery]),
    )
    return {
        "primary": {"map": round(mean_ap, 5)},
        # A gallery shorter than a rank saturates the CMC curve at its last entry.
        "secondary": {
            f"rank{k}": round(float(cmc[min(k, len(cmc)) - 1]), 5) for k in RANKS
        },
        "sample_count": len(samples),
    }


SPLIT_DIRS = ("query", "bounding_box_test", "bounding_box_train")


def locate_split(root: Path, name: str) -> Path:
    """Find one upstream split directory anywhere under an extracted archive."""
    hits = sorted(
        path
        for path in root.rglob(name)
        if path.is_dir()
        # macOS archives ship a __MACOSX shadow tree of resource-fork files.
        and not any(
            part.startswith(("__MACOSX", ".")) for part in path.relative_to(root).parts
        )
    )
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name!r} directory under {root}, found {len(hits)}"
        )
    return hits[0]


def jpg_files(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in directory.glob("*.jpg") if path.is_file()))


def check_counts(root: Path, expected: dict[str, int]) -> None:
    """Fail unless every split directory holds exactly the published image count."""
    for name, count in expected.items():
        found = len(jpg_files(root / name))
        if found != count:
            raise ValueError(f"{root / name} holds {found} images, expected {count}")
