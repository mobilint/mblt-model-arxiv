"""Bounded image streaming with one model launch and ordered sync/async results."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import postprocessing
import preprocessing
from inference import MxqRunner, add_runtime_arguments, open_runner
from package_common import (
    ARTIFACTS_DIR,
    PackageError,
    artifact_path,
    load_contract,
    require_file,
    write_json,
)

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def stream_arguments(customer_contract: dict, asynchronous: bool) -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        required=True,
        help="images or image directories",
    )
    add_runtime_arguments(argument_parser, customer_contract)
    argument_parser.add_argument("--preprocess-workers", type=int, default=2)
    argument_parser.add_argument(
        "--queue-depth",
        type=int,
        default=4,
        help="maximum in-flight NPU requests (async only)",
    )
    argument_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the image list without relaunching the model",
    )
    argument_parser.add_argument(
        "--limit", type=int, default=None, help="maximum images before repetitions"
    )
    argument_parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="save raw arrays and task summaries; omitted for throughput measurement",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR
        / ("inference_async" if asynchronous else "inference_sync"),
    )
    postprocessing.add_arguments(argument_parser, customer_contract)
    args = argument_parser.parse_args()
    for name in ("preprocess_workers", "queue_depth", "repeat"):
        if getattr(args, name) <= 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        argument_parser.error("--limit must be positive")
    if asynchronous and args.framework != "mxq":
        argument_parser.error(
            "asynchronous inference uses qbruntime and requires --framework mxq"
        )
    return args


def image_sources(sources: list[Path], limit: int | None) -> list[Path]:
    result = []
    for source in sources:
        source = source.expanduser().resolve()
        if source.is_dir():
            result.extend(
                sorted(
                    p
                    for p in source.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        else:
            result.append(require_file(source, "pass an existing image or directory"))
    if limit is not None:
        result = result[:limit]
    if not result:
        raise PackageError("no input images found")
    return result


def prepared_images(
    sources: list[Path], repeat: int, customer_contract: dict, workers: int
):
    """Prefetch at most twice the worker count, retaining deterministic source order."""
    paths = (source for _ in range(repeat) for source in sources)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers * 2):
            source = next(paths, None)
            if source is None:
                break
            pending.append(
                (
                    source,
                    executor.submit(
                        preprocessing.preprocess, source, customer_contract
                    ),
                )
            )
        while pending:
            source, future = pending.popleft()
            tensor, geometry = future.result()
            next_source = next(paths, None)
            if next_source is not None:
                pending.append(
                    (
                        next_source,
                        executor.submit(
                            preprocessing.preprocess, next_source, customer_contract
                        ),
                    )
                )
            yield source, tensor, geometry


def add_pipeline_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument(
        "--inference-mode", choices=("sync", "async"), default="sync"
    )
    argument_parser.add_argument(
        "--queue-depth",
        type=int,
        default=4,
        help="maximum in-flight NPU requests (async only)",
    )
    argument_parser.add_argument("--preprocess-workers", type=int, default=2)


def validate_pipeline_arguments(argument_parser: argparse.ArgumentParser, args) -> None:
    for name in ("queue_depth", "preprocess_workers"):
        if getattr(args, name) <= 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.inference_mode == "async" and args.framework != "mxq":
        argument_parser.error(
            "asynchronous inference uses qbruntime and requires --framework mxq"
        )


@contextmanager
def inference_outputs(sources, repeat, customer_contract, args, artifact, asynchronous):
    """Yield ordered outputs; bound preprocessing and native requests, then drain and close.

    The context owns the runner and both iterators, including when a consumer raises
    during decoding or stops early. Each pending request retains its input storage.
    """
    runner = (
        MxqRunner(artifact, args, async_pipeline=True)
        if asynchronous
        else open_runner(args.framework, artifact, args)
    )
    pending = deque()
    prepared = prepared_images(
        sources, repeat, customer_contract, args.preprocess_workers
    )

    def ordered_outputs():
        for source, tensor, geometry in prepared:
            if asynchronous:
                tensor = np.ascontiguousarray(tensor, dtype=np.float32)
                pending.append(
                    (source, geometry, tensor, runner.model.infer_async(tensor))
                )
                if len(pending) < args.queue_depth:
                    continue
                source, geometry, tensor, future = pending.popleft()
                outputs = future.get()
            else:
                outputs = runner(tensor)
            yield source, outputs, geometry
        while pending:
            source, geometry, tensor, future = pending.popleft()
            yield source, future.get(), geometry

    results = ordered_outputs()
    try:
        yield results
    finally:
        try:
            results.close()
            prepared.close()
        finally:
            try:
                while pending:
                    _, _, _tensor, future = pending.popleft()
                    try:
                        future.get()
                    except Exception:
                        LOGGER.exception(
                            "An outstanding inference failed during cleanup"
                        )
            finally:
                runner.close()


def output_result(index, source, outputs, geometry, customer_contract, args):
    if not args.save_outputs:
        return
    folder = args.output_dir.expanduser().resolve() / f"{index:08d}"
    folder.mkdir(parents=True, exist_ok=False)
    raw = folder / "outputs.npz"
    np.savez(raw, *outputs)
    summary = {
        "source": str(source),
        "raw_outputs": str(raw),
        "output_shapes": [list(x.shape) for x in outputs],
        "geometry": geometry,
    }
    summary.update(
        postprocessing.summarize(
            outputs, customer_contract, geometry, args, folder, source
        )
    )
    write_json(folder / "summary.json", summary)


def stream_inference(*, asynchronous: bool) -> None:
    customer_contract = load_contract()
    args = stream_arguments(customer_contract, asynchronous)
    sources = image_sources(args.source, args.limit)
    artifact = (
        args.artifact.expanduser().resolve()
        if args.artifact
        else artifact_path(customer_contract, args.framework)
    )
    require_file(artifact, "run export.py or compile.py first")
    completed = 0
    with inference_outputs(
        sources, args.repeat, customer_contract, args, artifact, asynchronous
    ) as results:
        started = time.perf_counter()
        for source, outputs, geometry in results:
            output_result(completed, source, outputs, geometry, customer_contract, args)
            completed += 1
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "model": customer_contract["model"],
                "mode": "async" if asynchronous else "sync",
                "framework": args.framework,
                "images": completed,
                "seconds": elapsed,
                "images_per_second": completed / elapsed,
                "includes": "preprocessing and inference; postprocessing and output writes only when enabled; excludes model launch",
                "queue_depth": args.queue_depth if asynchronous else 1,
                "preprocess_workers": args.preprocess_workers,
                "outputs_saved": args.save_outputs,
            },
            indent=2,
        )
    )
