#!/usr/bin/env python3
"""Evaluate the ONNX model (CPU) or the compiled MXQ model (NPU) on its benchmark.

Every validation sample of the benchmark Dataset is run in the order the Dataset's
`score.py samples` lists. The predictions are written to --out, scored with the Dataset's
`score.py score`, and compared with the result Mobilint recorded in model_contract.json:
the NPU result for --framework mxq, the float result for --framework onnx. Prerequisites:
the Dataset folder (default: the `dataset_code` path in model_contract.json, relative to
this folder) and its prepared evaluation
directory, passed as --data-path; the runtime prerequisites of inference.py.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import postprocessing
from inference import add_runtime_arguments
from inference_stream import (
    add_pipeline_arguments,
    inference_outputs,
    validate_pipeline_arguments,
)
from package_common import (
    ARTIFACTS_DIR,
    BUNDLE_DIR,
    PackageError,
    artifact_path,
    exit_with_package_error,
    load_contract,
    refuse_overwrite,
    write_json,
)

# The summary written beside the prediction artifact.
RESULT_NAME = "result.json"


def parse_args(customer_contract: dict, evaluation: dict) -> argparse.Namespace:
    dataset = evaluation["dataset"]
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help=f"the prepared {dataset} evaluation directory, <dataset-dir>/eval",
    )
    argument_parser.add_argument(
        "--dataset-code",
        type=Path,
        default=(
            BUNDLE_DIR / evaluation["dataset_code"]
            if "dataset_code" in evaluation
            else None
        ),
        required="dataset_code" not in evaluation,
        help=f"the {dataset} Dataset folder whose score.py lists and scores the "
        "benchmark (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--out",
        type=Path,
        default=ARTIFACTS_DIR / "evaluation" / "predictions.json",
        help="prediction artifact to write; the result goes beside it as "
        f"{RESULT_NAME} (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N samples, to check the setup; the result is then "
        "not comparable with the expected one",
    )
    argument_parser.add_argument(
        "--force",
        action="store_true",
        help=f"replace an existing prediction artifact and its {RESULT_NAME}",
    )
    add_runtime_arguments(argument_parser, customer_contract)
    add_pipeline_arguments(argument_parser)
    postprocessing.add_arguments(argument_parser, customer_contract)
    args = argument_parser.parse_args()
    validate_pipeline_arguments(argument_parser, args)
    if args.out.name == RESULT_NAME:
        argument_parser.error(
            f"--out names the prediction artifact; {RESULT_NAME} beside it holds the "
            "result, so choose another file name"
        )
    return args


def dataset_command(dataset_code: Path, *arguments: str) -> Any:
    """Run the Dataset folder's score.py and return the JSON it prints."""
    script = dataset_code / "score.py"
    if not script.is_file():
        raise PackageError(
            f"{script} not found; pass the Dataset folder with --dataset-code"
        )
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=str(dataset_code),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PackageError(
            f"{script.name} {arguments[0]} failed: "
            + (result.stderr or result.stdout).strip()
        )
    return json.loads(result.stdout)


def compare(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """Every recorded metric must come back from the scorer as a finite number."""
    difference = {}
    for group in ("primary", "secondary"):
        for metric, value in expected.get(group, {}).items():
            measured = result.get(group, {}).get(metric)
            if (
                not isinstance(measured, (int, float))
                or isinstance(measured, bool)
                or not math.isfinite(measured)
            ):
                raise PackageError(
                    f"the scorer returned {group} {metric} = {measured!r}; expected a "
                    "finite number to compare with the recorded result"
                )
            difference[metric] = round(measured - value, 5)
    return difference


def check_sample_count(result: dict, recorded: dict, dataset: str) -> None:
    """A full run must score as many samples as the recorded result did."""
    scored = result.get("sample_count")
    if not isinstance(scored, int) or isinstance(scored, bool) or scored <= 0:
        raise PackageError(
            f"the {dataset} scorer reported sample_count {scored!r}; expected a positive "
            "integer"
        )
    if scored != recorded["sample_count"]:
        raise PackageError(
            f"scored {result.get('sample_count')} samples, but the recorded result has "
            f"{recorded['sample_count']}; check that --data-path is the complete "
            f"prepared {dataset} evaluation directory"
        )


def main() -> None:
    customer_contract = load_contract()
    evaluation = customer_contract.get("evaluation")
    if not evaluation:
        raise PackageError("model_contract.json declares no benchmark to evaluate on")
    args = parse_args(customer_contract, evaluation)
    if args.limit is not None and args.limit <= 0:
        raise PackageError("--limit must be positive")
    data_path = args.data_path.expanduser().resolve()
    artifact = (
        args.artifact.expanduser().resolve()
        if args.artifact is not None
        else artifact_path(customer_contract, args.framework)
    )
    if not artifact.is_file():
        step = "export.py" if args.framework == "onnx" else "compile.py"
        raise PackageError(f"model file not found: {artifact}; run {step} first")
    # Before inference, so a rerun never spends an evaluation only to be refused.
    out = args.out.expanduser().resolve()
    refuse_overwrite(out, args.force)
    refuse_overwrite(out.with_name(RESULT_NAME), args.force)
    dataset_code = args.dataset_code.expanduser().resolve()
    samples = dataset_command(
        dataset_code,
        "samples",
        "--task",
        evaluation["task"],
        "--data-path",
        str(data_path),
    )["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]
    records: list[dict[str, Any]] = []
    sources = [Path(sample["image"]) for sample in samples]
    with inference_outputs(
        sources, 1, customer_contract, args, artifact, args.inference_mode == "async"
    ) as results:
        for index, (sample, (_source, outputs, geometry)) in enumerate(
            zip(samples, results, strict=True), start=1
        ):
            records.extend(
                postprocessing.prediction_records(
                    outputs, customer_contract, geometry, args, sample["image_id"]
                )
            )
            if index % 1000 == 0 or index == len(samples):
                print(f"{index}/{len(samples)} samples", file=sys.stderr, flush=True)
    predictions = {
        "task": evaluation["task"],
        "image_ids": [sample["image_id"] for sample in samples],
        "records": records,
    }
    write_json(out, predictions)
    try:
        result = dataset_command(
            dataset_code,
            "score",
            "--data-path",
            str(data_path),
            "--artifact",
            str(out),
        )
    except PackageError as error:
        if args.limit is None:
            raise
        result = {"not_scored": f"the evaluator needs the full set: {error}"}
    # The MXQ runs on the NPU are compared with the recorded NPU result, the ONNX runs on
    # the CPU with the recorded float result.
    recorded = evaluation if args.framework == "mxq" else evaluation.get("onnx")
    summary: dict[str, Any] = {
        "model": customer_contract["model"],
        "framework": args.framework,
        "dataset": evaluation["dataset"],
        "predictions": str(out),
        "result": result,
    }
    if recorded is not None:
        summary["expected"] = {
            "primary": recorded["primary"],
            "secondary": recorded.get("secondary", {}),
            "sample_count": recorded["sample_count"],
        }
    if args.limit is None and recorded is not None:
        summary["difference"] = compare(result, recorded)
    else:
        summary["note"] = (
            f"--limit {args.limit}: not comparable with the expected result"
        )
    write_json(out.with_name(RESULT_NAME), summary)
    print(json.dumps(summary, indent=2))
    if args.limit is None and recorded is not None:
        check_sample_count(result, recorded, evaluation["dataset"])


if __name__ == "__main__":
    exit_with_package_error(main)
