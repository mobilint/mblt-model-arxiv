#!/usr/bin/env python3
"""Run one image through the ONNX model (CPU) or the compiled MXQ model (NPU).

Input: --source, an image file. Output: artifacts/inference/<image>.outputs.npz with the raw
model outputs and <image>.summary.json with the task-specific interpretation (see the
task options below). Prerequisites: onnxruntime for --framework onnx; the Mobilint NPU
SDK with qbruntime and an attached NPU for --framework mxq.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import postprocessing
import preprocessing
from package_common import (
    ARTIFACTS_DIR,
    PackageError,
    artifact_path,
    exit_with_package_error,
    import_optional,
    load_contract,
    require_file,
    write_json,
)

CORE_MODES = ("single", "multi", "global4", "global8")


def add_runtime_arguments(argument_parser: Any, customer_contract: dict) -> None:
    """The model file and NPU options shared by inference.py and evaluate.py."""
    argument_parser.add_argument(
        "--framework",
        choices=("mxq", "onnx"),
        default="mxq",
        help="run the compiled MXQ on the NPU or the ONNX model on the CPU (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="model file (default: artifacts/<model>.<framework>)",
    )
    argument_parser.add_argument(
        "--target-device",
        choices=("aries-rb", "regulus-ra", "regulus-rb"),
        default=customer_contract["compile"]["target_device"],
        help="NPU family the MXQ was compiled for (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--dev-no", type=int, default=0, help="NPU device index (default: 0)"
    )
    argument_parser.add_argument(
        "--core-mode",
        choices=CORE_MODES,
        default="single",
        help="NPU core allocation for aries-rb (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--npu-timeout-ms",
        type=int,
        default=60_000,
        help="NPU timeout (default: 60000)",
    )


def parse_args(customer_contract: dict) -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument(
        "--source", type=Path, required=True, help="input image file"
    )
    add_runtime_arguments(argument_parser, customer_contract)
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "inference",
        help="where outputs are written (default: %(default)s)",
    )
    postprocessing.add_arguments(argument_parser, customer_contract)
    return argument_parser.parse_args()


class OnnxRunner:
    """One ONNX Runtime session on the CPU, reused for every image."""

    def __init__(self, path: Path) -> None:
        ort = import_optional("onnxruntime", "install the packages in requirements.txt")
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.model_input = self.session.get_inputs()[0]

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        value = tensor
        if len(self.model_input.shape) == 4 and tensor.ndim == 3:
            value = (
                tensor.transpose(2, 0, 1)[None]
                if self.model_input.shape[1] in (1, 2, 3, 4)
                else tensor[None]
            )
        outputs = self.session.run(
            None,
            {self.model_input.name: np.ascontiguousarray(value, dtype=np.float32)},
        )
        return [np.asarray(item) for item in outputs]

    def close(self) -> None:
        self.session = None


class MxqRunner:
    """One MXQ model launched on the NPU, reused for every image until closed."""

    def __init__(
        self, path: Path, args: argparse.Namespace, *, async_pipeline: bool = False
    ) -> None:
        if args.npu_timeout_ms <= 0:
            raise PackageError("--npu-timeout-ms must be positive")
        os.environ["QBRUNTIME_NPU_TIMEOUT_MS"] = str(args.npu_timeout_ms)
        qbruntime = import_optional(
            "qbruntime",
            "MXQ inference needs the qbruntime package from the Mobilint NPU SDK and an "
            "attached NPU; use --framework onnx to check the model without one",
        )
        # The runtime logs to standard output by default, which would interleave with the
        # JSON summary; keep warnings and errors only. The summary is also written to a
        # file, so callers that need machine-readable output should read that file.
        if hasattr(qbruntime, "set_log_level") and hasattr(qbruntime, "LogLevel"):
            qbruntime.set_log_level(qbruntime.LogLevel.Warn)
        config = qbruntime.ModelConfig()
        if async_pipeline:
            config.set_async_pipeline_enabled(True)
        if args.target_device == "aries-rb":
            cluster, core, core_id = qbruntime.Cluster, qbruntime.Core, qbruntime.CoreId
            if args.core_mode == "single":
                config.set_single_core_mode(
                    None,
                    [
                        core_id(cl, co)
                        for cl in (cluster.Cluster0, cluster.Cluster1)
                        for co in (core.Core0, core.Core1, core.Core2, core.Core3)
                    ],
                )
            elif args.core_mode == "multi":
                config.set_multi_core_mode([cluster.Cluster0, cluster.Cluster1])
            elif args.core_mode == "global4":
                config.set_global4_core_mode([cluster.Cluster0, cluster.Cluster1])
            else:
                config.set_global8_core_mode()
        elif args.core_mode != "single":
            raise PackageError("Regulus devices support only --core-mode single")
        try:
            self.model = qbruntime.Model(str(path), config)
            self.model.launch(qbruntime.Accelerator(args.dev_no))
        except Exception as error:
            raise PackageError(
                f"could not open NPU device {args.dev_no} with {path.name}: {error}. "
                "Check that the NPU driver is loaded and the device index is correct"
            ) from error

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        result = self.model.infer(np.ascontiguousarray(tensor, dtype=np.float32))
        values = result if isinstance(result, (tuple, list)) else [result]
        return [np.asarray(item) for item in values]

    def close(self) -> None:
        self.model.dispose()


def open_runner(
    framework: str, path: Path, args: argparse.Namespace
) -> OnnxRunner | MxqRunner:
    return MxqRunner(path, args) if framework == "mxq" else OnnxRunner(path)


def main() -> None:
    customer_contract = load_contract()
    args = parse_args(customer_contract)
    source = args.source.expanduser().resolve()
    require_file(source, "pass an existing image file with --source")
    artifact = (
        args.artifact.expanduser().resolve()
        if args.artifact is not None
        else artifact_path(customer_contract, args.framework)
    )
    if not artifact.is_file():
        step = "export.py" if args.framework == "onnx" else "compile.py"
        raise PackageError(f"model file not found: {artifact}; run {step} first")
    tensor, geometry = preprocessing.preprocess(source, customer_contract)
    runner = open_runner(args.framework, artifact, args)
    try:
        outputs = runner(tensor)
    finally:
        runner.close()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{source.stem}.outputs.npz"
    np.savez(raw_path, *outputs)
    summary: dict[str, Any] = {
        "model": customer_contract["model"],
        "framework": args.framework,
        "artifact": str(artifact),
        "source": str(source),
        "raw_outputs": str(raw_path),
        "output_shapes": [list(item.shape) for item in outputs],
        "geometry": geometry,
    }
    summary.update(
        postprocessing.summarize(
            outputs, customer_contract, geometry, args, output_dir, source
        )
    )
    write_json(output_dir / f"{source.stem}.summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    exit_with_package_error(main)
