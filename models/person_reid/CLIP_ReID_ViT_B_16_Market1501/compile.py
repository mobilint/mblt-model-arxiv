#!/usr/bin/env python3
"""Compile the exported ONNX model to an MXQ device artifact with qbcompiler.

Inputs: artifacts/<model>.onnx (from export.py) and the calibration directory (from
prepare_calibration.py). Output: artifacts/<model>.mxq and a <model>.compile.json report
beside it. Prerequisite: a qbcompiler wheel
delivered with this package or supplied with the Mobilint SDK; a CUDA GPU
is used by default (--device cpu is slower). --preflight checks the compiler and the
compile configuration without compiling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from package_common import (
    COMPILE_CONFIG_FILE,
    PackageError,
    artifact_path,
    calibration_output,
    exit_with_package_error,
    import_optional,
    load_calibration_spec,
    load_contract,
    refuse_overwrite,
    require_file,
    sha256_file,
)

TARGET_DEVICES = ("aries-rb", "regulus-ra", "regulus-rb")


def parse_args(customer_contract: dict, calibration_spec: dict) -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument(
        "--onnx",
        type=Path,
        default=artifact_path(customer_contract, "onnx"),
        help="exported ONNX model (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=calibration_output(calibration_spec),
        help="calibration images from prepare_calibration.py (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(customer_contract, "mxq"),
        help="destination MXQ file (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--mblt-output",
        type=Path,
        default=None,
        help="also write the MBLT intermediate artifact to this path",
    )
    argument_parser.add_argument(
        "--target-device",
        choices=TARGET_DEVICES,
        default=customer_contract["compile"]["target_device"],
        help="NPU target (default: %(default)s, the validated profile)",
    )
    argument_parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="where calibration and quantization run (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--inference-scheme",
        choices=("single", "all"),
        default=None,
        help="plans to compile: single for single-core inference, all for all supported "
        "plans (default: all on Aries, single on Regulus)",
    )
    argument_parser.add_argument(
        "--preflight",
        action="store_true",
        help="check qbcompiler and the compile configuration, then exit without compiling",
    )
    argument_parser.add_argument(
        "--force", action="store_true", help="replace an existing MXQ file"
    )
    return argument_parser.parse_args()


def inference_scheme(target_device: str) -> str:
    return "single" if target_device.startswith("regulus") else "all"


def load_compile_config(qbcompiler) -> object:
    require_file(COMPILE_CONFIG_FILE, "unpack the complete package")
    try:
        return qbcompiler.CompileConfig.from_file(str(COMPILE_CONFIG_FILE))
    except Exception as error:
        raise PackageError(
            f"the installed qbcompiler cannot load compile_config.json: {error}. Install the "
            "qbcompiler wheel delivered with this package or your Mobilint SDK"
        ) from error


def count_images(directory: Path, suffixes: list[str]) -> int:
    allowed = {suffix.lower() for suffix in suffixes}
    return sum(
        path.is_file() and path.suffix.lower() in allowed
        for path in directory.iterdir()
    )


def main() -> None:
    customer_contract = load_contract()
    calibration_spec = load_calibration_spec()
    args = parse_args(customer_contract, calibration_spec)
    qbcompiler = import_optional(
        "qbcompiler",
        "install the qbcompiler wheel supplied with your Mobilint SDK (it is not on PyPI)",
    )
    if not args.preflight:
        require_file(args.onnx.expanduser().resolve(), "run export.py first")
    selected_inference_scheme = args.inference_scheme or inference_scheme(
        args.target_device
    )
    config = load_compile_config(qbcompiler)
    version = getattr(qbcompiler, "__version__", "unknown")
    report = {
        "qbcompiler_version": version,
        "target_device": args.target_device,
        "inference_scheme": selected_inference_scheme,
        "compile_device": args.device,
        "config_version": customer_contract["compile"].get("config_version"),
    }
    if args.preflight:
        report["preflight"] = "ok"
        print(json.dumps(report, indent=2))
        return
    onnx = require_file(args.onnx.expanduser().resolve(), "run export.py first")
    calibration_dir = args.calibration_dir.expanduser().resolve()
    if not calibration_dir.is_dir():
        raise PackageError(
            f"{calibration_dir} does not exist; run prepare_calibration.py first"
        )
    found = count_images(calibration_dir, calibration_spec["layout"]["image_suffixes"])
    expected = int(calibration_spec["expected_sample_count"])
    if found != expected:
        raise PackageError(
            f"{calibration_dir} holds {found} images, the calibration profile expects "
            f"{expected}; rerun prepare_calibration.py"
        )
    output = args.output.expanduser().resolve()
    refuse_overwrite(output, args.force)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mblt_output is not None:
        mblt = args.mblt_output.expanduser().resolve()
        refuse_overwrite(mblt, args.force)
        mblt.parent.mkdir(parents=True, exist_ok=True)
        qbcompiler.mblt_compile(
            model=str(onnx),
            mblt_save_path=str(mblt),
            target_device=args.target_device,
            backend="onnx",
            device="cpu",
        )
        report["mblt"] = str(mblt)
        report["mblt_sha256"] = sha256_file(mblt)
    started = time.perf_counter()
    qbcompiler.mxq_compile(
        model=str(onnx),
        calib_data_path=str(calibration_dir),
        save_path=str(output),
        backend="onnx",
        target_device=args.target_device,
        inference_scheme=selected_inference_scheme,
        compile_config=config,
        device=args.device,
    )
    report["mxq"] = str(output)
    report["mxq_sha256"] = sha256_file(output)
    report["compile_seconds"] = round(time.perf_counter() - started, 1)
    # qbcompiler logs to standard output, so the report is also kept beside the artifact.
    report_path = output.with_suffix(".compile.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["report"] = str(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    exit_with_package_error(main)
