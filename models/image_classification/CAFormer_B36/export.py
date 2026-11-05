#!/usr/bin/env python3
"""Export the model to ONNX.

Inputs: the upstream checkpoint named in model_contract.json (downloaded on first use unless
--weights points at a local file). Output: artifacts/<model>.onnx with the input signature
declared in the contract. Prerequisites: the export dependencies in requirements.txt
(torch and the upstream library) and network access or a populated checkpoint cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import exporter
from package_common import (
    PackageError,
    artifact_path,
    exit_with_package_error,
    import_optional,
    load_contract,
    refuse_overwrite,
    sha256_file,
)


def parse_args(customer_contract: dict) -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(customer_contract, "onnx"),
        help="destination ONNX file (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--opset",
        type=int,
        default=customer_contract["export"]["opset"],
        help="ONNX opset version (default: %(default)s, the validated value)",
    )
    argument_parser.add_argument(
        "--weights",
        default=None,
        help=(
            "local checkpoint path to use instead of downloading "
            f"{customer_contract['export']['weights']!r} from upstream"
        ),
    )
    argument_parser.add_argument(
        "--force", action="store_true", help="replace an existing ONNX file"
    )
    return argument_parser.parse_args()


def check_signature(onnx_module, path: Path, customer_contract: dict) -> None:
    model = onnx_module.load(str(path))
    onnx_module.checker.check_model(model)
    declared = customer_contract["input"]
    inputs = [tensor for tensor in model.graph.input]
    if len(inputs) != 1 or inputs[0].name != declared["name"]:
        raise PackageError(
            f"exported ONNX input names {[t.name for t in inputs]} differ from the contract "
            f"input {declared['name']!r}"
        )
    dims = inputs[0].type.tensor_type.shape.dim
    actual = [dim.dim_value if dim.dim_value else None for dim in dims]
    expected = list(declared["shape"])
    if len(actual) != len(expected) or actual[1:] != expected[1:]:
        raise PackageError(
            f"exported ONNX input shape {actual} differs from the contract shape {expected}"
        )


def main() -> None:
    customer_contract = load_contract()
    args = parse_args(customer_contract)
    output = args.output.expanduser().resolve()
    refuse_overwrite(output, args.force)
    onnx_module = import_optional("onnx", "install the packages in requirements.txt")
    weights = args.weights or customer_contract["export"]["weights"]
    if args.weights is not None and not Path(args.weights).is_file():
        raise PackageError(f"--weights file not found: {args.weights}")
    output.parent.mkdir(parents=True, exist_ok=True)
    exporter.export_onnx(
        customer_contract, weights=weights, output=output, opset=args.opset
    )
    check_signature(onnx_module, output, customer_contract)
    print(
        json.dumps(
            {
                "model": customer_contract["model"],
                "onnx": str(output),
                "sha256": sha256_file(output),
                "opset": args.opset,
                "input": customer_contract["input"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    exit_with_package_error(main)
