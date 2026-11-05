#!/usr/bin/env python3
"""Export the model to ONNX with its weights in external files beside the graph.

Inputs: the upstream checkpoint named in model_contract.json (downloaded on first use unless
--weights points at a local file). Output: artifacts/<model>.onnx with the input signature
declared in the contract, plus the external weight files it references in the same
directory. The graph exceeds the 2 GiB protobuf limit, so it is checked from its path and
the weights are never loaded inline. Prerequisites: the export dependencies in
requirements.txt (torch and the upstream library) and network access or a populated
checkpoint cache.
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
    # A path check streams the graph; a loaded ModelProto over 2 GiB cannot be serialized.
    try:
        onnx_module.checker.check_model(str(path))
    except onnx_module.checker.ValidationError as error:
        raise PackageError(f"exported ONNX {path} is invalid: {error}") from error
    model = onnx_module.load(str(path), load_external_data=False)
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


def _message_tensors(onnx_module, message):
    """Every TensorProto nested anywhere in an ONNX protobuf message.

    Walking every message field covers graphs, sparse tensors, node and function attributes,
    local functions and training graphs alike, including fields a later ONNX release adds.
    """
    if isinstance(message, onnx_module.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        for item in value if field.label == field.LABEL_REPEATED else (value,):
            yield from _message_tensors(onnx_module, item)


def external_weight_files(onnx_module, path: Path) -> dict[str, str]:
    """SHA-256 of every external weight file the model references, by relative location."""
    model = onnx_module.load(str(path), load_external_data=False)
    directory = path.parent.resolve()
    locations = set()
    for tensor in _message_tensors(onnx_module, model):
        if not onnx_module.external_data_helper.uses_external_data(tensor):
            continue
        entries = {entry.key: entry.value for entry in tensor.external_data}
        location = entries.get("location", "")
        target = (directory / location).resolve()
        if not location or target.parent != directory:
            raise PackageError(
                f"external weight {location!r} of {tensor.name!r} is not a file beside {path}"
            )
        if not target.is_file():
            raise PackageError(f"external weight file {target} is missing")
        locations.add(location)
    if not locations:
        raise PackageError(
            f"{path} keeps no weights in external files, but the contract declares "
            "export.weight_storage: external-data.v1"
        )
    return {
        location: sha256_file(directory / location) for location in sorted(locations)
    }


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
    # Reference checks first, so a missing or misplaced file is reported as such.
    external_weights = external_weight_files(onnx_module, output)
    check_signature(onnx_module, output, customer_contract)
    print(
        json.dumps(
            {
                "model": customer_contract["model"],
                "onnx": str(output),
                "sha256": sha256_file(output),
                "external_weights": external_weights,
                "opset": args.opset,
                "input": customer_contract["input"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    exit_with_package_error(main)
