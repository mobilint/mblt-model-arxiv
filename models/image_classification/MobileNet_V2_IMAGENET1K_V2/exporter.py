"""Export a TorchVision model, named by its weight enum, to ONNX."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from package_common import PackageError, import_optional


def _load_model(weights: str, declared: str) -> Any:
    torch = import_optional("torch", "install the packages in requirements.txt")
    torchvision = import_optional(
        "torchvision", "install the packages in requirements.txt"
    )
    models = torchvision.models
    weight_enum = models.get_weight(declared)
    constructor_name = weight_enum.__class__.__name__.removesuffix("_Weights").lower()
    constructor = getattr(models, constructor_name)
    if weights == declared:
        try:
            return constructor(weights=weight_enum)
        except Exception as error:  # network or cache failure
            raise PackageError(
                f"could not obtain the checkpoint for {declared}: {error}. Provide network "
                "access, a populated TORCH_HOME cache, or pass --weights <local .pth file>"
            ) from error
    model = constructor(weights=None)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    return model


def export_onnx(
    customer_contract: dict[str, Any], *, weights: str, output: Path, opset: int
) -> None:
    torch = import_optional("torch", "install the packages in requirements.txt")
    model = _load_model(weights, customer_contract["export"]["weights"]).cpu().eval()
    _, channels, height, width = customer_contract["input"]["shape"]
    example = torch.zeros(1, channels, height, width, dtype=torch.float32)
    input_name = customer_contract["input"]["name"]
    output_names = [item["name"] for item in customer_contract["outputs"]]
    dynamic_axes = {input_name: {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    export_kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        (example,),
        str(output),
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        **export_kwargs,
    )
