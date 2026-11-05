"""Export an Ultralytics checkpoint to ONNX by tracing it through TorchScript."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from package_common import ARTIFACTS_DIR, PackageError, import_optional


def _output_names(result: Any) -> list[str]:
    if isinstance(result, (tuple, list)):
        return [f"output{index}" for index in range(len(result))]
    return ["output0"]


def export_onnx(
    customer_contract: dict[str, Any], *, weights: str, output: Path, opset: int
) -> None:
    torch = import_optional("torch", "install the packages in requirements.txt")
    ultralytics = import_optional(
        "ultralytics",
        "install the packages in requirements.txt (ultralytics is AGPL-3.0)",
    )
    checkpoint = Path(weights)
    if not checkpoint.is_absolute() and not checkpoint.is_file():
        # Ultralytics downloads a bare checkpoint name into the working directory; keep
        # that download beside the other artifacts instead of the caller's cwd.
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = ARTIFACTS_DIR / checkpoint.name
    try:
        yolo = ultralytics.YOLO(str(checkpoint))
    except Exception as error:
        raise PackageError(
            f"could not load the checkpoint {weights!r}: {error}. Provide network access "
            "for the upstream download or pass --weights <local .pt file>"
        ) from error
    _, channels, height, width = customer_contract["input"]["shape"]
    torchscript_path = Path(
        yolo.export(
            format="torchscript", imgsz=(height, width), device="cpu", nms=False
        )
    )
    try:
        model = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
        _, channels, height, width = customer_contract["input"]["shape"]
        example = torch.zeros(1, channels, height, width, dtype=torch.float32)
        with torch.no_grad():
            result = model(example)
        output_names = _output_names(result)
        input_name = customer_contract["input"]["name"]
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
    finally:
        if (
            torchscript_path.is_file()
            and torchscript_path.resolve() != output.resolve()
        ):
            torchscript_path.unlink()
