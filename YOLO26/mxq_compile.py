import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from qbcompiler import mblt_compile, mxq_compile

from tasks import raw_dir, task_of, tensor_dir

TARGET_DEVICES = ["regulus-rb", "aries-rb"]
MODES = ("uint8", "float32")


def get_compile_device() -> str:
    """Use CUDA when available and otherwise compile on the CPU."""
    return "gpu" if torch.cuda.is_available() else "cpu"


def get_device_inference_scheme(target_device: str) -> str:
    # REGULUS supports only the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported")


def resolve_onnx_path(model: str) -> str:
    """Fetch the model's ONNX from its mobilint repo; the HF cache handles reuse."""
    repo_id = "mobilint/" + model.replace("yolo26", "YOLO26", 1)
    return hf_hub_download(repo_id=repo_id, filename=f"{model}.onnx")


def resolve_calib_data_path(mode: str, task: str) -> str:
    """uint8 configs fuse preprocessing, so they calibrate on the repo-root raw images.
    float32 configs expect the already-preprocessed tensors from prepare_calibset.py."""
    if mode == "uint8":
        path, hint = raw_dir(task), f"run the repo-root download script for {task} first"
    else:
        path, hint = tensor_dir(task), f"run: python prepare_calibset.py --task {task}"
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No calibration data at {path}; {hint}")
    return path


def parse_args():
    parser = ArgumentParser(description="YOLO26 compile: ONNX -> MBLT + MXQ")
    parser.add_argument("--model", required=True, help="model name, e.g. yolo26m-obb")
    parser.add_argument("--target-device", required=True, choices=TARGET_DEVICES)
    parser.add_argument(
        "--config",
        required=True,
        help="quantization config; the _uint8 / _float32 suffix selects the calibration input",
    )
    parser.add_argument("--onnx-path", help="skip the download and use this ONNX instead")
    parser.add_argument("--out-dir", default="./mxq", help="directory for the .mblt and .mxq outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = Path(args.config).stem.rsplit("_", 1)[-1]
    if mode not in MODES:
        raise ValueError(f"config name must end with _uint8 or _float32: {args.config}")

    task = task_of(args.model)  # the model name fully determines the task
    calib_data_path = resolve_calib_data_path(mode, task)
    onnx_path = args.onnx_path or resolve_onnx_path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mblt_path = out_dir / f"{args.model}.{args.target_device}.mblt"
    print(f"[1/2] mblt_compile -> {mblt_path}")
    mblt_compile(
        model=onnx_path,
        mblt_save_path=str(mblt_path),
        target_device=args.target_device,
        backend="onnx",
        device="cpu",
        yolo_decode_include=False,  # decode stays on the host; not a CompileConfig field
    )

    mxq_path = out_dir / f"{args.model}.{args.target_device}.{mode}.mxq"
    print(f"[2/2] mxq_compile -> {mxq_path} (task: {task}, calib: {calib_data_path})")
    mxq_compile(
        model=onnx_path,
        calib_data_path=calib_data_path,
        save_path=str(mxq_path),
        backend="onnx",
        target_device=args.target_device,
        device=get_compile_device(),
        inference_scheme=get_device_inference_scheme(args.target_device),
        image_channels=3,  # convert grayscale calibration images to RGB
        yolo_decode_include=False,  # decode stays on the host; not a CompileConfig field
        compile_config=args.config,
    )
    print(f"Done: {mxq_path}")


if __name__ == "__main__":
    main()
