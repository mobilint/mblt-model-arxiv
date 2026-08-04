import argparse
import os

import cv2
import numpy as np
from qbcompiler.calibration import make_calib_man

from tasks import TASKS

PAD_VALUE: int = 114
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSIFY_RESIZE: int = 256


def letterbox(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return an RGB HWC float32 image letterboxed to (height, width) with pad 114."""
    source_height, source_width = image.shape[:2]
    scale = min(height / source_height, width / source_width)
    resized = int(round(source_width * scale)), int(round(source_height * scale))
    if (source_width, source_height) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)

    pad_height, pad_width = height - resized[1], width - resized[0]
    top, bottom = int(round(pad_height / 2 - 0.1)), int(round(pad_height / 2 + 0.1))
    left, right = int(round(pad_width / 2 - 0.1)), int(round(pad_width / 2 + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return (image / 255).astype(np.float32)


def center_crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize to 256x256, center-crop, then apply ImageNet normalization.

    Mirrors the resize/centerCrop/normalize pipeline the _uint8 configs declare, so both
    calibration paths feed the model identical tensors.
    """
    image = cv2.resize(image, (CLASSIFY_RESIZE, CLASSIFY_RESIZE), interpolation=cv2.INTER_LINEAR)
    top, left = (image.shape[0] - height) // 2, (image.shape[1] - width) // 2
    image = image[top : top + height, left : left + width]
    return ((image / 255).astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD


def make_preprocess(task: str):
    """Build the per-task pre_ftn consumed by make_calib_man."""
    _, height, width = TASKS[task]
    transform = center_crop if task == "classify" else letterbox

    def pre_ftn(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), height, width)

    return pre_ftn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO26 calibration tensors from the repo-root raw images. "
        "Only the float32 configs need this; uint8 configs calibrate on raw images directly."
    )
    parser.add_argument("--task", required=True, choices=list(TASKS), help="which task's calibset to build")
    parser.add_argument("--npy-path", help="output tensor dir (default: ./calibset_{task})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = TASKS[args.task][0]
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"No raw images at {raw_dir}; run the repo-root download script first")

    npy_path = args.npy_path or f"./calibset_{args.task}"
    make_calib_man(
        pre_ftn=make_preprocess(args.task),
        data_dir=raw_dir,
        save_dir=os.path.dirname(npy_path) or ".",
        save_name=os.path.basename(npy_path),
        remove_npy=True,  # clean the destination before writing new npy files
    )
    print(f"Done: {args.task} calibset from {raw_dir} in {os.path.abspath(npy_path)}")


if __name__ == "__main__":
    main()
