"""Per-task calibration inputs and the variant -> task mapping.

Imported by both prepare_calibset.py and mxq_compile.py so the raw-data paths and
input resolutions live in exactly one place. A divergence here would silently
calibrate a model against the wrong dataset.
"""

# task -> (raw image dir written by the repo-root download scripts, height, width)
TASKS: dict[str, tuple[str, int, int]] = {
    "detect": ("../coco_files", 640, 640),
    "classify": ("../imagenet_files", 224, 224),
    "obb": ("../dota_files", 1024, 1024),
    "semantic": ("../cityscapes_files", 1024, 2048),
    "semantic-ade20k": ("../ade20k_files", 640, 640),
    "depth": ("../depth_files", 768, 768),
}

# variant suffix -> task. base/distill/seg/pose share COCO at 640.
VARIANT_TASK: dict[str, str] = {
    "": "detect",
    "-distill": "detect",
    "-seg": "detect",
    "-pose": "detect",
    "-cls": "classify",
    "-obb": "obb",
    "-sem": "semantic",
    "-sem-ade20k": "semantic-ade20k",
    "-depth": "depth",
}

MODEL_PREFIX: str = "yolo26"
SIZES: tuple[str, ...] = ("n", "s", "m", "l", "x")


def task_of(model: str) -> str:
    """Derive the calibration task from a model name such as yolo26m-obb."""
    if not model.startswith(MODEL_PREFIX) or len(model) <= len(MODEL_PREFIX):
        raise ValueError(f"Not a YOLO26 model name: {model!r}")
    size, suffix = model[len(MODEL_PREFIX)], model[len(MODEL_PREFIX) + 1 :]
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r} in {model!r}; expected one of {SIZES}")
    if suffix not in VARIANT_TASK:
        raise ValueError(f"Unknown YOLO26 variant {suffix!r} in {model!r}")
    return VARIANT_TASK[suffix]


def raw_dir(task: str) -> str:
    """Root download directory for task; uint8 configs calibrate straight from it."""
    return TASKS[task][0]


def tensor_dir(task: str) -> str:
    """Preprocessed .npy directory for task; float32 configs need prepare_calibset.py first."""
    return f"./calibset_{task}"


def model_names() -> list[str]:
    """Every YOLO26 model name, 9 variants x 5 sizes."""
    return [f"{MODEL_PREFIX}{size}{suffix}" for suffix in VARIANT_TASK for size in SIZES]
