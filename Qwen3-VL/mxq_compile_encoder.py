import argparse
from pathlib import Path

import torch
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict_new.parser.backend.torch.input_capture import capture_forward_inputs
from qbcompiler.model_dict_new.parser.patcher.models.hf_models import qwen3vl
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

MODEL_IDS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]
TARGET_DEVICES = ["aries-rb", "regulus-rb"]
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"


def load_model_and_processor(model_name):
    """Load a Qwen3-VL model (float32) and processor, patched for compilation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get Mobilint's patched subclass
    wrapper_cls = qwen3vl.ensure_qwen3vl_classes_loaded()
    model = wrapper_cls.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # deepstack merge -> plain addition
    language_model = model.model.language_model
    language_model._deepstack_process = qwen3vl.patched_deepstack_process.__get__(
        language_model, type(language_model)
    )

    # the compile unit: decoder = language_model + lm_head
    model.projection = qwen3vl.Projection(language_model, model.lm_head)

    # vision features as one stacked tensor instead of List[Tensor]
    model.model.get_image_feature_class = qwen3vl.Qwen3VLModel_get_image_feature(model.model)
    model.model.get_image_features = qwen3vl.Qwen3VL_get_image_features.__get__(
        model.model, type(model.model)
    )
    return model, processor


def prepare_inputs(processor, messages, model_device, image_size=None):
    """Build model.generate() inputs from chat messages; image_size optionally resizes."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size)
    if image_size is not None and image_inputs:
        image_inputs = [image_inputs[0].resize(image_size)]
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model_device, dtype=torch.float32)
    inputs.pop("mm_token_type_ids", None)
    return inputs


def build_feed_dict(model, processor, image_size):
    """Capture real vision-tower inputs from one generate step. Returns (feed_dict, grid_thw)."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_URL},
        {"type": "text", "text": "Describe this image in detail."},
    ]}]
    inputs = prepare_inputs(processor, messages, model.device, tuple(image_size))

    with capture_forward_inputs(model.visual, to_cpu=False) as captured:
        model.generate(**inputs, max_new_tokens=1, do_sample=False)

    # Qwen3VLVisionModel.forward(hidden_states, grid_thw): pixel_values arrives positionally
    pixel_values, grid_thw = captured["hidden_states"], captured["grid_thw"]

    # hf patch layout -> npu image layout; the compiled tower takes a single `images` tensor
    images = qwen3vl.repreprocess_pixel_values(pixel_values, grid_thw[0])
    return {"images": images}, grid_thw


def build_compilable_model(model, grid_thw):
    """Bake the grid-dependent pos/rope buffers into the vision tower so the compiled
    encoder needs no runtime grid_thw, and return the compilable module."""
    vision_model = qwen3vl.VisionModelForQwen3VL(model)
    vision_model.set_grid_thw(grid_thw.to(model.device))
    return vision_model.to(model.device).eval()


def get_device_inference_scheme(target_device):
    # REGULUS supports only the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL encoder: HF -> MBLT -> MXQ (consumes decoder SpinR1)")
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--target-device", required=True, choices=TARGET_DEVICES)
    parser.add_argument("--config", required=True, help="quantization config path, e.g. ./configs/Qwen3-VL-2B-Instruct_encoder.yaml")
    parser.add_argument("--mblt-path", help="output .mblt path (default: ./mblt/{name}_encoder.mblt)")
    parser.add_argument("--out-dir", default="./mxq", help="bundle root; outputs go to {out-dir}/{config-name}/")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.model.split("/")[-1]
    mblt_path = args.mblt_path or f"./mblt/{name}_encoder.mblt"
    calib_data_path = f"./calibset_{name}/vision/npy_files.txt"
    config_name = Path(args.config).stem
    bundle_dir = Path(args.out_dir) / config_name
    Path(mblt_path).parent.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    device = "gpu" if torch.cuda.is_available() else "cpu"

    print(f"[1/2] build encoder mblt -> {mblt_path}")
    model, processor = load_model_and_processor(args.model)
    feed_dict, grid_thw = build_feed_dict(model, processor, args.image_size)
    vision_model = build_compilable_model(model, grid_thw)
    mblt_compile(
        model=vision_model,
        mblt_save_path=mblt_path,
        target_device=args.target_device,
        backend="torch",
        device=device,
        feed_dict=feed_dict,
        cpu_offload=False,  # -> largest_supported_only=True
    )

    # SpinR1 comes from the decoder MXQ (head_out_ch_rotation in the config); run the decoder first.
    mxq_path = bundle_dir / f"{config_name}.mxq"
    print(f"[2/2] mxq_compile -> {mxq_path} (config: {config_name})")
    mxq_compile(
        model=mblt_path,
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=calib_data_path,
        device=device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        compile_config=args.config,
    )
    print(f"Done: {mxq_path}")


if __name__ == "__main__":
    main()
