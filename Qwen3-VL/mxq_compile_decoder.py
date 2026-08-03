import argparse
import os
from pathlib import Path

import torch
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict_new.parser.backend.torch.input_capture import capture_forward_inputs
from qbcompiler.model_dict_new.parser.patcher.models.hf_models import qwen3vl
from qwen_vl_utils import process_vision_info
from safetensors.torch import save_file
from transformers import AutoProcessor

MODEL_IDS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]
TARGET_DEVICES = ["aries-rb", "regulus-rb"]
EMBED_TOKENS_KEY = "model.language_model.embed_tokens.weight"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"

# Axes carrying the dynamic sequence length.
DYNAMIC_AXES = {
    "inputs_embeds": [-2],
    "cache_position": [-1],
    "rope_deltas": [-1],
    "deepstack_visual_embeds": [-2],
}


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


def build_feed_dict(model, processor):
    """Capture real Projection inputs from one generate step. Returns (feed_dict, position_ids)."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_URL},
        {"type": "text", "text": "Describe this image in detail."},
    ]}]
    inputs = prepare_inputs(processor, messages, model.device, image_size=(224, 224))

    with capture_forward_inputs(model.projection, to_cpu=False) as feed_dict:
        model.generate(**inputs, max_new_tokens=1, do_sample=False)

    # captured masked (L, N_vis, D) -> full sequence (L, T, D); Projection splits it internally
    feed_dict["deepstack_visual_embeds"] = qwen3vl.build_full_visual_embeds(
        feed_dict["deepstack_visual_embeds"], feed_dict["visual_pos_masks"]
    )
    return dict(feed_dict), feed_dict["position_ids"]


def build_compilable_model(model, position_ids):
    """Bake mrope cos/sin into a cached RoPE buffer so the compiled decoder needs no
    runtime position_ids, and return the compilable Projection module."""
    language_model = model.projection.language_model
    language_model.rotary_emb = qwen3vl.CachedQwen3VLTextRotaryEmbedding(language_model.rotary_emb)
    language_model.rotary_emb.set_rope(position_ids)
    return model.projection


def get_device_inference_scheme(target_device):
    # REGULUS supports only the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported")


def rotate_embeddings(model, rotation_path, save_path):
    # embed_tokens is a host-side lookup, so pre-rotate it by R1 to match the quantized decoder.
    emb = model.model.language_model.embed_tokens.weight.detach().cpu()
    rot = torch.jit.load(rotation_path, map_location="cpu").state_dict()["0"]
    save_file({EMBED_TOKENS_KEY: (emb.double() @ rot.double()).float()}, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL decoder: HF -> MBLT -> MXQ (generates SpinR1)")
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--target-device", required=True, choices=TARGET_DEVICES)
    parser.add_argument("--config", required=True, help="quantization config path, e.g. ./configs/Qwen3-VL-2B-Instruct_decoder_w8.yaml")
    parser.add_argument("--mblt-path", help="output .mblt path (default: ./mblt/{name}_decoder.mblt)")
    parser.add_argument("--out-dir", default="./mxq", help="bundle root; outputs go to {out-dir}/{config-name}/")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.model.split("/")[-1]
    mblt_path = args.mblt_path or f"./mblt/{name}_decoder.mblt"
    calib_data_path = f"./calibset_{name}/language/npy_files.json"
    config_name = Path(args.config).stem
    bundle_dir = Path(args.out_dir) / config_name
    Path(mblt_path).parent.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    device = "gpu" if torch.cuda.is_available() else "cpu"

    print(f"[1/3] build decoder mblt -> {mblt_path}")
    model, processor = load_model_and_processor(args.model)
    feed_dict, position_ids = build_feed_dict(model, processor)
    projection = build_compilable_model(model, position_ids)
    mblt_compile(
        model=projection,
        mblt_save_path=mblt_path,
        target_device=args.target_device,
        backend="torch",
        device=device,
        feed_dict=feed_dict,
        dynamic_axes=DYNAMIC_AXES,
        cpu_offload=False,  # -> largest_supported_only=True
    )

    mxq_path = bundle_dir / f"{config_name}.mxq"
    print(f"[2/3] mxq_compile -> {mxq_path} (config: {config_name})")
    mxq_compile(
        model=mblt_path,
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=calib_data_path,
        device=device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        compile_config=args.config,
    )

    # SpinR1 (generated above) is keyed by the mblt stem; the encoder MXQ reuses it.
    rotation_path = f"./spinWeight/{name}_decoder/R1/global_rotation.pth"
    emb_path = bundle_dir / "model.safetensors"
    if os.path.isfile(rotation_path):
        print(f"[3/3] rotate embeddings -> {emb_path}")
        rotate_embeddings(model, rotation_path, str(emb_path))
    else:
        print(f"[3/3] no R1 rotation at {rotation_path}; skip embedding rotation")
    print(f"Done: {bundle_dir}  (SpinR1: ./spinWeight/{name}_decoder/R1/)")


if __name__ == "__main__":
    main()
