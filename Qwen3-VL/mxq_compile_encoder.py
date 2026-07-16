import argparse
import importlib.util
import os
from pathlib import Path

import torch
from qbcompiler import mxq_compile
from qbcompiler.model_dict.common import DataFormat, LayerType
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    Projection,
    Qwen3VL_get_image_features,
    Qwen3VLForConditionalGenerationWrapper,
    Qwen3VLModel_get_image_feature,
    VisionModelForQwen3VL,
    patched_deepstack_process,
    repreprocess_pixel_values,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from qbcompiler.model_dict.parser.parser import ModelParser
from qbcompiler.model_dict.serialize import ChainedByteObj, SerializeMeta
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

MODEL_IDS = ["Qwen/Qwen3-VL-2B-Instruct"]
TARGET_DEVICES = ["regulus-ra", "aries-rb", "regulus-rb"]
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"


def load_model_and_processor(model_name):
    """Load a Qwen3-VL model (float32) and processor, patched for compilation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGenerationWrapper.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(model_name)

    language_model = model.model.language_model
    language_model._deepstack_process = patched_deepstack_process.__get__(
        language_model, type(language_model)
    )
    model.projection = Projection(language_model, model.lm_head)

    model.model.get_image_feature_class = Qwen3VLModel_get_image_feature(model.model)
    model.model.get_image_features = Qwen3VL_get_image_features.__get__(
        model.model, type(model.model)
    )
    return model, processor


def prepare_inputs(processor, messages, model_device, image_size=None):
    """Build model.generate() inputs from chat messages; image_size optionally resizes."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size
    )
    if image_size is not None and image_inputs:
        image_inputs = [image_inputs[0].resize(image_size)]
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model_device, dtype=torch.float32)
    inputs.pop("mm_token_type_ids", None)  # rejected by the wrapper's generate(); handled internally
    return inputs


def serialize_to_mblt(model_dict, weight_dict, output_path, ignore_weight=False):
    """Serialize a ModelDict/WeightDict pair to the MBLT binary and write it to disk."""
    barr = SerializeMeta().serialize(model_dict, weight_dict, ignore_weight=ignore_weight)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        if isinstance(barr, bytes):
            f.write(barr)
        elif isinstance(barr, ChainedByteObj):
            barr.write(f)
    print(f"Saved {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")


def capture_vision_inputs(model, inputs):
    """Capture pixel_values / image_grid_thw at the vision feature entry point."""
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model.model.get_image_feature_class, 1, container):
        model(**inputs)
    if not container.captured_kwargs:
        raise RuntimeError("Vision feature hook captured no inputs.")
    captured = container.captured_kwargs[0]
    return captured["pixel_values"], captured["image_grid_thw"]


def set_input_constants_nhwc(model_dict):
    # Vision input constants use the NHWC layout on the NPU.
    for subgraph in model_dict.subgraphs:
        for op in subgraph.operators:
            if op.layertype == LayerType.InputConstant:
                subgraph.activations[op.options.outputs[0]].dataformat = DataFormat.NHWC


def build_vision_feed_dict(model, processor, image_size):
    """Capture one sample image's vision inputs and wrap them as the parser feed_dict."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_URL},
        {"type": "text", "text": "Describe this image in detail."},
    ]}]
    inputs = prepare_inputs(processor, messages, model.device, tuple(image_size))

    pixel_values, grid_thw = capture_vision_inputs(model, inputs)
    images = repreprocess_pixel_values(pixel_values, grid_thw[0])

    vision_model = VisionModelForQwen3VL(model.model)
    vision_model.set_grid_thw(grid_thw.to(model.device))
    vision_model.to(model.device)

    feed_dict = {"images": wrap_tensor("images", images.to(model.device))}
    return vision_model, feed_dict


def tmp_mblt_compile(vision_model, feed_dict, target_device, mblt_path):
    """Local mblt_compile for the vision encoder: parse -> NHWC fixup -> serialize.

    qbcompiler's mblt_compile() serializes internally with no hook, so we replicate its
    shape here to insert set_input_constants_nhwc between parse and serialize.
    """
    parser = ModelParser(
        model=vision_model, backend="torch", target_device=target_device
    )
    parser.parse(feed_dict=feed_dict, save_subgraph_type=1, debug=False)
    model_dict, weight_dict = parser.get_md_wd(body_only=False)
    set_input_constants_nhwc(model_dict)
    serialize_to_mblt(model_dict, weight_dict, mblt_path)


def load_configs(config_path):
    spec = importlib.util.spec_from_file_location("mxq_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_configs()


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
    parser.add_argument("--config", required=True, help="config path, e.g. ./configs/Qwen3-VL-2B-Instruct_encoder.py")
    parser.add_argument("--mblt-path", help="output .mblt path (default: ./mblt/{name}_encoder.mblt)")
    parser.add_argument("--out-dir", default="./mxq", help="bundle root; outputs go to {out-dir}/{config-name}/")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.model.split("/")[-1]
    mblt_path = args.mblt_path or f"./mblt/{name}_encoder.mblt"
    os.makedirs(os.path.dirname(mblt_path), exist_ok=True)

    print(f"[1/2] build encoder mblt -> {mblt_path}")
    model, processor = load_model_and_processor(args.model)
    vision_model, feed_dict = build_vision_feed_dict(model, processor, args.image_size)
    tmp_mblt_compile(vision_model, feed_dict, args.target_device, mblt_path)

    # [2/2] mxq (consumes the decoder's SpinR1 via head_out_ch_rotation; run decoder first):
    # config_name = Path(args.config).stem
    # bundle_dir = Path(args.out_dir) / config_name
    # bundle_dir.mkdir(parents=True, exist_ok=True)
    # mxq_path = bundle_dir / f"{config_name}.mxq"
    # mxq_compile(
    #     model=mblt_path,
    #     target_device=args.target_device,
    #     save_path=str(mxq_path),
    #     calib_data_path=f"./calibset_{name}/vision/npy_files.txt",
    #     device="gpu" if torch.cuda.is_available() else "cpu",
    #     inference_scheme=get_device_inference_scheme(args.target_device),
    #     **load_configs(args.config),
    # )
    # print(f"Done: {mxq_path}")


if __name__ == "__main__":
    main()
