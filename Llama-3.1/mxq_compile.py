import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor

MODEL_IDS = [
    "meta-llama/Llama-3.1-8B-Instruct",
]


def load_model_and_tokenizer(model_id):
    """Load the model on CPU in float32 (tracing runs on CPU) plus its tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_id).eval().cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def build_feed_dict(model, tokenizer, device):
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me a short introduction to large language model."},
    ]
    inputs = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)  # some chat templates emit it; generate() rejects it
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model, max_call_limit=1, inputs_container=container):
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
    captured = container.captured_kwargs[-1]  # capture forward inputs

    feed_dict = {}
    for name, value in captured.items():
        if not isinstance(value, torch.Tensor):  # if not Tensor, store name and value
            feed_dict[name] = value
            continue
        tensor = wrap_tensor(name, value.to(device))  # if Tensor, wrapping `PT_Activation`
        if name == "cache_position":  # 1-D: (seq,)
            tensor.src_shape[0].set_dynamic(True)
        elif name in ("input_ids", "attention_mask", "position_ids"):  # 2-D: (1, seq)
            tensor.src_shape[-1].set_dynamic(True)
        feed_dict[name] = tensor

    set_attention_mask(feed_dict["attention_mask"], "causal_mask")  # treat mask as causal
    feed_dict.setdefault("past_key_values", DynamicCache())  # trace with a KV cache
    feed_dict["logits_to_keep"] = 1  # keep only the last token
    return feed_dict


def get_device_inference_scheme(target_device):
    # REGULUS supports only the single scheme; ARIES supports all schemes in one model.
    # Recipe default is batch_size=1 -> "all"; use "single" when compiling for batch > 1.
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported")


def rotate_embeddings(model, rotation_path, save_path):
    # rotated input embedding = emb @ R1 rotation; global_rotation.pth is a TorchScript archive
    emb = model.get_input_embeddings().weight.detach().cpu()
    rot = torch.jit.load(rotation_path, map_location="cpu").state_dict()["0"]
    torch.save((emb.double() @ rot.double()).float(), save_path)


def bundle_spin_weights(model, name, config_name, bundle_dir):
    dst = bundle_dir / "spinWeight"
    src = next((Path("spinWeight") / k for k in (name, config_name)
                if (Path("spinWeight") / k).is_dir()), None)
    if src is not None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    elif not dst.is_dir():
        print("spinWeight not generated; skip bundling")
        return

    # R1 rotates the hidden space, so the host-side input embedding is rotated to match.
    rotation = dst / "R1" / "global_rotation.pth"
    if not rotation.is_file():
        print("no R1 rotation; skip embedding rotation")
        return

    emb_path = bundle_dir / f"{config_name}_embeddings_rotated.pth"
    print(f"rotate embeddings -> {emb_path}")
    rotate_embeddings(model, rotation, emb_path)


def parse_args():
    parser = ArgumentParser(description="Llama-3.1 compile: mblt_compile (torch/fx) -> mxq_compile")
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--target-device", required=True, choices=["regulus-rb", "aries-rb"])
    parser.add_argument("--config", required=True, help="quantization config path, e.g. ./configs/Llama-3.1-8B-Instruct_w8.yaml")
    parser.add_argument("--mblt-path", help="output .mblt path (default: ./mblt/{model}.mblt)")
    parser.add_argument("--out-dir", default="./mxq", help="bundle root; outputs go to {out-dir}/{config-name}/")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.model.split("/")[-1]
    mblt_path = args.mblt_path or f"./mblt/{name}.mblt"
    calib_data_path = f"./calibset_{name}/calib_files.json"  # single manifest (a dir list fails the count check)
    config_name = Path(args.config).stem  # names the mxq/spinWeight bundle
    bundle_dir = Path(args.out_dir) / config_name  # ./mxq/{config}/ : mxq + spinWeight + rotated embedding
    Path(mblt_path).parent.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    device = "gpu" if torch.cuda.is_available() else "cpu"

    print(f"[1/2] mblt_compile (torch) -> {mblt_path}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    feed_dict = build_feed_dict(model, tokenizer, device="cpu")
    mblt_compile(
        model=model,
        mblt_save_path=mblt_path,
        backend="torch",
        target_device=args.target_device,
        device="cpu",
        feed_dict=feed_dict,
        output_meta={"type": "list", "keys": [0]},  # in mblt compile process, only need logits(index 0)
    )

    mxq_path = bundle_dir / f"{config_name}.mxq"
    print(f"[2/2] mxq_compile -> {mxq_path} (config: {config_name})")
    mxq_compile(
        model=mblt_path,
        target_device=args.target_device,
        calib_data_path=calib_data_path,
        save_path=str(mxq_path),
        device=device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        compile_config=args.config,
    )

    bundle_spin_weights(model, name, config_name, bundle_dir)
    print(f"Done: {bundle_dir}")


if __name__ == "__main__":
    main()
