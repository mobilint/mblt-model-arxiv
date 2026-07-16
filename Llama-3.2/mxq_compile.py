"""Compile Llama-3.2: mblt_compile -> mxq_compile (two-step).

Requires transformers==4.47.1 (the hf Llama parser needs LlamaSdpaAttention, removed
in transformers 4.48). qbcompiler's Llama parser auto-capture (_make_feed_dict, which
runs model.generate()) is broken there, so we replace it with a minimal prefill
feed_dict. Calibration data is passed as a single manifest .json (a list of dirs is
read as one-per-model and fails the count check).
"""

import importlib.util
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
import qbcompiler.model_dict.parser.backend.hf.transformers.models.llama as _llama_parser
from qbcompiler import mblt_compile, mxq_compile
from transformers import AutoModelForCausalLM

SEQ_LEN = 52  # prefill sequence length for tracing

MODEL_IDS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def _prefill_feed_dict(model, target_sg_state=0):
    """Minimal prefill inputs (empty KV cache); replaces the parser's broken _make_feed_dict."""
    return {
        "input_ids": torch.randint(0, 1000, (1, SEQ_LEN), dtype=torch.long),
        "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.long),
        "position_ids": torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0),
        "cache_position": torch.arange(SEQ_LEN, dtype=torch.long),
        "past_key_values": {"key_cache": [], "value_cache": [], "_seen_tokens": 0},
    }


_llama_parser._make_feed_dict = _prefill_feed_dict


def get_device_inference_scheme(target_device):
    # REGULUS only supports the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported")


def load_configs(config_path):
    """Import a quantization config .py by path and return its mxq_compile config kwargs."""
    spec = importlib.util.spec_from_file_location("mxq_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_configs()


def rotate_embeddings(model, rotation_path, save_path):
    # rotated input embedding = emb @ R1 rotation; global_rotation.pth is a TorchScript archive
    emb = model.get_input_embeddings().weight.detach().cpu()
    rot = torch.jit.load(rotation_path, map_location="cpu").state_dict()["0"]
    torch.save((emb.double() @ rot.double()).float(), save_path)


def bundle_spin_weights(model, name, config_name, bundle_dir):
    src = next((Path("spinWeight") / k for k in (name, config_name)
                if (Path("spinWeight") / k).is_dir()), None)
    if src is None:
        print("spinWeight not generated; skip bundling")
        return

    dst = bundle_dir / "spinWeight"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))

    # R1 rotates the hidden space, so the host-side input embedding is rotated to match.
    rotation = dst / "R1" / "global_rotation.pth"
    if not rotation.is_file():
        print("no R1 rotation; skip embedding rotation")
        return

    emb_path = bundle_dir / f"{config_name}_embeddings_rotated.pth"
    print(f"rotate embeddings -> {emb_path}")
    rotate_embeddings(model, rotation, emb_path)


def parse_args():
    parser = ArgumentParser(description="Compile Llama: mblt_compile -> mxq_compile")
    parser.add_argument("--model", required=True, choices=MODEL_IDS, help="model id to compile")
    parser.add_argument("--target-device", required=True, choices=["regulus-rb", "aries-rb"])
    parser.add_argument("--config", required=True, help="quantization config path, e.g. ./configs/Llama-3.2-1B-Instruct_w8.py")
    parser.add_argument("--mblt-path", help="output .mblt path (default: ./mblt/{model}.mblt)")
    parser.add_argument("--out-dir", default="./mxq", help="bundle root; outputs go to {out-dir}/{config-name}/")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.model.split("/")[-1]
    mblt_path = args.mblt_path or f"./mblt/{name}.mblt"
    calib_data_path = f"./calibset_{name}/calib_files.json"
    config_name = Path(args.config).stem  # e.g. Llama-3.2-1B-Instruct_w8; names the mxq/spinWeight bundle
    bundle_dir = Path(args.out_dir) / config_name  # ./mxq/{config}/ : mxq + spinWeight + rotated embedding
    os.makedirs(os.path.dirname(mblt_path), exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    device = "gpu" if torch.cuda.is_available() else "cpu"

    print(f"[1/2] mblt_compile -> {mblt_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model).eval().cpu()
    mblt_compile(
        model=model,
        mblt_save_path=mblt_path,
        backend="hf",
        target_device=args.target_device,
        device="cpu",
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
        **load_configs(args.config),
    )

    bundle_spin_weights(model, name, config_name, bundle_dir)
    print(f"Done: {bundle_dir}")


if __name__ == "__main__":
    main()
