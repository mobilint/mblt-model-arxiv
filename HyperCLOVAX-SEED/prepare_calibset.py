import argparse
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT_DIR = "../wikipedia_files"     # raw text from download_wikipedia_data.py (run from this model dir)

# Selectable model ids. Sizes in a cluster share a tokenizer but have different
# embeddings, so their calibsets (inputs_embeds) differ even from identical text.
MODEL_IDS = [
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
]

# Per-language calibration quota (recipe distribution). Scaled by --ratio.
QUOTA = {
    "en": 50,
    "ko": 50,
}

MIN_SEQLEN = 512
MAX_SEQLEN = 2048


def load_embedding_layer(model_id: str, device: str) -> torch.nn.Embedding:
    """Load `model_id` and return its input-embedding layer (float32); the rest of the model is freed."""
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
    weight = model.get_input_embeddings().weight.detach().float()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.nn.Embedding.from_pretrained(weight).to(device)


def generate_for_model(
    tokenizer: AutoTokenizer,
    embedding_layer: torch.nn.Embedding,
    device: str,
    ratio: float,
    save_dir: str,
) -> int:
    """Embed raw text into calibration inputs_embeds .npy files under save_dir. Returns count saved."""
    total = 0
    for lang, base in tqdm(QUOTA.items(), desc=os.path.basename(save_dir), unit="lang"):
        target = round(base * ratio)
        out_dir = os.path.join(save_dir, lang)
        os.makedirs(out_dir, exist_ok=True)
        for stale in glob.glob(os.path.join(out_dir, "inputs_embeds_*.npy")):  # exact count on re-run
            os.remove(stale)

        saved = 0
        for path in sorted(glob.glob(os.path.join(TEXT_DIR, f"{lang}_*.txt"))):
            if saved >= target:
                break
            with open(path, encoding="utf-8") as f:
                text = f.read()

            token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze().to(device)
            if token_ids.ndim == 0:  # empty text -> scalar after squeeze
                continue

            inputs_embeds = embedding_layer(token_ids).unsqueeze(0)  # [1, seq_len, embed_dim]
            seq_len = inputs_embeds.shape[1]
            if seq_len < MIN_SEQLEN:
                continue
            if seq_len > MAX_SEQLEN:
                inputs_embeds = inputs_embeds[:, :MAX_SEQLEN, :]

            out_path = os.path.join(out_dir, f"inputs_embeds_{saved}.npy")
            np.save(out_path, inputs_embeds.detach().cpu().numpy())
            saved += 1
        total += saved
    return total


def write_calib_manifest(save_dir: str) -> str:
    """Write calib_files.json (info + calib paths) covering every .npy, for mxq_compile's calib_data_path.

    A single manifest file is one calib_data_path (a list of dirs is read as
    one-per-model and fails the count check); the manifest covers all languages.
    """
    npy_files = sorted(glob.glob(os.path.join(save_dir, "*", "*.npy")))
    if not npy_files:
        raise FileNotFoundError(
            f"No calibration .npy under {save_dir}; check --ratio and the {MIN_SEQLEN}-token minimum"
        )
    shape = list(np.load(npy_files[0], mmap_mode="r").shape)  # [1, seq_len, embed_dim]
    shape[1] = -1  # sequence length is dynamic
    manifest = os.path.join(save_dir, "calib_files.json")
    with open(manifest, "w") as f:
        json.dump(
            {
                "info": {"input names": ["inputs_embeds"], "input shapes": [shape]},
                "calib paths": [[os.path.abspath(p)] for p in npy_files],
            },
            f,
            indent=2,
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a per-model calibration set (inputs_embeds) from raw text.")
    parser.add_argument("--model", required=True, choices=MODEL_IDS, help="model id to build the calibset for")
    parser.add_argument("--ratio", type=float, default=1.0, help="scale every language's quota (1.0 = recipe distribution)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    embedding_layer = load_embedding_layer(model_id, device)

    save_dir = f"./calibset_{model_id.split('/')[-1]}"
    total = generate_for_model(tokenizer, embedding_layer, device, args.ratio, save_dir)
    manifest = write_calib_manifest(save_dir)

    print(f"Done: {total} calib files in {os.path.abspath(save_dir)}")
    print(f"Manifest: {os.path.abspath(manifest)}")


if __name__ == "__main__":
    main()
