[한국어](README.ko.md)

# RepViT_M1_5

- Task: Image classification
- Target device: `aries-rb`
- Source model: [timm RepViT_M1_5](https://github.com/huggingface/pytorch-image-models) (license Apache-2.0)
- Checkpoint: <https://huggingface.co/timm/repvit_m1_5.dist_300e_in1k>

The checkpoint and the dataset are not included; each keeps its provider's terms.

## Expected result

| | Result | Conditions | Model Download |
| --- | --- | --- | --- |
| Mobilint recorded, float (ONNX) | Top-1 0.82372, Top-5 0.96028 | ImageNet-1k validation, 50,000 images, ONNX | — |
| Reproduced with this folder, NPU (MXQ) | Top-1 0.81618, Top-5 0.95666 | qbcompiler 1.4.0, qbruntime 1.5.0 | [mxq ↓](https://huggingface.co/mobilint/RepViT_M1_5/resolve/366a45f4a5a08eb92b5976e7c8d5e15bd1e45209/aries-rb/repvit_m1_5.mxq?download=true) |

MXQ SHA-256: `e3fb5b19d7502458d5a0a0f6058dfbbc15f101e2ec0dd77e555525583bf14235` (file `repvit_m1_5.mxq`). After downloading, verify with `sha256sum repvit_m1_5.mxq`; the digest must match exactly. The link is pinned to Hugging Face revision `366a45f4a5a08eb92b5976e7c8d5e15bd1e45209` of repository `mobilint/RepViT_M1_5`.

The Mobilint NPU reference that `evaluate.py` compares against (stored in `model_contract.json`) is a legacy record without an execution image digest or input lineage. It was measured with an internal build in the `mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04` image; the reproduction was checked against it within the stated tolerance.

## Reproduce: from compile to evaluation

Needs Python 3.10, `qbcompiler` and `qbruntime` from the Mobilint SDK, and an attached NPU. To match qbcompiler 1.4.0, use a `mobilint/qbcompiler` image whose tag starts with `1.4-`: pick the `-cuda...` tag to compile on a GPU or the `-cpu-...` tag to compile on CPU ([published tags](https://hub.docker.com/r/mobilint/qbcompiler/tags)). If no published tag matches both that prefix and the requested execution mode, use the latest published tag of the same execution mode, still install the version above of the qbcompiler wheel, and record the image you actually used with your result. Only fall back after a successful registry listing confirms the tag is absent; stop on network or registry errors.
The Dataset code is in this repository's [`datasets/imagenet`](../../../datasets/imagenet/).

```bash
python -m pip install -r requirements.txt
python ../../../datasets/imagenet/prepare.py --dataset-dir /path/to/imagenet
python compile.py --preflight
python export.py
python prepare_calibration.py --source /path/to/imagenet/eval
python compile.py
python evaluate.py --data-path /path/to/imagenet/eval
```

By default, aries and aries-rb compile all execution plans (Single, Multi, Global4, and Global8). Pass `--inference-scheme single` to build an artifact with only the single plan.

`qbcompiler` pins its own torch; if `export.py` needs a newer torch, run the export alone in
a separate environment. `evaluate.py` writes `artifacts/evaluation/result.json` and shows
the difference: `--framework mxq` (the default) compares against Mobilint's internally
recorded NPU result stored in `model_contract.json` (the reference the published
reproduction was accepted against, within the stated tolerance), and `--framework onnx`
against the float result above. `--limit N` checks the setup on the first N
images only. `evaluate.py` refuses to overwrite an earlier result unless `--force` is
passed. `inference.py` overwrites existing output files with the same name and does not
support `--force`.

## Single-image inference

```bash
python inference.py --source /path/to/image.jpg --framework mxq
python inference.py --source /path/to/image.jpg --framework onnx
```

## Synchronous and asynchronous image streams

```bash
python inference_sync.py --source /path/to/images --core-mode multi
python inference_async.py --source /path/to/images --core-mode multi --queue-depth 4
```

Both versions load the model once and prefetch a bounded number of preprocessed images.
The asynchronous version uses qbruntime's `infer_async` and completion futures, preserving
input order. Tune preprocessing with `--preprocess-workers`; add `--save-outputs` to save
per-image raw outputs and task summaries. By default, output files are omitted for
throughput measurement. Timing includes preprocessing and inference (plus postprocessing
and writes when enabled), excluding model loading. Compare the same images with `--limit`
and `--repeat`; asynchronous execution is not guaranteed to be faster for every workload.

Every script accepts `--help`.
