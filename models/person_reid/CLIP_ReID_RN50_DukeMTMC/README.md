[한국어](README.ko.md)

# CLIP_ReID_RN50_DukeMTMC

- Task: person_reid
- Target device: `aries-rb`
- Source model: [Syliz517 CLIP-ReID CNN-CLIP-ReID (DukeMTMC-reID)](https://github.com/Syliz517/CLIP-ReID) (license MIT)
- Checkpoint: <https://github.com/Syliz517/CLIP-ReID>

The checkpoint and the dataset are not included; each keeps its provider's terms.

## Expected result

| | Result | Conditions | Model Download |
| --- | --- | --- | --- |
| Mobilint recorded, float (ONNX) | map 0.80713, rank1 0.89901, rank10 0.97307, rank5 0.96409 | dukemtmc-reid validation, 19,889 images, ONNX | — |
| Reproduced with this folder, NPU (MXQ) | map 0.80805, rank1 0.89677, rank10 0.97172, rank5 0.96409 | qbcompiler 1.3.0, qbruntime 1.5.0 | [mxq ↓](https://huggingface.co/mobilint/CLIP_ReID_RN50_DukeMTMC/resolve/20b1017124e65ee21abf772dd777cff03bb8b210/aries-rb/clip_reid_rn50_dukemtmc.mxq?download=true) |

MXQ SHA-256: `b402081572fed92ca426a295595a59b819e17dd73dd5159ad3a7dc778d1c8551` (file `clip_reid_rn50_dukemtmc.mxq`). After downloading, verify with `sha256sum clip_reid_rn50_dukemtmc.mxq`; the digest must match exactly. The link is pinned to Hugging Face revision `20b1017124e65ee21abf772dd777cff03bb8b210` of repository `mobilint/CLIP_ReID_RN50_DukeMTMC`.


## Reproduce: from compile to evaluation

Needs Python 3.10, `qbcompiler` and `qbruntime` from the Mobilint SDK, and an attached NPU. To match qbcompiler 1.3.0, use a `mobilint/qbcompiler` image whose tag starts with `1.3-`: pick the `-cuda...` tag to compile on a GPU or the `-cpu-...` tag to compile on CPU ([published tags](https://hub.docker.com/r/mobilint/qbcompiler/tags)). If no published tag matches both that prefix and the requested execution mode, use the latest published tag of the same execution mode, still install the version above of the qbcompiler wheel, and record the image you actually used with your result. Only fall back after a successful registry listing confirms the tag is absent; stop on network or registry errors.
The Dataset code is in this repository's [`datasets/dukemtmc-reid`](../../../datasets/dukemtmc-reid/).

```bash
python -m pip install -r requirements.txt
python -m pip install -r ../../../datasets/dukemtmc-reid/requirements.txt
python ../../../datasets/dukemtmc-reid/prepare.py --dataset-dir /path/to/dukemtmc-reid
python compile.py --preflight
python export.py
python prepare_calibration.py --source /path/to/dukemtmc-reid/eval/bounding_box_train
python compile.py
python evaluate.py --data-path /path/to/dukemtmc-reid/eval
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
