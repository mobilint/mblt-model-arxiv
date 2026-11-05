[한국어](ARCHITECTURE.ko.md)

# Architecture

This repository has two kinds of folders. Each folder is self-contained: copy it out with
the folders it depends on and it runs as documented.

```text
models/<task>/<Model>/   one Model: export, calibration, compile, evaluation, inference
datasets/<Dataset>/      one Dataset: preparation and scoring
```

## Public interface

- A folder's public interface is its entry scripts and their JSON input and output. Every
  entry script accepts `--help`.
  - Model: `export.py`, `prepare_calibration.py`, `compile.py`, `evaluate.py`, `inference.py`, `inference_sync.py`, `inference_async.py`
  - Dataset: `prepare.py`, `score.py` (`samples` and `score`)
- Every other `.py` file is a helper of its own folder. Only that folder's scripts import
  it, and nothing outside the folder may import it.
- The `.json`, `.yaml`, and `.txt` files are the folder's configuration and data, read by
  its own scripts.

## Dependencies

- Model folders do not depend on each other, and Dataset folders do not depend on each
  other.
- A Model depends only on the Datasets it names: `calibration_spec.json` names the
  Dataset whose prepared images calibrate it, and `model_contract.json` names the
  evaluation Dataset and its folder (`evaluation.dataset_code`).
- A Model uses a Dataset only by running its `score.py` and reading the JSON it prints.
  It never imports Dataset code, and a Dataset never uses a Model.
- Everything else a folder needs is passed in by the user: data as path arguments, and
  public Python packages and the Mobilint SDK as installed dependencies. A folder's own
  `requirements.txt` lists its public packages.

## Data contracts

- `score.py samples --task TASK --data-path DIR` prints
  `{"samples": [{"image": ..., "image_id": ...}, ...]}` in scoring order.
- A prediction artifact is `{"task": ..., "image_ids": [...], "records": [...]}`; the
  Dataset's `dataloader.py` and `evaluator.py` define an id and the record fields.
- `score.py score --data-path DIR --artifact FILE` prints the result:
  `{"primary": {...}, "secondary": {...}, "sample_count": N}`.

## Changes

This repository is generated from Mobilint's own repository and is not edited by hand.
Report a problem as an issue rather than a pull request.
