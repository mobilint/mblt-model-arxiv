[한국어](README.ko.md)

# occluded-duke

Prepare Occluded-Duke by re-splitting a supplied DukeMTMC-reID archive with the published lists.

## Preparation

```bash
python -m pip install -r requirements.txt
python prepare.py --help
python prepare.py --dataset-dir /path/to/occluded-duke
```

The full validation data is written to `eval/` and the calibration images to `calib/`.
The dataset itself is not included: review its provider's terms before downloading, or
point the source options listed by `--help` at an archive or directory you already have.

## Scoring

| File | Role |
| --- | --- |
| `prepare.py` | Download, organize, and select calibration images |
| `dataloader.py` | Read the prepared validation samples in their fixed order |
| `evaluator.py` | Ground truth, label mapping, and metrics |
| `score.py` | Entry point listing the validation samples (`samples`) and scoring a prediction artifact (`score`) |

```bash
python score.py score --data-path /path/to/occluded-duke/eval --artifact predictions.json
```

A prediction artifact is a JSON object with the `task`, the `image_ids` it covers, and the
`records`; `dataloader.py` and `evaluator.py` define what an id means and which record
fields are read.
