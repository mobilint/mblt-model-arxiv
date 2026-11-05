[한국어](README.ko.md)

# COCO 2017

Prepare COCO 2017 validation data for calibration and evaluation.

The script downloads (or accepts local copies of) the COCO validation images
and annotations, writes the full validation set to `eval/`, and creates a
deterministic flat calibration set in `calib/`.

## Preparation

```bash
python -m pip install -r requirements.txt
python prepare.py --help
python prepare.py --dataset-dir /path/to/coco
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
python score.py score --data-path /path/to/coco/eval --artifact predictions.json
```

A prediction artifact is a JSON object with the `task`, the `image_ids` it covers, and the
`records`; `dataloader.py` and `evaluator.py` define what an id means and which record
fields are read.
