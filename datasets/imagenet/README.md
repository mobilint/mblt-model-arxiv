[한국어](README.ko.md)

# ImageNet-1k

Prepare ImageNet-1k validation data for calibration and evaluation.

The official validation image archive has no class directories. This script
uses the official validation bounding-box XML archive to place each image in
its ImageNet synset directory, then selects one image from each class for
calibration. The resulting directories support a full 50,000-image evaluation
and a reproducible 1,000-image calibration run.

## Preparation

```bash
python prepare.py --help
python prepare.py --dataset-dir /path/to/imagenet
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
python score.py score --data-path /path/to/imagenet/eval --artifact predictions.json
```

A prediction artifact is a JSON object with the `task`, the `image_ids` it covers, and the
`records`; `dataloader.py` and `evaluator.py` define what an id means and which record
fields are read.
