"""The cityscapes benchmark contract: its validation samples and its scorer.

`python score.py samples --task TASK --data-path DIR` prints the validation samples in
scoring order as JSON (`{"samples": [{"image", "image_id"}, ...]}`).
`python score.py score --data-path DIR --artifact FILE` scores a prediction artifact (a
JSON object with the `task`, the `image_ids` it covers, and the `records`;
dataloader.py and evaluator.py define what an id means and which record fields are read)
and prints the result as JSON.
"""

import argparse
import json
from pathlib import Path

import dataloader
import evaluator


def main() -> int:
    argument_parser = argparse.ArgumentParser(prog="python score.py", description=__doc__)
    commands = argument_parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser("samples", help="print the validation samples")
    samples.add_argument("--task", required=True)
    scoring = commands.add_parser("score", help="score a prediction artifact")
    scoring.add_argument(
        "--artifact", type=Path, required=True, help="prediction artifact (JSON)"
    )
    for command in (samples, scoring):
        command.add_argument(
            "--data-path",
            type=Path,
            required=True,
            help="the prepared evaluation directory, <dataset-dir>/eval",
        )
    args = argument_parser.parse_args()
    if args.command == "samples":
        payload = {"samples": dataloader.samples(args.task, args.data_path)}
    else:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        payload = evaluator.score_artifact(args.data_path, artifact)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
