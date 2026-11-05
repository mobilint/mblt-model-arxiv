#!/usr/bin/env python3
"""Select the calibration images used by compile.py.

Input: --source, a directory of the dataset named in calibration_spec.json, laid out as that
file describes (the customer supplies this data; it is never included in the package).
Output: calibration/<dataset>-<profile>/ with the selected images and a manifest.json. The
selection is deterministic for a given source directory. Prerequisite: a licensed copy of
the dataset; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import calibration_selection
from package_common import (
    PackageError,
    calibration_output,
    exit_with_package_error,
    load_calibration_spec,
    make_staging,
    refuse_overwrite,
    replace_directory,
    sha256_file,
    write_json,
)


def parse_args(spec: dict) -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=__doc__
        + f"\nexpected source layout: {calibration_selection.describe_source(spec)}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argument_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=f"root directory of the {spec['dataset']} data ({spec['source']['description']})",
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=calibration_output(spec),
        help="calibration directory to create (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--force", action="store_true", help="replace an existing calibration directory"
    )
    return argument_parser.parse_args()


def main() -> None:
    spec = load_calibration_spec()
    args = parse_args(spec)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise PackageError(f"--source directory not found: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise PackageError("--output must not overlap --source")
    refuse_overwrite(output, args.force)
    selected = calibration_selection.select(spec, source)
    expected = int(spec["expected_sample_count"])
    if len(selected) != expected:
        raise PackageError(
            f"selected {len(selected)} images, the profile expects {expected}"
        )
    staging = make_staging(output)
    try:
        records = []
        for item in selected:
            destination = staging / item["file"]
            shutil.copy2(item["path"], destination)
            record = {key: value for key, value in item.items() if key != "path"}
            record["sha256"] = sha256_file(destination)
            records.append(record)
        write_json(
            staging / "manifest.json",
            {
                "dataset": spec["dataset"],
                "profile": spec["profile"],
                "layout": spec["layout"]["adapter"],
                "selection": spec["selection"],
                "sample_count": len(records),
                "samples": records,
            },
        )
        replace_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "calibration_dir": str(output),
                "sample_count": len(records),
                "manifest": str(output / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    exit_with_package_error(main)
