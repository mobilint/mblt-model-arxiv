"""Shared helpers for the four package entry points.

Everything here is plain Python and NumPy. The package is self-contained: the contract files
beside this module describe the model, and every default output lands under `artifacts/`
or `calibration/` inside the package directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

BUNDLE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BUNDLE_DIR / "artifacts"
CALIBRATION_DIR = BUNDLE_DIR / "calibration"
CONTRACT_FILE = BUNDLE_DIR / "model_contract.json"
CALIBRATION_SPEC_FILE = BUNDLE_DIR / "calibration_spec.json"
COMPILE_CONFIG_FILE = BUNDLE_DIR / "compile_config.json"
EXIT_USAGE = 2


class PackageError(Exception):
    """A user-facing failure: the message says what is missing and what to do next."""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PackageError(
            f"package file is missing: {path.name}; unpack the complete package"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PackageError(f"{path.name} is not a JSON object")
    return payload


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT_FILE)


def load_calibration_spec() -> dict[str, Any]:
    return load_json(CALIBRATION_SPEC_FILE)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(customer_contract: dict[str, Any], suffix: str) -> Path:
    return ARTIFACTS_DIR / f"{customer_contract['model']}.{suffix}"


def calibration_output(spec: dict[str, Any]) -> Path:
    return CALIBRATION_DIR / f"{spec['dataset']}-{spec['profile']}"


def require_file(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise PackageError(f"{path} does not exist; {hint}")
    return path


def refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise PackageError(f"{path} already exists; pass --force to replace it")


def import_optional(module: str, hint: str) -> Any:
    """Import a customer-supplied dependency or explain how to obtain it."""
    try:
        return __import__(module)
    except ImportError as error:
        raise PackageError(
            f"the Python package {module!r} is not installed; {hint}"
        ) from error


def replace_directory(staging: Path, output: Path) -> None:
    """Atomically move a fully written staging directory into place."""
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists():
        raise PackageError(
            f"a stale backup exists; inspect and remove it first: {backup}"
        )
    if output.exists():
        output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def make_staging(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))


def exit_with_package_error(cli_main: Callable[[], None]) -> None:
    """Run a customer CLI and report PackageError as a one-line usage failure."""
    try:
        cli_main()
    except PackageError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE) from None
