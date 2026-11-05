"""Export original YOLO checkpoints using their upstream Python model definitions."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

# Verified official upstream commits. v3/v7/v9 match the checkouts used for the
# standalone export validation; the Model records contain no upstream source SHA.
# sunsmarterjie/yolov12 is the main-branch head when the YOLO12 segmentation records were
# exported (2026-08-19); later commits change only training-label handling.
UPSTREAM_CHECKOUT_ENV = "MOBILINT_UPSTREAM_CHECKOUT"
UPSTREAM_COMMITS = {
    "ultralytics/yolov5": "f9578796b8a64889aa9e66dc06914d19f40d338e",
    "ultralytics/yolov3": "9e14913298c1ece6eb6bebf40652f150bf733a85",
    "WongKinYiu/yolov7": "a207844b1ce82d204ab36d87d496728d3d2348e7",
    "WongKinYiu/yolov9": "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff",
    "sunsmarterjie/yolov12": "01a22c0603e0eaa6d9bd62120a391e744d92cea2",
}


# SHA-256 values below are measured from the official release URLs, not from
# pre-existing customer caches. Unknown declarations fail before any loading.
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt": (
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt",
        "154bb05fceb4f06f062a085a7f3d99d006cf5f1f53237c3a1bdbd66b0ba1a74c",
    ),
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt": (
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt",
        "ec340c38bacae4a99f586acb84d53e1f9c38b60835ccdf992222b033cb7d5ca3",
    ),
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt": (
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt",
        "b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb",
    ),
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt": (
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt",
        "2be96abd35939335e5eb08bcde2685ae44c3fadd8af7d601734ba10c8737425c",
    ),
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt": (
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
        "d1ac3c74eb96a3eec77949c0f37a06bc272756606ff011d9353f7abff4e0c71d",
    ),
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt": (
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt",
        "4873202bc3fa8b1d4c4865a37b7c61ce7b7b1cdbe6f31017c33be2bd11826e98",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt",
        "754301bd0f2b36799166047b9fff7b02b8c352d61c503595e37f1bf7abf57650",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt",
        "0c41b9c016a53e0ba1848979a22d5535c3aa858548c484c4288b865e234d660b",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt",
        "11777ccf5faf5d4a23ef1a2059a88b7eb972c1b2f81a3f68960121c1f391130d",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt",
        "4420263ca93c1c052352adcda5f33f0cb259d4756b7cf6492bf8101176263578",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-s.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-s.pt",
        "43d1662df7ed02ca1c0590c333e7ff612bb02392d5b421281d6b8fc8effa3450",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt",
        "39b6b490b1e4034b4fefdcdac81380155e0a30ba6c938a9751e2c351f81a517f",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-seg-converted.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-seg-converted.pt",
        "b42bb1f79a93744affadce16cbd56560e4311ee89ea35f0e2dd2d78d58ac1647",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt",
        "6763139daa09adfe3c694226725b460211f1f717b971bfade10a548b600716e2",
    ),
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt": (
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt",
        "4f60eef3aca520d4a74e80d1c246c6b094ba40228f22bea13e864cd7a49c2349",
    ),
    "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-spp.pt": (
        "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-spp.pt",
        "c79836b6b21e39bca1fa83b090ac36776878d7cc51d2c68904b817db67164de8",
    ),
    "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12l-seg.pt": (
        "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12l-seg.pt",
        "ec0a4bcd911f4080e5bc8d623a2ba7c544c96d8b865eb375620039f81be6e7d0",
    ),
    "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt": (
        "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt",
        "792a2068359a9d95861b91f4771b04ac76b5148637e3dde700c6a84277cc86ca",
    ),
    "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12n-seg.pt": (
        "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12n-seg.pt",
        "c43e1d4025b9b03248dea7b65b9b41b3388d40c7a1efbeb98efe5e1adbd66a2d",
    ),
    "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt": (
        "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt",
        "b31eaf827203baf8057ae20230b37f4647b7b5d044c8f390c4db652ecc35a508",
    ),
    "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt": (
        "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt",
        "7efb43e49634284ca19a3adbf6e1534425bb22816adb0153482fb99e4a283ac8",
    ),
    "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt": (
        "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt",
        "ba6568ba6a771d99e71546595079f93de7c69701ba483853dbefef4fc2750160",
    ),
    "yolov5l-cls.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt",
        "629a14d447589a5d57e9e980d7c003b4234f25e0f215acc67c1600e61bceb8d7",
    ),
    "yolov5l-seg.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt",
        "06fb9b8403a16d2b08db33b879790b5b68871b395166eef2ed2b37c82605580f",
    ),
    "yolov5l.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
        "2f603b7354c25454d1270663a14d8ddc1eea98e5eebc1d84ce0c6e3150fa155f",
    ),
    "yolov5l6.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt",
        "b357b77f646b0190912985937c44e2923441369d5fda69ba86da2aa919212618",
    ),
    "yolov5m-cls.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-cls.pt",
        "dbfdf14826b47fd45b178f652048339d83dc437d93a893f5dfbea6729578794d",
    ),
    "yolov5m-seg.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt",
        "2a66141ea97b14c2ccb6440d442c0757ff03e51b4e311904b3629b0249a4dc4f",
    ),
    "yolov5m.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
        "61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56",
    ),
    "yolov5m6.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt",
        "7afe7fa0f29a8351467200a2a5a33c7996d1155ed6636c7edd8ca70b14951c64",
    ),
    "yolov5n-cls.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt",
        "77b15f65325b37c24e616cbe0e5349d63c35508607458b9f2bcb03875ab75f33",
    ),
    "yolov5n-seg.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt",
        "1ee00e5e14e3863898c5446c62734776759e29be08fd1c174774992f7aaa9985",
    ),
    "yolov5n6.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt",
        "496c05a2b991da15acc6c4408c63da287f2513a46c6756618776d4dd71170781",
    ),
    "yolov5s-cls.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt",
        "53ebfe1c19d60bea82c821b93b52bf58d81d5a06b693668e516fd8d44769881e",
    ),
    "yolov5s-seg.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt",
        "78558867515e9d9654c52ed87c40c22d90a983bea63450c611cbd8c2b13e3aa2",
    ),
    "yolov5s.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        "8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee",
    ),
    "yolov5s6.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt",
        "95bda9019ada63f37338308d0c81a66047a6fbaba061abaff271ba9334cf2a6f",
    ),
    "yolov5x-cls.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-cls.pt",
        "d1786d2a292db7141c5026726042783b737803ca947f61bdb0166e1b03ebf157",
    ),
    "yolov5x-seg.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt",
        "a7f2c881dd1e82b305875de01c44ad28c12807f58dcd2be41ef6544166e4c3e0",
    ),
    "yolov5x.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",
        "9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859",
    ),
    "yolov5x6.pt": (
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt",
        "257c991b216d625427dc4d9c7c76d6c6d93cf3732797538da34bdc02f97bfef0",
    ),
}


def _checkpoint_spec(declared: str) -> tuple[str, str]:
    try:
        return CHECKPOINTS[declared]
    except KeyError as error:
        raise ValueError(f"No verified release checkpoint for {declared!r}") from error


def _reject_checkpoint_symlinks(path: Path) -> None:
    absolute = path.absolute()
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise ValueError(f"Checkpoint path must not contain symlinks: {path}")


def _verify_checkpoint(path: Path, declared: str) -> None:
    """Verify ordinary file bytes before permitting pickle-backed model loading."""
    _, expected = _checkpoint_spec(declared)
    _reject_checkpoint_symlinks(path)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"Checkpoint is not a regular file: {path}")
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Cannot read verified checkpoint: {path}") from error
    if digest.hexdigest() != expected:
        raise ValueError(
            f"Checkpoint SHA-256 differs from the declared release: {path}"
        )


def _download_checkpoint(path: Path, declared: str) -> None:
    """Publish only complete hash-verified downloads; retain no partial cache."""
    url, _ = _checkpoint_spec(declared)
    _reject_checkpoint_symlinks(path)
    if path.exists() or path.is_symlink():
        _verify_checkpoint(path, declared)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".download", delete=False
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _verify_checkpoint(temporary, declared)
        # A second downloader may have completed while this request was in flight.
        if path.exists() or path.is_symlink():
            _verify_checkpoint(path, declared)
        else:
            os.replace(temporary, path)
        _verify_checkpoint(path, declared)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_checkpoint(declared: str, weights: str, artifacts: Path) -> Path:
    url, _ = _checkpoint_spec(declared)
    local = Path(weights)
    if weights != declared or (
        not weights.startswith("https://") and (local.exists() or local.is_symlink())
    ):
        _verify_checkpoint(local, declared)
        return local
    checkpoint = artifacts / Path(url).name
    _download_checkpoint(checkpoint, declared)
    return checkpoint


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    ).stdout.strip()


def _verify_tracked_files(repository: Path, commit: str) -> None:
    """Compare real bytes and types to the commit, never the mutable Git index."""
    tree = subprocess.run(
        ["git", "-C", str(repository), "ls-tree", "-rz", "--full-tree", commit],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    ).stdout
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, relative = entry.split(b"\t", 1)
        mode, kind, expected = metadata.split()
        path = repository / os.fsdecode(relative)
        for parent in path.relative_to(repository).parents:
            if (repository / parent).is_symlink():
                raise ValueError(f"Upstream source directory is a symlink: {parent}")
        try:
            actual = path.lstat()
            if kind != b"blob":
                raise ValueError(f"Unsupported upstream tree entry: {path}")
            if mode == b"120000" and stat.S_ISLNK(actual.st_mode):
                content = os.fsencode(os.readlink(path))
            elif mode in (b"100644", b"100755") and stat.S_ISREG(actual.st_mode):
                if bool(actual.st_mode & stat.S_IXUSR) != (mode == b"100755"):
                    raise ValueError(
                        f"Upstream source mode differs from pinned tree: {path}"
                    )
                content = path.read_bytes()
            else:
                raise ValueError(
                    f"Upstream source type differs from pinned tree: {path}"
                )
        except OSError as error:
            raise ValueError(f"Cannot read pinned upstream source: {path}") from error
        blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
        if (
            hashlib.sha1(blob, usedforsecurity=False).hexdigest().encode("ascii")
            != expected
        ):
            raise ValueError(f"Upstream source bytes differ from pinned tree: {path}")


def _verify_checkout(repository: Path, upstream: str) -> None:
    """Refuse a replaced, stale, modified or shadowed cached source tree."""
    expected_url = f"https://github.com/{upstream}.git"
    if (
        repository.is_symlink()
        or not (repository / ".git").is_dir()
        or (repository / ".git").is_symlink()
    ):
        raise ValueError(
            f"Upstream checkout is not an owned Git directory: {repository}"
        )
    if (
        Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
        != repository.resolve()
    ):
        raise ValueError(f"Upstream checkout root differs from {repository}")
    if _git(repository, "remote", "get-url", "origin") != expected_url:
        raise ValueError(f"Upstream checkout origin must be {expected_url}")
    if _git(repository, "rev-parse", "HEAD") != UPSTREAM_COMMITS[upstream]:
        raise ValueError(f"Upstream checkout HEAD must be {UPSTREAM_COMMITS[upstream]}")
    _verify_tracked_files(repository, UPSTREAM_COMMITS[upstream])
    # Ignored Python files can shadow imports just like untracked files.
    if _git(repository, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError(
            f"Upstream checkout is not clean: {repository}; use a fresh artifact directory"
        )


def _prepare_checkout(repository: Path, upstream: str) -> None:
    if not repository.exists() and not repository.is_symlink():
        repository.mkdir(parents=True)
        _git(repository, "init")
        _git(
            repository, "remote", "add", "origin", f"https://github.com/{upstream}.git"
        )
        _git(repository, "fetch", "--depth", "1", "origin", UPSTREAM_COMMITS[upstream])
        _git(repository, "checkout", "--detach", UPSTREAM_COMMITS[upstream])
    _verify_checkout(repository, upstream)


def _checkout(upstream: str, scratch: Path, model_dir: Path) -> Path:
    """A verified checkout of the pinned upstream commit, outside the Model folder.

    By default a fresh one in *scratch* (removed with it). Offline, the directory named by
    MOBILINT_UPSTREAM_CHECKOUT is used after the same verification and is never removed.
    """
    supplied = os.environ.get(UPSTREAM_CHECKOUT_ENV)
    if supplied:
        repository = Path(supplied).resolve()
        if (
            repository == model_dir.resolve()
            or model_dir.resolve() in repository.parents
        ):
            raise ValueError(
                f"{UPSTREAM_CHECKOUT_ENV} must point outside the Model folder: {repository}"
            )
        _verify_checkout(repository, upstream)
        return repository
    repository = scratch / "upstream"
    try:
        _prepare_checkout(repository, upstream)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Fetching pinned upstream source {upstream}@{UPSTREAM_COMMITS[upstream]} needs "
            f"network access; offline, set {UPSTREAM_CHECKOUT_ENV} to a clean checkout of "
            f"https://github.com/{upstream}.git at that commit"
        ) from error
    return repository


def export_onnx(
    customer_contract: dict[str, Any], *, weights: str, output: Path, opset: int
) -> None:
    from package_common import ARTIFACTS_DIR, PackageError

    declared_weights = customer_contract["export"]["weights"]
    _checkpoint_spec(declared_weights)
    if not declared_weights.startswith("https://"):
        upstream = "ultralytics/yolov5"
    else:
        upstream = next(
            (
                name
                for name in UPSTREAM_COMMITS
                if name != "ultralytics/yolov5"
                and declared_weights.startswith(f"https://github.com/{name}/releases/")
            ),
            None,
        )
        if upstream is None:
            raise PackageError(
                "legacy-yolo.v2 requires a checkpoint release URL from a supported upstream repository"
            )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = _prepare_checkpoint(declared_weights, weights, ARTIFACTS_DIR)
    # Upstream source is scratch for this export only: artifacts/ keeps data, never Python.
    with tempfile.TemporaryDirectory(prefix="mobilint-upstream-") as scratch:
        repository = _checkout(upstream, Path(scratch), ARTIFACTS_DIR.parent)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,runpy,sys; runpy.run_path(sys.argv[1])['_export_checkpoint'](json.loads(sys.argv[2]))",
                str(Path(__file__).resolve()),
                json.dumps(
                    {
                        "contract": customer_contract,
                        "weights": str(checkpoint.resolve()),
                        "output": str(output.resolve()),
                        "opset": opset,
                        "upstream": upstream,
                    }
                ),
            ],
            cwd=repository,
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )


def _export_checkpoint(payload: dict[str, Any]) -> None:
    """Run in a clean interpreter whose working directory is the upstream checkout."""
    _verify_checkout(Path.cwd(), payload["upstream"])
    checkpoint = Path(payload["weights"])
    _verify_checkpoint(checkpoint, payload["contract"]["export"]["weights"])
    import torch

    if payload["upstream"] == "ultralytics/yolov5":
        model = torch.hub.load(
            str(Path.cwd()),
            "custom",
            str(checkpoint),
            source="local",
            autoshape=False,
            trust_repo=True,
            device="cpu",
        )
        _export_model(
            model, payload["contract"], Path(payload["output"]), payload["opset"]
        )
        return
    # Upstream YOLO releases contain Python model objects, not state_dict-only files.
    checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = (
        (checkpoint_object.get("ema") or checkpoint_object.get("model"))
        if isinstance(checkpoint_object, dict)
        else checkpoint_object
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Unsupported model object in {checkpoint}")
    model = model.float()
    if hasattr(model, "fuse"):
        model = model.fuse()
    for module in model.modules():
        if hasattr(module, "export"):
            module.export = True
    _export_model(model, payload["contract"], Path(payload["output"]), payload["opset"])


def _export_model(
    model: Any, customer_contract: dict[str, Any], output: Path, opset: int
) -> None:
    import torch

    model = model.cpu().eval()
    example = torch.zeros(customer_contract["input"]["shape"], dtype=torch.float32)
    with torch.no_grad():
        result = model(example)
    names = (
        [f"output{i}" for i in range(len(result))]
        if isinstance(result, (tuple, list))
        else ["output0"]
    )
    input_name = customer_contract["input"]["name"]
    axes = {name: {0: "batch"} for name in [input_name, *names]}
    kwargs = (
        {"dynamo": False}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters
        else {}
    )
    torch.onnx.export(
        model,
        (example,),
        str(output),
        input_names=[input_name],
        output_names=names,
        dynamic_axes=axes,
        opset_version=opset,
        **kwargs,
    )
