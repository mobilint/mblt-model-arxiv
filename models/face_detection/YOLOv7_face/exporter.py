"""Export legacy face heads with exact convolutional landmark removal."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

DETECTION_CHANNELS = 5 + 1

UPSTREAM_CHECKOUT_ENV = "MOBILINT_UPSTREAM_CHECKOUT"
# Exact upstream source each face family is exported from; checkouts are verified against it.
UPSTREAM_COMMITS = {
    "deepcam-cn/yolov5-face": "152c688d551aefb973b7b589fb0691c93dab3564",
    "derronqi/yolov7-face": "74a2a1df1439f7f5b87c87d7857d874f891acbdc",
}

LANDMARK_LAYOUTS = {
    "yolov5-face": {"channels": 16, "class_index": 15},
    "yolov7-face": {"channels": 21, "class_index": 5},
}


def _output_names(output: Any) -> list[str]:
    if isinstance(output, (tuple, list)):
        return [f"output{index}" for index in range(len(output))]
    return ["output0"]


def _export(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output: Path,
    opset: int,
    input_name: str = "input",
) -> None:
    model = model.cpu().eval()
    # Some legacy YOLO models cache tensors created during this probe.  Use
    # no_grad instead of inference_mode so the later ONNX trace can reuse them.
    with torch.no_grad():
        result = model(input_tensor)
    output_names = _output_names(result)
    dynamic_axes = {input_name: {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        (input_tensor,),
        output,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        **export_kwargs,
    )
    _resolve_static_shapes(output)
    print(f"Wrote {output}")


def _resolve_static_shapes(output: Path) -> None:
    """Fill in output dimensions the exporter left symbolic.

    The yolov7-face heads end in the landmark Slice, and the TorchScript
    exporter does not propagate shapes through it, so each head is annotated
    ``(batch, 3, Sliceoutput0_dim_2, Sliceoutput0_dim_3, 6)``. Those axes are
    static in reality — only the batch dimension varies — so ONNX shape
    inference recovers them, leaving the declared shapes agreeing with what the
    graph actually produces.
    """
    import onnx
    from onnx import checker, shape_inference

    model = onnx.load(str(output))
    try:
        inferred = shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
    except (shape_inference.InferenceError, checker.ValidationError) as exc:
        # Shape inference only improves annotations, so a graph it cannot
        # analyse keeps the exporter's own shapes rather than failing the export.
        print(f"Left exporter-declared shapes in place: {exc}")
        return
    onnx.save(inferred, str(output))


def _detection_head(model: torch.nn.Module) -> torch.nn.Module:
    """Return the anchor-based detection head of a loaded face checkpoint."""
    heads = [
        module
        for module in model.modules()
        if hasattr(module, "nl") and hasattr(module, "na") and hasattr(module, "m")
    ]
    if not heads:
        raise ValueError("Checkpoint exposes no anchor-based detection head.")
    return heads[-1]


def _yolov5_face_head(self: Any, x: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Emit yolov5-face raw heads with the landmark channels already selected."""
    outputs = []
    for index in range(self.nl):
        value = self.face_select[index](self.m[index](x[index]))
        _batch, _channels, height, width = value.shape
        outputs.append(
            value.view(-1, self.na, self.face_no, height, width)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    return tuple(outputs)


def _yolov7_face_head(self: Any, x: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Emit yolov7-face raw heads with the keypoint channels already selected."""
    outputs = []
    for index in range(self.nl):
        detection = self.im[index](self.m[index](self.ia[index](x[index])))
        value = torch.cat((detection, self.m_kpt[index](x[index])), 1)
        value = self.face_select[index](value)
        _batch, _channels, height, width = value.shape
        outputs.append(
            value.view(-1, self.na, self.face_no, height, width)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    return tuple(outputs)


FACE_HEAD_FORWARDS = {
    "yolov5-face": _yolov5_face_head,
    "yolov7-face": _yolov7_face_head,
}


def _install_face_head(model: torch.nn.Module, family: str) -> None:
    """Select the box/objectness/class channels with a 1x1 convolution.

    Slicing the head tensor is not enough for this repository's MXQ path:
    qbcompiler cuts the graph at the head convolution and drops the trailing
    Slice, so the compiled artifact still exposes the upstream per-anchor layout
    (16 channels for yolov5-face, 21 for yolov7-face) and the repository's
    ``_decode_anchor_heads`` then reads the landmark channels as extra classes.
    Compiling with ``--no-yolo-decode-include`` does not change that.

    A convolution survives the cut, so the same selection is expressed as a 1x1
    convolution with one-hot weights appended to the head. It is exact — each
    output channel copies one input channel — and works for every variant,
    including `yolov7-tiny-face`, whose ``dw_conv_kpt`` keypoint branch is a deep
    Sequential that cannot be folded into the detection convolution.
    """
    import types

    head = _detection_head(model)
    layout = LANDMARK_LAYOUTS[family]
    channels, class_index = int(layout["channels"]), int(layout["class_index"])
    keep = [*range(DETECTION_CHANNELS - 1), class_index]
    anchors, total = int(head.na), int(head.na) * channels
    indices = [anchor * channels + k for anchor in range(anchors) for k in keep]

    selections = torch.nn.ModuleList()
    for _ in range(int(head.nl)):
        convolution = torch.nn.Conv2d(total, len(indices), 1, bias=False)
        with torch.no_grad():
            convolution.weight.zero_()
            for row, source in enumerate(indices):
                convolution.weight[row, source, 0, 0] = 1.0
        selections.append(convolution)
    head.face_select = selections
    head.face_no = len(keep)
    head.forward = types.MethodType(FACE_HEAD_FORWARDS[family], head)
    print(
        f"Installed a 1x1 channel-selection head: {total} -> {len(indices)} channels."
    )


def _load_legacy_checkpoint(weights: Path, repository: Path) -> torch.nn.Module:
    """Load checkpoints whose model classes are supplied by a cloned repository."""
    if not repository.is_dir():
        raise ValueError(f"Legacy repository does not exist: {repository}")
    if not weights.is_file():
        raise ValueError(f"Checkpoint does not exist: {weights}")
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location="cpu")
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema") or checkpoint.get("model")
    else:
        model = checkpoint
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Unsupported checkpoint format in {weights}")
    model = model.float()
    if hasattr(model, "fuse"):
        model = model.fuse()
    # Both face heads must stay on their decoding path: the raw convolution
    # output an ``export`` flag would expose concatenates the detection and
    # landmark convolutions along the channel axis, so its channels do not line
    # up with the per-anchor layout the checkpoints were trained through.
    for module in model.modules():
        if hasattr(module, "export"):
            module.export = False
        if hasattr(module, "export_cat"):
            module.export_cat = False
    return model.eval()


def _patch_channel_shuffle() -> None:
    """Make the ShuffleNet-style backbones export with a dynamic batch axis.

    Both forks implement channel shuffle as::

        batchsize, num_channels, height, width = x.data.size()
        x = x.view(batchsize, groups, num_channels // groups, height, width)

    ``x.data.size()`` returns plain ints, so tracing bakes the probe's batch
    size into the Reshape and the exported graph then rejects any other batch
    ("input_shape_size == size was false"). Rebuilding the same reshape with a
    ``-1`` batch dimension is arithmetically identical and resolves at runtime,
    which keeps the dynamic batch dimension this repository requires.

    Only the ShuffleV2/lite backbones (yolov5n, yolov5n-0.5, yolov7-lite-*) go
    through this helper, so it is a no-op for the other variants.
    """
    # torch.load has already imported the checkpoint's upstream module.
    common = sys.modules.get("models.common")
    if common is None or not hasattr(common, "channel_shuffle"):
        return

    def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        # Keep channels and spatial extent as plain ints exactly as upstream
        # does, so they stay static in the graph; only the batch dimension
        # becomes -1. Reading them as x.shape[i] instead would trace every axis
        # as dynamic and leave the exported heads with symbolic spatial dims.
        _batch, num_channels, height, width = x.data.size()
        x = x.view(-1, groups, num_channels // groups, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(-1, num_channels, height, width)

    common.channel_shuffle = channel_shuffle
    print("Patched channel_shuffle for a dynamic batch dimension.")


# Checkout verification shared byte-for-byte in behaviour with legacy-yolo.v1.
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
    from package_common import ARTIFACTS_DIR

    name = customer_contract["model"]
    family = "yolov5-face" if name.startswith("YOLOv5") else "yolov7-face"
    upstream = (
        "deepcam-cn/yolov5-face" if family == "yolov5-face" else "derronqi/yolov7-face"
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Upstream source is scratch for this export only: artifacts/ keeps data, never Python.
    with tempfile.TemporaryDirectory(prefix="mobilint-upstream-") as scratch:
        repository = _checkout(upstream, Path(scratch), ARTIFACTS_DIR.parent)
        if weights.startswith("https://"):
            import gdown

            checkpoint = ARTIFACTS_DIR / f"{name}.pt"
            if not checkpoint.exists():
                gdown.download(weights, str(checkpoint), quiet=False)
        else:
            checkpoint = Path(weights)
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
                        "family": family,
                        "upstream": upstream,
                    }
                ),
            ],
            cwd=repository,
            check=True,
            # A retained offline checkout must stay clean for the next verification.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )


def _export_checkpoint(payload: dict[str, Any]) -> None:
    """Keep upstream's models/utils modules confined to a fresh interpreter."""
    _verify_checkout(Path.cwd(), payload["upstream"])
    customer_contract = payload["contract"]
    checkpoint = Path(payload["weights"])
    repository = Path.cwd()
    family = payload["family"]
    output, opset = Path(payload["output"]), payload["opset"]
    model = _load_legacy_checkpoint(checkpoint, repository)
    _patch_channel_shuffle()
    _install_face_head(model, family)
    tensor = torch.zeros(customer_contract["input"]["shape"], dtype=torch.float32)
    _export(model, tensor, output, opset, customer_contract["input"]["name"])
