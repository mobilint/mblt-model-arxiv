# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "torch==2.7.1",
#   "onnx==1.17.0",
#   "onnxsim==0.4.36",
#   "onnxruntime==1.20.1",
#   "numpy<2",
# ]
# ///
"""Export an arcface_torch face-recognition checkpoint to ONNX.

The upstream repository (deepinsight/insightface) is checked out at a pinned commit with
only ``recognition/arcface_torch`` materialized; its ``backbones.get_model`` builds the
IResNet backbone with the 512-dim BN embedding head. The checkpoint is the Model Zoo's
``<folder>/backbone.pth`` in the upstream OneDrive share named by ``--weights`` as
``<share url>#path=<folder>/backbone.pth&sha256=<64 hex>``; it is read through the share's
anonymous API and verified against the digest before loading. The graph is exported with
a static batch of one on the 112x112 aligned face and constant-folded with onnxsim.

The upstream head flattens the final 512x7x7 map and applies a Linear layer. The exported
graph computes the same map with a 7x7 Conv whose kernel is that Linear weight reshaped
to (out, C, 7, 7): identical arithmetic (checked against the upstream forward before
export), but no Flatten over a spatial map, which the NPU compiler lowers in its own
channel-last layout and so scrambled the embedding (MXQ-to-float cosine near 0 with the
Flatten+Gemm head, about 0.99 with the Conv head, qbcompiler 1.3.0).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PureWindowsPath
from typing import Any

import torch

UPSTREAM_COMMIT = "1480e705287bc5d59f923b46c260ec6e3e4150f6"
UPSTREAM_DIR = "recognition/arcface_torch"
CLONE_NAME = "insightface"
BADGER_APP = (
    "5cbed6ac-a083-4e14-b191-b4ba07653de2"  # OneDrive web viewer's public app id
)
SHARE_API = "https://my.microsoftpersonalcontent.com/_api/v2.0"


def network_of(name: str) -> str:
    """The upstream factory name for a Model ID: ``*_IResNet_<depth>*`` is ``r<depth>``."""
    match = re.search(r"_IResNet_(\d+)", name)
    if match is None:
        raise ValueError(f"{name!r} does not name an IResNet depth")
    return f"r{match.group(1)}"


def _checkout(repo: str, checkout_root: Path) -> Path:
    """A sparse clone of ``repo`` at UPSTREAM_COMMIT with no local edits or stray files."""
    target = checkout_root / CLONE_NAME
    if not (target / ".git").is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                repo,
                str(target),
            ],
            check=True,
        )
    git = ["git", "-C", str(target)]
    subprocess.run([*git, "sparse-checkout", "set", UPSTREAM_DIR], check=True)
    subprocess.run(
        [*git, "checkout", "--quiet", "--detach", UPSTREAM_COMMIT], check=True
    )
    subprocess.run([*git, "reset", "--hard", "--quiet", UPSTREAM_COMMIT], check=True)
    subprocess.run([*git, "clean", "-fdxq"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        [*git, "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != UPSTREAM_COMMIT or dirty:
        raise RuntimeError(
            f"{target} is not a clean checkout of {UPSTREAM_COMMIT} (HEAD {head}, "
            f"status {dirty[:200]!r})"
        )
    return target / UPSTREAM_DIR


def weights_reference(weights: str) -> tuple[str, str, str]:
    """Split ``<share url>#path=<p>&sha256=<hex>`` into share URL, item path, digest."""
    share, _, fragment = weights.partition("#")
    fields = dict(urllib.parse.parse_qsl(fragment))
    path, digest = fields.get("path", ""), fields.get("sha256", "")
    if (
        not share.startswith("https://")
        or not path
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError(
            "weights must be '<OneDrive share url>#path=<folder>/backbone.pth&sha256=<64 hex>'"
        )
    if (
        "\\" in path
        or "\0" in path
        or PureWindowsPath(path).drive
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(
            "checkpoint path must be a relative path without dot components"
        )
    return share, path, digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers)
    ) as response:
        return json.loads(response.read())


def _download_url(share: str, item_path: str) -> str:
    """Resolve one file of an anonymous OneDrive share to a short-lived download URL."""
    request = urllib.request.Request(
        "https://api-badgerp.svc.ms/v1.0/token",
        data=json.dumps({"appId": BADGER_APP}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        token = json.loads(response.read())["token"]
    headers = {"Authorization": f"Badger {token}", "Prefer": "autoredeem"}
    encoded = base64.b64encode(share.encode()).decode().rstrip("=")
    encoded = "u!" + encoded.replace("/", "_").replace("+", "-")
    item = _get_json(
        f"{SHARE_API}/shares/{encoded}/driveitem?%24expand=children", headers
    )
    drive = item["parentReference"]["driveId"]
    for part in item_path.split("/"):
        children = {child["name"]: child for child in item.get("children", [])}
        if part not in children:
            raise FileNotFoundError(f"{item_path!r} is not in the share {share}")
        child = children[part]
        query = "?%24expand=children" if "folder" in child else ""
        item_id = urllib.parse.quote(child["id"])
        item = _get_json(f"{SHARE_API}/drives/{drive}/items/{item_id}{query}", headers)
    if "file" not in item:
        raise IsADirectoryError(f"{item_path!r} in the share {share} is not a file")
    return item["@content.downloadUrl"]


def download(weights: str, work_dir: Path) -> Path:
    share, item_path, expected = weights_reference(weights)
    root = work_dir.resolve() / "checkpoints"
    target = root / item_path
    # Reused work directories must not redirect cache reads or writes through links.
    if (
        root.is_symlink()
        or any(
            path.is_symlink()
            for path in (target, *target.parents)
            if path == root or root in path.parents
        )
        or not target.resolve().is_relative_to(root)
    ):
        raise ValueError("checkpoint path must stay inside the checkpoint cache")
    if target.is_file() and _sha256(target) == expected:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with (
            urllib.request.urlopen(_download_url(share, item_path)) as response,
            tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=target.name + ".",
                suffix=".partial",
                delete=False,
            ) as handle,
        ):
            partial = Path(handle.name)
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        actual = _sha256(partial)
        if actual != expected:
            raise RuntimeError(f"{item_path} sha256 {actual} != recorded {expected}")
        partial.replace(target)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return target


class ConvHeadIResNet(torch.nn.Module):
    """The upstream IResNet with its Flatten+Linear head computed as one Conv."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        out_features, flat = backbone.fc.weight.shape
        channels = backbone.bn2.num_features
        side = round((flat / channels) ** 0.5)
        if channels * side * side != flat:
            raise ValueError(
                f"fc input {flat} is not {channels} channels on a square map"
            )
        self.head = torch.nn.Conv2d(channels, out_features, side)
        with torch.no_grad():
            self.head.weight.copy_(
                backbone.fc.weight.view(out_features, channels, side, side)
            )
            self.head.bias.copy_(backbone.fc.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        net = self.backbone
        x = net.prelu(net.bn1(net.conv1(image)))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        x = torch.flatten(self.head(net.bn2(x)), 1)
        return net.features(x)


def build(checkout: Path, name: str, checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, str(checkout))
    from backbones import get_model

    backbone = get_model(network_of(name), dropout=0.0, fp16=False, num_features=512)
    state = torch.load(checkpoint, map_location="cpu")
    backbone.load_state_dict(state, strict=True)
    backbone = backbone.float().eval()
    model = ConvHeadIResNet(backbone).eval()
    probe = torch.randn(2, 3, 112, 112, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        drift = float((model(probe) - backbone(probe)).abs().max())
    if drift > 1e-4:
        raise RuntimeError(f"Conv head differs from the upstream forward by {drift}")
    print(f"Conv head matches the upstream forward (max |diff| {drift:.2e})")
    return model


def export(
    model: torch.nn.Module, input_size: tuple[int, int], output: Path, opset: int
):
    height, width = input_size
    image = torch.zeros(1, 3, height, width)
    with torch.no_grad():
        embedding = model(image)
    print(f"embedding shape {tuple(embedding.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    traced = output.with_name(output.stem + ".traced.onnx")
    torch.onnx.export(
        model,
        (image,),
        str(traced),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
        **kwargs,
    )
    import onnx
    import onnxsim

    simplified, ok = onnxsim.simplify(
        onnx.load(str(traced)), overwrite_input_shapes={"input": [1, 3, height, width]}
    )
    if not ok:
        raise RuntimeError("onnxsim could not verify the simplified graph")
    onnx.save(simplified, str(output))
    traced.unlink()
    onnx.checker.check_model(str(output))
    print(f"Wrote {output}")


def export_model(
    *,
    name: str,
    work_dir: Path,
    output: Path,
    input_size: tuple[int, int],
    opset: int,
    weights: str,
    repo: str | None,
) -> None:
    if not repo:
        raise ValueError("--repo must name the InsightFace upstream repository")
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        Path(weights) if Path(weights).is_file() else download(weights, work_dir)
    )
    with tempfile.TemporaryDirectory(
        prefix="insightface-", dir="/tmp"
    ) as temporary_directory:
        checkout = _checkout(repo, Path(temporary_directory))
        try:
            export(build(checkout, name, checkpoint), input_size, output, opset)
        finally:
            if str(checkout) in sys.path:
                sys.path.remove(str(checkout))


def main(argv: list[str] | None = None) -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--name", required=True, help="Model ID; names the IResNet depth."
    )
    argument_parser.add_argument("--work-dir", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument(
        "--input-size", type=int, nargs=2, metavar=("H", "W"), required=True
    )
    argument_parser.add_argument("--opset", type=int, default=17)
    argument_parser.add_argument(
        "--weights",
        required=True,
        help="OneDrive share URL with '#path=<folder>/backbone.pth&sha256=<hex>'.",
    )
    argument_parser.add_argument("--repo", default=None, help="InsightFace git URL.")
    args = argument_parser.parse_args(argv)
    export_model(
        name=args.name,
        work_dir=args.work_dir,
        output=args.output,
        input_size=(args.input_size[0], args.input_size[1]),
        opset=args.opset,
        weights=args.weights,
        repo=args.repo,
    )


if __name__ == "__main__":
    main()


def export_onnx(
    customer_contract: dict[str, Any], *, weights: str, output: Path, opset: int
) -> None:
    """Export in an isolated worker with its own temporary upstream checkout."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="model-export-") as directory:
        height, width = customer_contract["input"]["shape"][2:]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--name",
                customer_contract["model"],
                "--work-dir",
                directory,
                "--output",
                str(output),
                "--input-size",
                str(height),
                str(width),
                "--opset",
                str(opset),
                "--weights",
                weights,
                "--repo",
                "https://github.com/deepinsight/insightface",
            ],
            check=True,
        )
