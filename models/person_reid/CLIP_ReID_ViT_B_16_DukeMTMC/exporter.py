# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "torch==2.7.1",
#   "onnx==1.17.0",
#   "onnxsim==0.4.36",
#   "onnxruntime==1.20.1",
#   "gdown==5.2.0",
#   "numpy<2",
# ]
# ///
"""Export a CLIP-ReID person re-identification checkpoint to ONNX.

The checkpoint (a Google Drive file from the upstream README's model table) holds the
fine-tuned CLIP image encoder with its position embedding already resized to the ReID
input, so no OpenAI CLIP base weights are needed. The upstream repository is cloned at a
pinned commit for its ``model/clip/model.py`` architecture definitions only; nothing else
from it is imported.

The exported graph is the test-time feature upstream evaluates with
(``TEST.NECK_FEAT: before``): the concatenation of the backbone feature and the CLIP
projection, before the BNNeck. L2 normalization stays on the Model side at prediction
time. Camera-aware (SIE) checkpoints get a second input, a camera one-hot vector, and the
``SIE_COE * cv_embed`` table becomes a bias-free linear layer inside the graph.

The graph is exported with a static batch of one and constant-folded with onnxsim: the
traced transformer otherwise carries a hundred Shape/Gather/Reshape chains from
``nn.MultiheadAttention``, and the compiler routed the calibration image into a
LayerNorm on that dynamic graph.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

UPSTREAM_COMMIT = "eb1898b72c882875f478bebfc6d41644eece0a5d"
CLONE_NAME = "CLIP-ReID"
# configs/person/vit_clipreid.yml runs SIE-OLP with `MODEL.SIE_COE 1.0`; the published
# SIE-OLP test logs record the same value.
SIE_COE = 1.0
PATCH_STRIDES = (16, 12, 8)


def _drive_id(weights: str) -> str:
    match = re.search(r"/d/([\w-]+)", weights) or re.search(r"id=([\w-]+)", weights)
    if match:
        return match.group(1)
    if re.fullmatch(r"[\w-]{20,}", weights):
        return weights
    raise ValueError(f"weights must be a Google Drive file URL or id: {weights!r}")


def _checkout(repo: str, work_dir: Path) -> Path:
    """A clone of ``repo`` at UPSTREAM_COMMIT with no local edits or stray files.

    A reused clone is reset hard and cleaned, then HEAD and ``git status`` are verified,
    so the code imported below is exactly the pinned revision.
    """
    target = work_dir / CLONE_NAME
    if not (target / ".git").is_dir():
        subprocess.run(["git", "clone", "--quiet", repo, str(target)], check=True)
    git = ["git", "-C", str(target)]
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
    return target


def _weights_reference(weights: str) -> tuple[str, str]:
    """Split ``<drive url or id>#sha256=<hex>`` into the Drive file id and its digest."""
    location, _, fragment = weights.partition("#")
    match = re.fullmatch(r"sha256=([0-9a-f]{64})", fragment)
    if match is None:
        raise ValueError(
            "weights must carry the checkpoint digest as '#sha256=<64 hex>'; the export "
            "input records it so a cached or re-downloaded file is verified before use"
        )
    return _drive_id(location), match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(weights: str, work_dir: Path) -> Path:
    """The checkpoint at the recorded digest, downloading (again) when the cache differs."""
    import gdown

    file_id, expected = _weights_reference(weights)
    target = work_dir / f"{file_id}.pth"
    for attempt in range(2):
        if target.is_file():
            actual = _sha256(target)
            if actual == expected:
                return target
            print(f"cached {target.name} sha256 {actual} != {expected}; re-downloading")
            target.unlink()
        if attempt == 0:
            gdown.download(id=file_id, output=str(target), quiet=False)
    raise RuntimeError(
        f"checkpoint {file_id} does not match the recorded sha256 {expected}"
    )


def _clip_module(checkout: Path) -> Any:
    """Import only upstream architecture definitions from the pinned checkout.

    Import the file as a top-level module, not ``model.clip``: that package imports
    upstream training helpers and extra dependencies unrelated to image export.
    The dedicated export worker owns this import namespace; evict any cached
    ``model`` first so a reused interpreter cannot select a different checkout.
    """
    architecture_dir = str((checkout / "model" / "clip").resolve())
    for name in tuple(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    sys.path.insert(0, architecture_dir)
    try:
        import model

        if Path(model.__file__).resolve() != Path(architecture_dir) / "model.py":
            raise RuntimeError(
                "CLIP architecture did not load from the pinned checkout"
            )
        return model
    finally:
        sys.path.remove(architecture_dir)


def _encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "image_encoder."
    encoder = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not encoder:
        raise ValueError("checkpoint holds no image_encoder.* tensors")
    return encoder


def _vit_geometry(
    tokens: int, patch: int, input_size: tuple[int, int]
) -> tuple[int, int, int]:
    height, width = input_size
    for stride in PATCH_STRIDES:
        h_res = (height - patch) // stride + 1
        w_res = (width - patch) // stride + 1
        if h_res * w_res == tokens:
            return h_res, w_res, stride
    raise ValueError(
        f"no patch stride in {PATCH_STRIDES} yields {tokens} tokens for {input_size}"
    )


class ReidEncoder(nn.Module):
    """Upstream ``build_transformer`` test path without the BNNeck and classifier."""

    def __init__(self, visual: nn.Module, vit: bool, camera_table: torch.Tensor | None):
        super().__init__()
        self.visual = visual
        self.vit = vit
        self.camera = None
        if camera_table is not None:
            cameras, width = camera_table.shape
            self.camera = nn.Linear(cameras, width, bias=False)
            with torch.no_grad():
                self.camera.weight.copy_((SIE_COE * camera_table).T)

    def _vit_forward(self, image: torch.Tensor, camera: torch.Tensor | None):
        v = self.visual
        x = v.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, x.shape[-1])
        if self.camera is not None:
            if camera is None:
                raise ValueError("this Model needs a camera one-hot input")
            cls = cls + self.camera(camera).unsqueeze(1)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x).permute(1, 0, 2)
        x = v.transformer(x).permute(1, 0, 2)
        x12 = v.ln_post(x)
        proj = x12 @ v.proj
        return torch.cat([x12[:, 0], proj[:, 0]], dim=1)

    def _rn50_forward(self, image: torch.Tensor):
        _x3, x4, proj = self.visual(image)
        # Upstream global-average-pools x4 over its full spatial extent; the mean is the
        # same operation without the dynamic kernel-size shape gather ONNX cannot trace.
        feat = x4.mean(dim=(2, 3))
        return torch.cat([feat, proj[0]], dim=1)

    def forward(self, image: torch.Tensor, camera: torch.Tensor | None = None):
        if self.vit:
            return self._vit_forward(image, camera)
        return self._rn50_forward(image)


def build(
    state: dict[str, torch.Tensor], clip: Any, input_size: tuple[int, int]
) -> ReidEncoder:
    encoder = _encoder_state(state)
    vit = "proj" in encoder
    if vit:
        width = encoder["conv1.weight"].shape[0]
        patch = encoder["conv1.weight"].shape[-1]
        layers = len(
            {k.split(".")[2] for k in encoder if k.startswith("transformer.resblocks.")}
        )
        tokens = encoder["positional_embedding"].shape[0] - 1
        h_res, w_res, stride = _vit_geometry(tokens, patch, input_size)
        visual = clip.VisionTransformer(
            h_resolution=h_res,
            w_resolution=w_res,
            patch_size=patch,
            stride_size=stride,
            width=width,
            layers=layers,
            heads=width // 64,
            output_dim=encoder["proj"].shape[1],
        )
        print(f"ViT: {layers} layers, width {width}, patch {patch}, stride {stride}")
    else:
        counts = tuple(
            len({k.split(".")[1] for k in encoder if k.startswith(f"layer{b}.")})
            for b in (1, 2, 3, 4)
        )
        width = encoder["layer1.0.conv1.weight"].shape[0]
        spatial = encoder["attnpool.positional_embedding"].shape[0] - 1
        visual = clip.ModifiedResNet(
            layers=counts,
            output_dim=encoder["attnpool.c_proj.weight"].shape[0],
            heads=width * 32 // 64,
            input_resolution=spatial,
            width=width,
        )
        print(
            f"ModifiedResNet: layers {counts}, width {width}, pooled tokens {spatial}"
        )
    visual.load_state_dict(encoder, strict=True)
    camera = state.get("cv_embed")
    if camera is not None and not vit:
        raise ValueError("camera embedding is defined for the ViT encoder only")
    return ReidEncoder(
        visual.float(), vit, camera.float() if camera is not None else None
    ).eval()


def export(model: ReidEncoder, input_size: tuple[int, int], output: Path, opset: int):
    height, width = input_size
    image = torch.zeros(1, 3, height, width)
    inputs: tuple[torch.Tensor, ...] = (image,)
    input_names = ["input"]
    if model.camera is not None:
        camera = torch.zeros(1, model.camera.in_features)
        camera[0, 0] = 1.0
        inputs = (image, camera)
        input_names.append("camera")
    with torch.no_grad():
        embedding = model(*inputs)
    print(f"embedding shape {tuple(embedding.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    traced = output.with_name(output.stem + ".traced.onnx")
    torch.onnx.export(
        model,
        inputs,
        str(traced),
        input_names=input_names,
        output_names=["embedding"],
        opset_version=opset,
        **kwargs,
    )
    import onnx
    import onnxsim

    simplified, ok = onnxsim.simplify(
        onnx.load(str(traced)),
        overwrite_input_shapes={
            name: list(tensor.shape) for name, tensor in zip(input_names, inputs)
        },
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
        raise ValueError("--repo must name the CLIP-ReID upstream repository")
    work_dir.mkdir(parents=True, exist_ok=True)
    checkout = _checkout(repo, work_dir)
    checkpoint = (
        Path(weights) if Path(weights).is_file() else _download(weights, work_dir)
    )
    state = torch.load(checkpoint, map_location="cpu")
    model = build(state, _clip_module(checkout), input_size)
    export(model, input_size, output, opset)


def main(argv: list[str] | None = None) -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--name", required=True, help="Model directory name.")
    argument_parser.add_argument("--work-dir", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument(
        "--input-size", type=int, nargs=2, metavar=("H", "W"), required=True
    )
    argument_parser.add_argument("--opset", type=int, default=17)
    argument_parser.add_argument(
        "--weights",
        required=True,
        help="Google Drive file URL or id of the checkpoint.",
    )
    argument_parser.add_argument("--repo", default=None, help="CLIP-ReID git URL.")
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
                "https://github.com/Syliz517/CLIP-ReID",
            ],
            check=True,
        )
