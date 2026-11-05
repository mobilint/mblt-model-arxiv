"""Path-independent prepared-source content hashing."""

import hashlib
from pathlib import Path


def source_content_sha256(root: Path, files: list[Path]) -> str:
    """One reproducible digest for a prepared source file set, computed only at prepare time."""
    digest = hashlib.sha256()
    root = root.resolve()
    if any(
        not path.is_relative_to(root)
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
        for path in files
    ):
        raise ValueError(
            f"Source file escapes the source root or is missing under {root}"
        )
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        read_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                read_size += len(chunk)
        if read_size != size or path.stat().st_size != size:
            raise ValueError(f"Source file changed while hashing: {path}")
    return digest.hexdigest()
