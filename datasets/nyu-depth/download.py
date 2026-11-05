"""Standard-library download helpers shared by dataset preparation scripts."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def is_url(source: str) -> bool:
    """Return whether source is an HTTP(S) URL."""

    return urllib.parse.urlparse(source).scheme in {"http", "https"}


def format_bytes(byte_count: int) -> str:
    """Return a compact, human-readable byte count."""

    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _stream_download(source: str, partial: Path) -> None:
    """Stream a URL into partial, resuming it when the server supports ranges."""

    existing_size = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mobilint-dataset-preparer/1.0"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"Resuming {partial.name} from {format_bytes(existing_size)}")
    else:
        print(f"Downloading {source} to {partial}")

    request = urllib.request.Request(source, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        append = existing_size > 0 and status == 206
        if existing_size and not append:
            print("Server did not accept the resume request; restarting the download.")
        mode = "ab" if append else "wb"
        initial_size = existing_size if append else 0
        content_length = response.headers.get("Content-Length")
        remaining_size = int(content_length) if content_length else None
        expected_size = (
            initial_size + remaining_size if remaining_size is not None else None
        )
        written_size = initial_size
        report_size = written_size
        reported_progress = False
        started_at = time.monotonic()

        with partial.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                written_size += len(chunk)
                now = time.monotonic()
                if (
                    written_size - report_size >= 16 * 1024 * 1024
                    or now - started_at >= 2
                ):
                    elapsed = max(now - started_at, 0.001)
                    progress = format_bytes(written_size)
                    if expected_size is not None:
                        progress += f" / {format_bytes(expected_size)}"
                    print(
                        f"\r{partial.name}: {progress} "
                        f"({format_bytes(int((written_size - initial_size) / elapsed))}/s)",
                        end="",
                        flush=True,
                    )
                    report_size = written_size
                    reported_progress = True
        if reported_progress:
            print()
        if expected_size is not None and written_size != expected_size:
            raise ConnectionError(
                f"Download ended early: received {format_bytes(written_size)}, "
                f"expected {format_bytes(expected_size)}."
            )


def download(source: str, download_dir: Path, retries: int = 3) -> Path:
    """Download a URL with resume, progress reporting, and transient retries."""

    filename = Path(urllib.parse.urlparse(source).path).name
    if not filename:
        raise ValueError(f"Cannot determine a filename from URL: {source}")
    destination = download_dir / filename
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using existing download: {destination}")
        return destination

    download_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(retries + 1):
        try:
            _stream_download(source, partial)
            partial.replace(destination)
            return destination
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == retries:
                raise
            retry_error: Exception = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt == retries:
                raise
            retry_error = error
        delay = 2**attempt
        print(
            f"Download interrupted ({retry_error}); retrying in {delay} seconds "
            f"({attempt + 1}/{retries})."
        )
        time.sleep(delay)
    raise RuntimeError("Download failed after all retry attempts.")


def resolve_source(source: str, download_dir: Path, retries: int = 3) -> Path:
    """Resolve a local path or download URL to a local path."""

    if is_url(source):
        return download(source, download_dir, retries)
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source does not exist: {path}")
    return path
