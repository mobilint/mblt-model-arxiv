#!/usr/bin/env python3
"""Run an image stream through qbruntime's bounded asynchronous inference pipeline."""

from inference_stream import stream_inference
from package_common import exit_with_package_error


def main() -> None:
    stream_inference(asynchronous=True)


if __name__ == "__main__":
    exit_with_package_error(main)
