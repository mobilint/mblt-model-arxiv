#!/usr/bin/env python3
"""Run an image stream synchronously with one persistent model and bounded preprocessing."""

from inference_stream import stream_inference
from package_common import exit_with_package_error


def main() -> None:
    stream_inference(asynchronous=False)


if __name__ == "__main__":
    exit_with_package_error(main)
