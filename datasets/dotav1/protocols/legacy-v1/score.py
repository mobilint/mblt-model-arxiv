"""DOTAv1 historical difficulty-ignoring scoring protocol."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "score":
        arguments += ["--protocol", "dotav1-ignore-difficult.v1"]
    raise SystemExit(subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "score.py"), *arguments],
        check=False,
    ).returncode)
