"""Screen local magnification using the frozen 960 checkpoint; no training."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.slicing_screen import run_slicing_screen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Smoke test only; never yields CONTINUE/STOP.")
    print(json.dumps(run_slicing_screen(**vars(parser.parse_args())), indent=2))
