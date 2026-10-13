"""Run the fixed standard fusion control; no detector training."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.fusion_screen import run_fusion_screen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Smoke check only; never yields CONTINUE/STOP.")
    print(json.dumps(run_fusion_screen(**vars(parser.parse_args())), indent=2))
