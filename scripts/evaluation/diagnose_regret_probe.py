"""Evaluate fixed epoch20 ranking heads against calibration, old gain and uniform ranking."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.regret_probe import diagnose_probe
from sdcl.training.regret_probe import RUN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=f"{RUN}/last.pt")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Smoke only; not a formal investment decision.")
    print(json.dumps(diagnose_probe(**vars(parser.parse_args())), indent=2))
