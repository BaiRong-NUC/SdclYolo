"""Evaluate the frozen epoch-20 rescorer and matched controls against Soft-NMS."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.rescue_screen import diagnose_rescue
from sdcl.training.rescue import RUN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=f"{RUN}/last.pt")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Smoke only; never yields a formal decision.")
    print(json.dumps(diagnose_rescue(**vars(parser.parse_args())), indent=2))
