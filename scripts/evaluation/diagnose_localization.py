"""Evaluate epoch20 against ordinary calibration and three localization controls."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.localization_screen import diagnose_localization
from sdcl.training.localization import RUN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=f"{RUN}/last.pt")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Smoke subset; never yields a formal decision.")
    print(json.dumps(diagnose_localization(**vars(parser.parse_args())), indent=2))
