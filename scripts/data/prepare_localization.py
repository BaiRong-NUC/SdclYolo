"""Prepare train-only localization actions using the frozen five-view detector."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.training.localization import DATA, prepare_localization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default=DATA)
    parser.add_argument("--limit", type=int, help="Random train-only smoke subset; not formal.")
    print(json.dumps(prepare_localization(**vars(parser.parse_args())), indent=2))
