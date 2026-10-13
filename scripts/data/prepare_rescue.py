"""Prepare resumable train-only rescoring features with the frozen detector."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.training.rescue import DATA, prepare_rescue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default=DATA)
    parser.add_argument("--limit", type=int, help="Random train-only smoke subset; never a formal run.")
    print(json.dumps(prepare_rescue(**vars(parser.parse_args())), indent=2))
