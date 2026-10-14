"""Evaluate every registered stable-order arm with the pinned official toolkit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.stable_order_official import OUTPUT, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
