"""Official evaluation of fixed coordinate unanimity using existing predictions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.coordinate_consensus import CACHE, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=2)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
