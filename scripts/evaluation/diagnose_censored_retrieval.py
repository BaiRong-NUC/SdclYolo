"""Run the fixed suppression-witness retrieval diagnostic without retraining."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.censored_retrieval import OUTPUT, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
