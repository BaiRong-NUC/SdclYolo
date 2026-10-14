"""Run the registered frozen-context signal screen without training."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.context_intervention import OUTPUT, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
