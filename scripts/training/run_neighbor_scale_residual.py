"""Run the fixed matched regret-residual context pilot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.training.neighbor_scale_residual import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--device", default="0")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
