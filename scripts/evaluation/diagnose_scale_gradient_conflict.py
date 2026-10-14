"""Run the preregistered scale-specific shared-gradient audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.scale_gradient_conflict import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--device", default="0")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
