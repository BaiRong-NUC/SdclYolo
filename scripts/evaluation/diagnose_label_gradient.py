"""Run the preregistered cached CIoU label perturbation audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.label_gradient_sensitivity import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
