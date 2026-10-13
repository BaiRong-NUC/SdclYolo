"""Official evaluation of the final threshold heads using shared test-dev actions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.candidate_series import CACHE
from sdcl.evaluation.threshold_gain_probe import RUN, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--checkpoint", default=f"{RUN}/last.pt")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=2)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
