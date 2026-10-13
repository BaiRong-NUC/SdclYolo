"""Train an ordinary prior-work-shaped control only after the seed screen passes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.training.prior_gap_confirmation import DATA, RUN, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--output", default=RUN)
    parser.add_argument("--device", default="0")
    print(json.dumps(train(**vars(parser.parse_args())), indent=2))
