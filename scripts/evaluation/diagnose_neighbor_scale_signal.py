"""Run the registered no-fitting neighboring-object scale information screen."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.neighbor_scale_signal import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
