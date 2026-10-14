"""Run the fixed CPU-only geometry prerequisite for a shared detector prefix."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.shared_prefix_lattice import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
