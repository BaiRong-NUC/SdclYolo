"""Run the fixed same-head whole-box versus four-side composition pilot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.training.side_composition_pilot import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--device", default="0")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
