"""Run a frozen official coordinate audit without training or detector inference."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.localization_official_audit import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=3)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
