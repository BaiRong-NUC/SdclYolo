"""Evaluate three previously frozen controls once on all1610test-dev images."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from sdcl.evaluation.testdev_transfer import OUTPUT, prepare, run_official


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--limit", type=int, help="Smoke only; not a transfer decision.")
    parser.add_argument("--official-only", action="store_true")
    args = parser.parse_args()
    if not args.official_only:
        prepare(args.output, args.device, args.limit)
    print(json.dumps(run_official(args.output), indent=2))
