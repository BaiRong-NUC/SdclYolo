"""Evaluate the single frozen coverage-assignment candidate against 960."""

import _bootstrap
import argparse
import json

from sdcl.coverage_screen import run_coverage_screen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output")
    result = run_coverage_screen(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
