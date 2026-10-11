"""Run the frozen investment screen after completing the one candidate training."""

import _bootstrap
import argparse
import json

from sdcl.phase_screen import BASELINE, CANDIDATE, run_phase_screen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(run_phase_screen(**vars(args)), indent=2))
