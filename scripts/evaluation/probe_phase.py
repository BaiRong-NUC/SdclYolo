"""Fixed validation-subset diagnostic, not the normal-AP acceptance test."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.phase_probe import run_phase_probe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--images", type=int, default=64)
    args = parser.parse_args()
    result = run_phase_probe(**vars(args))
    print(json.dumps(result, indent=2))
