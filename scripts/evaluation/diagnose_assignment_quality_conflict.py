"""Run the fixed cached supervised-classification feedback screen."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.assignment_quality_conflict import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    run(**vars(parser.parse_args()))
