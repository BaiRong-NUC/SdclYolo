"""Run preregistered small classification-projection updates."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.training.crossview_quality_rank import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    run(**vars(parser.parse_args()))
