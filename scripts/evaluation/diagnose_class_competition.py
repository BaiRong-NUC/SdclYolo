"""Run candidate64 exactly once against saved ordinary-calibration predictions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.class_competition import run


if __name__ == "__main__":
    run()
