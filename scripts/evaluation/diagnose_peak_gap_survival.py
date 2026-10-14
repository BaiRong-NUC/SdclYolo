"""Run frozen downstream-support audit; no detector inference or training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.peak_gap_survival import run


if __name__ == "__main__":
    run()
