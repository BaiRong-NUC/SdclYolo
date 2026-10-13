"""Train all three fixed rescorers in one run; the detector remains frozen."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.training.rescue import DATA, RUN, train_rescue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--output", default=RUN)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(train_rescue(**vars(parser.parse_args())), indent=2))
