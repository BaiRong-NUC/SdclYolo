"""Run one cached-prediction headroom diagnostic; never trains or authorizes training."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import json

from sdcl.evaluation.headroom_screen import diagnose_headroom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Smoke subset; never yields a formal diagnosis.")
    print(json.dumps(diagnose_headroom(**vars(parser.parse_args())), indent=2))
