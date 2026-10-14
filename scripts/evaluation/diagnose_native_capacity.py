"""Run the frozen native one-position capacity diagnostic."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.native_capacity import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(dict(output=result["output"], decision=result["decision"],
                         primary={h: m["argmax005"]["union"]["tiny"] for h, m in result["metrics"].items()},
                         verification=result["verification"]), indent=2))
