"""Run the fixed independent visual encoder information screen."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.external_content import OUTPUT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    r = run(**vars(parser.parse_args()))
    print(json.dumps(dict(output=r["output"], metrics=r["metrics"], decision=r["decision"],
                         training=r["training"], verification=r["verification"],
                         timing={k: v for k, v in r["timing"].items() if k != "batches"}), indent=2))
