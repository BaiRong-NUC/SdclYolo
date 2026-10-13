"""Prepare shared frozen test-dev features, then confirm one registered candidate."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.evaluation.candidate_series import CACHE, CANDIDATES, evaluate, prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("prepare")
    cache.add_argument("--cache", default=CACHE)
    cache.add_argument("--device", default="0")
    cache.add_argument("--limit", type=int, help="Smoke only; cannot generate formal decisions.")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--candidate", required=True, choices=CANDIDATES)
    evaluation.add_argument("--cache", default=CACHE)
    evaluation.add_argument("--output")
    evaluation.add_argument("--workers", type=int, default=2)
    evaluation.add_argument("--reuse", help="Reuse verified MAT results from this exact cache.")
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = prepare(**args) if command == "prepare" else evaluate(**args)
    print(json.dumps(result, indent=2))
