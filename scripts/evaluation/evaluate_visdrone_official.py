"""Prepare exact controls, then run the pinned official VisDrone MATLAB toolkit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

from sdcl.evaluation.official_visdrone import (
    ARMS, collect_results, prepare_official, run_official,
)
from sdcl.evaluation.official_runtime import (
    build_runtime, run_accelerated, validate_runtime,
)
from sdcl.evaluation.official_details import audit_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output")
    run = commands.add_parser("run")
    run.add_argument("--output", required=True)
    run.add_argument("--octave", required=True, help="Full path to octave-cli.exe")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--accelerated", action="store_true",
                     help="Use the native helpers after reference equivalence passes")
    for name in ("build-runtime", "verify-runtime"):
        runtime = commands.add_parser(name)
        runtime.add_argument("--output", required=True)
        runtime.add_argument("--octave", required=True)
    audit = commands.add_parser("audit-classes")
    audit.add_argument("--output", required=True)
    audit.add_argument("--octave", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"output": str(prepare_official(args.output))}, indent=2))
    elif args.command == "run":
        evaluator = run_accelerated if args.accelerated else run_official
        print(json.dumps(evaluator(args.output, args.octave, args.arm), indent=2))
    elif args.command == "build-runtime":
        print(json.dumps({"runtime": str(build_runtime(args.output, args.octave))}, indent=2))
    elif args.command == "verify-runtime":
        print(json.dumps(validate_runtime(args.output, args.octave), indent=2))
    elif args.command == "audit-classes":
        result = audit_details(args.output, args.octave)
        print(json.dumps({"output": args.output, "all_headline_metrics_reproduced":
                         all(arm["headline_reproduction_max_error"] <= 1e-12
                             for arm in result["arms"].values())}, indent=2))
    else:
        print(json.dumps(collect_results(args.output), indent=2))
