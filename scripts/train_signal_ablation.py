"""Train scale and difficulty signal ablations sequentially, without extra logs."""

import argparse
import subprocess
import sys
from pathlib import Path

if not __package__:
    import _bootstrap

from sdcl.config import PROJECT_ROOT, load_experiment


CONFIGS = ("sdcl_scale_100.yaml", "sdcl_difficulty_100.yaml")


def build_commands():
    commands = []
    for name in CONFIGS:
        config = PROJECT_ROOT / "configs" / "experiments" / name
        train, _, _ = load_experiment(config)
        run_dir = Path(train["project"]) / train["name"]
        if run_dir.exists():
            raise FileExistsError(
                f"Output already exists: {run_dir}. "
                "Inspect the run and use its individual training/resume command."
            )
        commands.append((train["name"], [
            sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config", str(config),
        ]))
    for required in ("yolo11s.pt", "data/visdrone/dataset.yaml"):
        if not (PROJECT_ROOT / required).is_file():
            raise FileNotFoundError(f"Required input is missing: {PROJECT_ROOT / required}")
    return commands


def run_queue(commands):
    for index, (name, command) in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] Starting {name}", flush=True)
        # Inherit the terminal so YOLO renders its own progress bar.
        result = subprocess.run(command, cwd=PROJECT_ROOT).returncode
        if result != 0:
            print(f"STOPPED: {name} exited with code {result}.", file=sys.stderr, flush=True)
            return result
        print(f"[{index}/{len(commands)}] Completed {name}", flush=True)
    print("\nBoth signal ablations completed.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs and print commands without training or writing files.")
    args = parser.parse_args(argv)
    try:
        commands = build_commands()
        if args.dry_run:
            for index, (name, command) in enumerate(commands, start=1):
                print(f"{index}. {name}: {subprocess.list2cmdline(command)}")
            print("Preflight passed. No training started.")
            return 0
        return run_queue(commands)
    except KeyboardInterrupt:
        print("\nInterrupted. No later experiment will start.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"Queue stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
