"""Run the four 100-epoch seed 1/2 experiments sequentially."""

import argparse
import codecs
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    import _bootstrap

from sdcl.config import PROJECT_ROOT, SDCLConfig, load_experiment, project_path


@dataclass
class Stage:
    name: str
    command: list[str]
    run_dir: Path
    action: str = "start"
    completed_epochs: int = 0


def recorded_epochs(run_dir):
    file = run_dir / "results.csv"
    if not file.is_file():
        return []
    with file.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames is None:
            return []
        if "epoch" not in reader.fieldnames:
            raise ValueError(f"Missing epoch column: {file}")
        epochs = [float(row["epoch"]) for row in reader]
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"Epoch records are incomplete or duplicated: {file}")
    return epochs


def validate_training_args(saved, expected, run_dir):
    for key, value in expected.items():
        if key == "model":
            # Resuming replaces model with last.pt in the saved training arguments.
            if saved.get("resume"):
                if project_path(saved.get("model", "")) != run_dir / "weights" / "last.pt":
                    raise ValueError(f"Unexpected resume model in {run_dir}")
                continue
            matches = project_path(saved.get(key, "")) == project_path(value)
        elif key in {"data", "project"}:
            matches = project_path(saved.get(key, "")) == project_path(value)
        elif key == "device":
            matches = str(saved.get(key)) == str(value)
        else:
            matches = saved.get(key) == value
        if not matches:
            raise ValueError(f"Saved training setting differs: {key} in {run_dir}")


def load_checkpoint(file):
    import torch

    try:
        # These are checkpoints produced by this local project.
        return torch.load(file, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Cannot read checkpoint {file}: {error}") from error


def inspect_run(run_dir, train, settings, ignore):
    if not run_dir.exists():
        return "start", 0
    manifest_file = run_dir / "experiment.json"
    if not manifest_file.is_file():
        raise ValueError(f"Cannot verify existing run without experiment.json: {run_dir}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (SDCLConfig.from_dict(manifest.get("sdcl")) != settings
            or manifest.get("ignore_regions") != ignore):
        raise ValueError(f"Saved method or ignore protocol differs: {run_dir}")
    validate_training_args(manifest.get("train", {}), train, run_dir)
    epochs = recorded_epochs(run_dir)
    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        if epochs or any(run_dir.rglob("*.pt")):
            raise ValueError(f"Training artifacts exist but last.pt is missing: {run_dir}")
        return "restart", 0
    checkpoint = load_checkpoint(last)
    validate_training_args(checkpoint.get("train_args", {}), train, run_dir)
    epoch = checkpoint.get("epoch", -1)
    if len(epochs) == train["epochs"] and epoch in {-1, train["epochs"] - 1}:
        best = run_dir / "weights" / "best.pt"
        if not best.is_file() or best.stat().st_size == 0:
            raise ValueError(f"Completed run is missing best.pt: {run_dir}")
        return "skip", len(epochs)
    completed = epoch + 1
    if (not 0 < completed < train["epochs"]
            or checkpoint.get("optimizer") is None or checkpoint.get("ema") is None):
        raise ValueError(f"Checkpoint cannot resume an unfinished run: {last}")
    if len(epochs) != completed:
        raise ValueError(f"results.csv and last.pt disagree about saved progress: {run_dir}")
    return "resume", completed


def archive_unstarted_run(run_dir):
    root = (PROJECT_ROOT / "output" / "runs").resolve()
    source = run_dir.resolve()
    target = root / f"{run_dir.name}_aborted_{datetime.now():%Y%m%d-%H%M%S-%f}"
    if source.parent != root or target.resolve().parent != root:
        raise ValueError(f"Archive paths must stay directly inside {root}")
    if recorded_epochs(source) or any(source.rglob("*.pt")):
        raise ValueError(f"Run has training artifacts and cannot be restarted: {source}")
    if target.exists():
        raise FileExistsError(f"Archive already exists: {target}")
    source.rename(target)
    print(f"Preserved interrupted setup: {target}", flush=True)
    return target


def build_stages():
    stages = []
    for seed in (1, 2):
        for method in ("baseline", "sdcl"):
            config = PROJECT_ROOT / "configs" / "experiments" / f"{method}_100_seed{seed}.yaml"
            train, settings, ignore = load_experiment(config)
            run_dir = (Path(train["project"]) / train["name"]).resolve()
            action, completed = inspect_run(run_dir, train, settings, ignore)
            command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "train.py"),
                       "--config", str(config)]
            if action == "resume":
                command.extend(["--resume", str(run_dir / "weights" / "last.pt")])
            stages.append(Stage(f"{method}_seed{seed}", command, run_dir, action, completed))
    for required in ("yolo11s.pt", "data/visdrone/dataset.yaml"):
        if not (PROJECT_ROOT / required).is_file():
            raise FileNotFoundError(f"Required input is missing: {PROJECT_ROOT / required}")
    return stages


def run_command(command, log_file):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with log_file.open("w", encoding="utf-8", newline="") as log:
        log.write(f"Command: {subprocess.list2cmdline(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            # Text-mode pipes turn progress-bar carriage returns into newlines.
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while chunk := process.stdout.read1(4096):
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(text)
                log.flush()
            tail = decoder.decode(b"", final=True)
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
                log.write(tail)
                log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def run_queue(stages, log_root):
    log_root.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {log_root}", flush=True)
    for index, stage in enumerate(stages, start=1):
        name, command = stage.name, stage.command
        if stage.action == "skip":
            print(f"\n[{index}/{len(stages)}] Skipping completed {name} "
                  f"({stage.completed_epochs} epochs)", flush=True)
            continue
        if stage.action == "restart":
            archive_unstarted_run(stage.run_dir)
        elif stage.action == "start" and stage.run_dir.exists():
            raise FileExistsError(f"Run appeared after preflight: {stage.run_dir}")
        log_file = log_root / f"{index:02d}_{name}.log"
        print(f"\n[{index}/{len(stages)}] {stage.action.upper()} {name} "
              f"(saved epochs: {stage.completed_epochs})", flush=True)
        print(f"Log: {log_file}", flush=True)
        result = run_command(command, log_file)
        if result != 0:
            print(f"STOPPED: {name} exited with code {result}. No later stage will start.",
                  file=sys.stderr, flush=True)
            return result
        print(f"[{index}/{len(stages)}] Completed {name}", flush=True)
    print("\nAll four experiments completed (including previously completed runs).", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs/output paths and print commands without training.")
    args = parser.parse_args(argv)
    try:
        stages = build_stages()
        if args.dry_run:
            for index, stage in enumerate(stages, start=1):
                print(f"{index}. {stage.name}: {stage.action.upper()} "
                      f"(saved epochs: {stage.completed_epochs})")
                if stage.action != "skip":
                    print(f"   {subprocess.list2cmdline(stage.command)}")
            print("Preflight passed. No training started.")
            return 0
        log_root = PROJECT_ROOT / "output" / "queue_logs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return run_queue(stages, log_root)
    except KeyboardInterrupt:
        print("\nInterrupted. No later stage will start.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"Queue stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
