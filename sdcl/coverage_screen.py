"""Frozen 960-vs-960 investment screen for coverage-preserving assignment."""

import csv
from datetime import datetime
import json
from pathlib import Path

from .config import SDCLConfig, load_coverage_config, project_path
from .coverage import CoverageConfig
from .phase import PhaseConfig
from .phase_screen import BASELINE, BASELINE_SHA256
from .resolution import GATES, decide_resolution
from .size_analysis import file_hash, run_size_analysis


CANDIDATE = "output/runs/coverage_yolo11s_960_100ep_seed0/weights/best.pt"
SCREEN_CONFIG = CoverageConfig(enabled=True)


def check_coverage_training_pair(baseline, candidate, verify_checkpoints=True):
    manifests = []
    for checkpoint in (baseline, candidate):
        checkpoint = project_path(checkpoint)
        run = checkpoint.parent.parent
        manifest = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
        train = manifest["train"]
        if (train["imgsz"], train["batch"], train["epochs"], train["seed"]) != (960, 8, 100, 0):
            raise ValueError("Requires 960 / batch 8 / 100 epochs / seed 0.")
        if train["model"] != "yolo11s.pt" or train.get("resume"):
            raise ValueError("Requires uninterrupted training from yolo11s.pt.")
        if SDCLConfig.from_dict(manifest["sdcl"]).enabled or not manifest["ignore_regions"]:
            raise ValueError("Requires SDCL off and ignore regions on.")
        if PhaseConfig.from_dict(manifest.get("phase")).enabled:
            raise ValueError("Phase objective must be off.")
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in rows] != list(range(1, 101)):
            raise ValueError("Missing complete 100-epoch training.")
        manifests.append(manifest)
    first, second = manifests
    if CoverageConfig.from_dict(first.get("coverage")).enabled:
        raise ValueError("Baseline cannot use coverage repair.")
    if (load_coverage_config("configs/experiments/coverage_960_100.yaml") != SCREEN_CONFIG
            or CoverageConfig.from_dict(second.get("coverage")) != SCREEN_CONFIG):
        raise ValueError("Candidate settings differ from frozen screen.")
    recipes = [{k: v for k, v in m["train"].items() if k not in {"name", "save_dir"}}
               for m in manifests]
    if recipes[0] != recipes[1]:
        raise ValueError("Training recipes differ beyond output name and assignment.")
    for key in ("dataset_hashes", "packages", "sdcl"):
        if first[key] != second[key]:
            raise ValueError(f"Training {key} differ.")
    for filename, expected in second["dataset_hashes"].items():
        if file_hash(project_path(filename)) != expected:
            raise ValueError("Current dataset differs from training dataset.")
    hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if second["sdcl_source_hashes"] != hashes:
        raise ValueError("Candidate training source differs from evaluation source.")
    stats = [json.loads(line) for line in
             (project_path(candidate).parent.parent / "coverage_statistics.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    if [row["epoch"] for row in stats] != list(range(100)):
        raise ValueError("Missing complete coverage diagnostics.")
    active = stats[SCREEN_CONFIG.warmup_epochs:]
    if any(row.get("batches", 0) <= 0 or row.get("rescued_gt", 0) <= 0 for row in active):
        raise ValueError("Coverage repair was inactive after warmup; invalid experiment.")
    if any(row.get("donor_starvation", 0) != 0 for row in active):
        raise ValueError("Donor coverage invariant failed; invalid experiment.")
    if any(row["small_uncovered_after"] != row["small_uncovered_before"] - row["rescued_gt"]
           for row in active):
        raise ValueError("Coverage diagnostics are inconsistent.")
    if verify_checkpoints:
        if file_hash(project_path(baseline)) != BASELINE_SHA256:
            raise ValueError("Baseline differs from verified 960 checkpoint.")
        from ultralytics import YOLO
        model = YOLO(str(project_path(candidate))).model
        if CoverageConfig.from_dict(getattr(model, "coverage_settings", {})) != SCREEN_CONFIG:
            raise ValueError("Candidate checkpoint lacks frozen coverage settings.")
        if model.model[-1].reg_max != 16 or list(model.stride) != [8, 16, 32]:
            raise ValueError("Candidate changed the inference head.")
    before = sum(row["small_uncovered_before"] for row in stats[-10:])
    rescued = sum(row["rescued_gt"] for row in stats[-10:])
    return {"baseline": first, "candidate": second, "mechanism": {
        "last10_small_uncovered_before": before, "last10_rescued_gt": rescued,
        "last10_uncovered_reduction_pct": 100 * rescued / max(1, before),
        "donor_starvation": 0,
        "scope": "Same-logit pre/post assignment counters; not detection efficacy."}}


def run_coverage_screen(baseline=BASELINE, candidate=CANDIDATE, output=None, device="0", batch=8):
    baseline, candidate = project_path(baseline), project_path(candidate)
    manifests = check_coverage_training_pair(baseline, candidate)
    if batch != 8:
        raise ValueError("Frozen evaluation uses batch 8.")
    output = project_path(output or f"output/analysis/coverage-{datetime.now():%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite screen: {output}")
    output.mkdir(parents=True)
    plan = {"version": "2026-10-12-v1", "gates_pp": GATES,
            "coverage_config": SCREEN_CONFIG.to_dict(), "training_manifests": manifests,
            "checkpoint_sha256": {"baseline": file_hash(baseline), "candidate": file_hash(candidate)},
            "protocol": "960 vs 960 / reference 640 / diagnostic IoF-ignore metrics"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    run_size_analysis(str(baseline), str(candidate), dataset="data/visdrone",
                      output=str(output / "normal"), device=device, batch=batch,
                      baseline_imgsz=960, sdcl_imgsz=960, reference_size=640)
    report = json.loads((output / "normal/report.json").read_text(encoding="utf-8"))
    decision = decide_resolution(report)
    decision["rule"] = "All five utility gates must pass for this coverage-assignment candidate."
    decision["scope"] = (
        "Single-seed investment screen. STOP stops this assignment candidate and retains "
        "the verified 960 baseline. Passing is not proof of novelty, causal components, "
        "or training-seed robustness.")
    decision["mechanism"] = manifests["mechanism"]
    decision["report"] = str(output / "normal/report.json")
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# Coverage assignment investment screen", "",
             f"Decision: **{decision['decision']}**", "",
             "| Metric | Change (pp) | Required (pp) | Pass |", "|---|---:|---:|---|"]
    for check in decision["checks"]:
        lines.append(f"| {check['metric']} | {check['value_pp']:.4f} | "
                     f"{check['comparison']} {check['minimum_pp']:.4f} | {check['passed']} |")
    lines.extend(["", decision["scope"], "",
                  "Mechanism: " + json.dumps(decision["mechanism"]), "",
                  "COCO-style diagnostics with IoF-ignore filtering, NOT official VisDrone AP."])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), **decision}
