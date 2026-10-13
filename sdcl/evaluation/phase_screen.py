"""Fixed one-candidate investment screen against the completed 960 baseline."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path

from sdcl.config import SDCLConfig, load_phase_config, project_path
from sdcl.training.phase import PhaseConfig
from sdcl.evaluation.phase_probe import run_phase_probe
from sdcl.evaluation.resolution import GATES, decide_resolution
from sdcl.evaluation.size_analysis import file_hash, run_size_analysis
from sdcl.runtime.environment import package_source_hashes


BASELINE = "output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt"
CANDIDATE = "output/runs/phase_yolo11s_960_100ep_seed0/weights/best.pt"
BASELINE_SHA256 = "eac88f7e348134d3bdaf8b8d3c93845002b634b66a4fa28bd6246abbbeba1a25"
PHASE_GATES = {"tiny_feature_distance_reduction_pct": 10.0,
               "tiny_hit_flip_reduction_pp": 0.5}
SCREEN_CONFIG = PhaseConfig(enabled=True)


def check_phase_training_pair(baseline, candidate, verify_checkpoints=True):
    manifests = []
    for checkpoint in (baseline, candidate):
        checkpoint = project_path(checkpoint)
        run = checkpoint.parent.parent
        manifest = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
        train = manifest["train"]
        if SDCLConfig.from_dict(manifest["sdcl"]).enabled or not manifest["ignore_regions"]:
            raise ValueError("Both runs must disable SDCL and enable ignore regions.")
        if (train["imgsz"], train["batch"], train["epochs"], train["seed"]) != (960, 8, 100, 0):
            raise ValueError("Requires 960 / batch 8 / 100 epochs / seed 0.")
        if train["model"] != "yolo11s.pt" or train.get("resume"):
            raise ValueError("Requires uninterrupted training from yolo11s.pt.")
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in rows] != list(range(1, 101)):
            raise ValueError(f"Not a complete continuous 100-epoch run: {run}")
        manifests.append(manifest)
    first, second = manifests
    if PhaseConfig.from_dict(first.get("phase")).enabled:
        raise ValueError("Baseline must not use phase consistency.")
    expected = load_phase_config("configs/experiments/phase_960_100.yaml")
    if expected != SCREEN_CONFIG or PhaseConfig.from_dict(second.get("phase")) != SCREEN_CONFIG:
        raise ValueError("Candidate phase settings differ from the frozen screen.")
    allowed = {"name", "save_dir"}
    recipes = [{k: v for k, v in m["train"].items() if k not in allowed} for m in manifests]
    if recipes[0] != recipes[1]:
        raise ValueError("Training recipes differ beyond output name and phase objective.")
    if first["dataset_hashes"] != second["dataset_hashes"]:
        raise ValueError("Dataset hashes differ.")
    if first["packages"] != second["packages"]:
        raise ValueError("Package versions differ.")
    if SDCLConfig.from_dict(first["sdcl"]) != SDCLConfig.from_dict(second["sdcl"]):
        raise ValueError("SDCL settings differ.")
    current_hashes = package_source_hashes()
    if second["sdcl_source_hashes"] != current_hashes:
        raise ValueError("Candidate training source differs from current evaluation source.")
    if verify_checkpoints:
        if file_hash(project_path(baseline)) != BASELINE_SHA256:
            raise ValueError("Baseline checkpoint differs from the verified 960 baseline.")
        from ultralytics import YOLO
        model = YOLO(str(project_path(candidate))).model
        if PhaseConfig.from_dict(getattr(model, "phase_settings", None)) != expected:
            raise ValueError("Candidate checkpoint does not contain the expected phase settings.")
    log = project_path(candidate).parent.parent / "phase_statistics.jsonl"
    stats = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if len(stats) != 100 or [row["epoch"] for row in stats] != list(range(100)):
        raise ValueError("Missing complete phase training diagnostics.")
    if not all(row.get("active_steps", 0) > 0
               and row.get("mean_weighted_loss_all_batches", 0) > 0
               for row in stats if row["epoch"] >= 15):
        raise ValueError("Phase objective was inactive after ramp-up.")
    return {"baseline": first, "candidate": second}


def phase_deltas(first, second):
    for key in ("settings", "data_hashes", "source_hashes"):
        if first[key] != second[key]:
            raise ValueError(f"Probe {key} differ; comparison is invalid.")
    for group in ("tiny", "small", "medium", "large"):
        for key in ("objects", "comparisons"):
            if first["groups"][group][key] != second["groups"][group][key]:
                raise ValueError("Probe GT masks or counts differ.")
    base, candidate = first["groups"]["tiny"], second["groups"]["tiny"]
    distance = base["mean_cosine_distance"]
    if distance <= 0 or base["objects"] == 0 or base["comparisons"] == 0:
        raise ValueError("No measurable baseline tiny phase sensitivity.")
    deltas = {
        "tiny_feature_distance_reduction_pct":
            100 * (1 - candidate["mean_cosine_distance"] / distance),
        "tiny_hit_flip_reduction_pp": 100 * (base["hit_flip_rate"] - candidate["hit_flip_rate"]),
    }
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("Non-finite probe metrics.")
    return deltas


def decide_phase(report, baseline_probe, candidate_probe):
    result = decide_resolution(report)
    measured = phase_deltas(baseline_probe, candidate_probe)
    for key, minimum in PHASE_GATES.items():
        result["checks"].append({"metric": key, "value": measured[key],
                                 "minimum": minimum, "unit": "percent" if key.endswith("pct") else "pp",
                                 "comparison": ">=", "passed": measured[key] >= minimum})
    result["utility_passed"] = all(check["passed"] for check in result["checks"][:5])
    result["mechanism_passed"] = all(check["passed"] for check in result["checks"][5:])
    result["decision"] = "CONTINUE" if all(check["passed"] for check in result["checks"]) else "STOP"
    result["rule"] = "All five normal-detection gates and both phase-diagnostic gates must pass."
    result["scope"] = (
        "Single-seed investment screen for this phase objective. STOP retains the verified 960 baseline; "
        "it does not reject high-resolution small-object research. Passing is not proof of novelty, "
        "seed robustness, or that ownership masking is the causal component."
    )
    return result


def run_phase_screen(baseline=BASELINE, candidate=CANDIDATE, output=None, device="0", batch=8):
    baseline, candidate = project_path(baseline), project_path(candidate)
    manifests = check_phase_training_pair(baseline, candidate)
    if batch != 8:
        raise ValueError("Frozen evaluation uses batch 8.")
    output = project_path(output or f"output/analysis/phase-{datetime.now():%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite screen: {output}")
    output.mkdir(parents=True)
    plan = {"version": "2026-10-11-v1", "normal_detection_gates_pp": GATES,
            "phase_gates": PHASE_GATES, "training_manifests": manifests,
            "checkpoint_sha256": {"baseline": file_hash(baseline), "candidate": file_hash(candidate)},
            "phase_config": load_phase_config("configs/experiments/phase_960_100.yaml").to_dict(),
            "protocol": "960 vs 960 / reference 640 / diagnostic IoF-ignore metrics",
            "probe": "64 evenly indexed val images / four fixed shifts / float32 / batch 1"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    run_size_analysis(str(baseline), str(candidate), dataset="data/visdrone",
                      output=str(output / "normal"), device=device, batch=batch,
                      baseline_imgsz=960, sdcl_imgsz=960, reference_size=640)
    for name, checkpoint in (("baseline", baseline), ("candidate", candidate)):
        run_phase_probe(str(checkpoint), str(output / f"{name}_phase.json"), device=device)
    read = lambda file: json.loads(file.read_text(encoding="utf-8"))
    decision = decide_phase(read(output / "normal/report.json"),
                            read(output / "baseline_phase.json"),
                            read(output / "candidate_phase.json"))
    decision["reports"] = {"normal": str(output / "normal/report.json"),
                           "baseline_phase": str(output / "baseline_phase.json"),
                           "candidate_phase": str(output / "candidate_phase.json")}
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# Phase consistency investment screen", "",
             f"Decision: **{decision['decision']}**", "",
             "| Metric | Change | Required | Pass |", "|---|---:|---:|---|"]
    for check in decision["checks"]:
        value, minimum = check.get("value_pp", check.get("value")), check.get("minimum_pp", check.get("minimum"))
        unit = check.get("unit", "pp")
        lines.append(f"| {check['metric']} | {value:.4f} {unit} | "
                     f"{check['comparison']} {minimum:.4f} {unit} | {check['passed']} |")
    lines.extend(["", decision["scope"], "", "Diagnostic COCO-style metrics, NOT official VisDrone AP."])
    (output / "decision.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"output": str(output), **decision}
