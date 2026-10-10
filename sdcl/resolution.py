"""One additional training run, with fixed gates for a resolution research screen."""

import csv
from datetime import datetime
import json
import math

from .config import SDCLConfig, project_path
from .size_analysis import file_hash, run_size_analysis


# Project investment gates, fixed before running the new experiment.
# These are percentage points, not universal significance thresholds.
GATES = {
    "tiny_ap50_95_pp": 0.5,
    "small_all_ap50_95_pp": 0.3,
    "all_ap50_95_pp": -0.1,
    "small_all_recall_fppi10_pp": 0.5,
    "small_all_recall_fppi10_ci_low_pp": 0.0,
}


def resolution_deltas(report):
    if report["reference_size"] != 640:
        raise ValueError("Resolution decision requires fixed 640 size groups.")
    matched = next(item for item in report["matched_fp_budgets"] if item["label"] == "fppi_10")
    values = {
        f"{group}_ap50_95_pp": 100 * (
            report["coco_reference"]["sdcl"][group]["ap50_95"]
            - report["coco_reference"]["baseline"][group]["ap50_95"]
        )
        for group in ("tiny", "small_all", "all")
    }
    values["small_all_recall_fppi10_pp"] = matched["delta_recall_pp"]["small_all"]
    values["small_all_recall_fppi10_ci_low_pp"] = (
        matched["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0]
    )
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite decision metrics.")
    return values


def decide_resolution(report):
    sizes = report["inference_sizes"]
    if sizes != {"baseline": 960, "sdcl": 960}:
        raise ValueError("Training decision requires both checkpoints evaluated at 960.")
    settings = [report["captures"][name]["settings"] for name in ("baseline", "sdcl")]
    if any(settings[0].get(key) != settings[1].get(key)
           for key in set(settings[0]) | set(settings[1]) if key != "checkpoint_sha256"):
        raise ValueError("Candidate and baseline evaluation settings differ.")
    measured = resolution_deltas(report)
    checks = [
        {"metric": key, "value_pp": measured[key], "minimum_pp": minimum,
         "comparison": ">" if key.endswith("ci_low_pp") else ">=",
         "passed": measured[key] > minimum if key.endswith("ci_low_pp")
                   else measured[key] >= minimum}
        for key, minimum in GATES.items()
    ]
    return {
        "decision": "CONTINUE" if all(check["passed"] for check in checks) else "STOP",
        "checks": checks,
        "rule": "All five gates must pass; failed gates stop investment in this "
                "whole-image high-resolution training route.",
        "scope": "Single-seed investment screen, not proof of general efficacy or novelty. "
                 "STOP does not rule out every small-object or detail-preservation method.",
        "metric_protocol": report["protocol"],
        "bootstrap_scope": "Validation-image recall resampling with fixed selected thresholds; "
                           "not training-seed uncertainty or AP significance.",
    }


def check_training_pair(baseline, candidate):
    manifests = []
    for checkpoint in (baseline, candidate):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        run = checkpoint.parent.parent
        manifest = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
        if SDCLConfig.from_dict(manifest["sdcl"]).enabled or not manifest["ignore_regions"]:
            raise ValueError("Both runs must disable SDCL and enable ignore regions.")
        train = manifest["train"]
        if (train["epochs"] != 100 or train["batch"] != 8 or train["seed"] != 0
                or train["model"] != "yolo11s.pt" or train["resume"]):
            raise ValueError("Screen requires continuous 100-epoch, batch-8, seed-0 "
                             "training from yolo11s.pt.")
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or int(rows[-1]["epoch"]) != 100:
            raise ValueError(f"Training is not complete at epoch 100: {run}")
        manifests.append(manifest)
    first, second = manifests
    if first["train"]["imgsz"] != 640 or second["train"]["imgsz"] != 960:
        raise ValueError("Screen requires the existing 640 baseline and new 960 candidate.")
    allowed = {"imgsz", "name", "save_dir"}
    first_train = {key: value for key, value in first["train"].items() if key not in allowed}
    second_train = {key: value for key, value in second["train"].items() if key not in allowed}
    if first_train != second_train:
        raise ValueError("Training recipes differ beyond resolution and output name.")
    if first["dataset_hashes"] != second["dataset_hashes"]:
        raise ValueError("Training dataset hashes differ.")
    if SDCLConfig.from_dict(first["sdcl"]) != SDCLConfig.from_dict(second["sdcl"]):
        raise ValueError("Saved SDCL configurations differ.")
    return {"baseline": first, "candidate": second}


def run_resolution_screen(
    baseline="output/runs/baseline_yolo11s_100ep_seed0/weights/best.pt",
    candidate="output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt",
    dataset="data/visdrone", output=None, device="0", batch=8,
):
    baseline, candidate = project_path(baseline), project_path(candidate)
    manifests = check_training_pair(baseline, candidate)
    output = project_path(output or f"output/analysis/resolution-{datetime.now():%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite resolution screen: {output}")
    if batch < 1:
        raise ValueError("batch must be positive.")
    output.mkdir(parents=True)
    plan = {
        "fixed_gates_pp": GATES, "gate_version": "2026-10-10",
        "baseline_checkpoint": str(baseline), "candidate_checkpoint": str(candidate),
        "checkpoint_sha256": {"baseline": file_hash(baseline), "candidate": file_hash(candidate)},
        "reference_size": 640, "validation_batch": batch,
        "comparison": "640-trained vs 960-trained, both inferred at 960",
        "training_manifests": manifests,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    inference_dir, training_dir = output / "inference_only", output / "training_increment"
    common = {"dataset": dataset, "device": device, "batch": batch, "reference_size": 640}
    run_size_analysis(
        str(baseline), str(baseline), output=str(inference_dir),
        baseline_imgsz=640, sdcl_imgsz=960, **common,
    )
    run_size_analysis(
        str(baseline), str(candidate), output=str(training_dir),
        baseline_imgsz=960, sdcl_imgsz=960,
        baseline_prediction_cache=str(inference_dir / "predictions/sdcl"), **common,
    )
    inference = json.loads((inference_dir / "report.json").read_text(encoding="utf-8"))
    training = json.loads((training_dir / "report.json").read_text(encoding="utf-8"))
    decision = decide_resolution(training)
    decision["inference_only_deltas_pp"] = resolution_deltas(inference)
    decision["training_increment_deltas_pp"] = resolution_deltas(training)
    decision["total_ap_deltas_pp"] = {
        f"{group}_ap50_95_pp": 100 * (
            training["coco_reference"]["sdcl"][group]["ap50_95"]
            - inference["coco_reference"]["baseline"][group]["ap50_95"]
        )
        for group in ("all", "tiny", "small_all")
    }
    decision["performance"] = {
        "640_trained_at_640": inference["captures"]["baseline"]["performance"],
        "640_trained_at_960": inference["captures"]["sdcl"]["performance"],
        "960_trained_at_960": training["captures"]["sdcl"]["performance"],
    }
    decision["reports"] = {"inference_only": str(inference_dir / "report.json"),
                           "training_increment": str(training_dir / "report.json")}
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    rows = [
        "# Resolution investment screen", "", f"Decision: **{decision['decision']}**", "",
        "Decision compares both checkpoints at 960 with fixed 640 size groups.", "",
        "| Metric | Delta (pp) | Required | Pass |", "|---|---:|---:|---|",
    ]
    for check in decision["checks"]:
        rows.append(f"| {check['metric']} | {check['value_pp']:.4f} | "
                    f"{check['comparison']} {check['minimum_pp']:.4f} | {check['passed']} |")
    rows.extend(["", decision["scope"], "", "These are project investment gates, "
                 "not universal significance thresholds. Diagnostic metrics, not official VisDrone AP."])
    (output / "decision.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"output": str(output), **decision}
