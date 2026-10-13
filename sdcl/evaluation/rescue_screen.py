"""Frozen investment decision against Soft-NMS and two matched learned controls."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from time import perf_counter

import numpy as np
import torch
import yaml

from sdcl.config import project_path
from sdcl.postprocess.rescue import (
    RAW_VAL_SHA256, STRONG, VARIANTS, RescueScorer, candidate_features, check_sources,
    load_recipe, normalized_features, save_json, score_correction, soft_nms_with_origins,
    source_hashes,
)
from sdcl.training.rescue import DATA, RUN, archive_source
from sdcl.evaluation.size_analysis import (
    analyze_predictions, file_hash, load_records, recall_at_fp_budgets, record_hash, write_csv,
)
from sdcl.evaluation.slicing_screen import (
    BASELINE, BASELINE_SHA256, DATA_YAML_SHA256, EXPECTED_IMAGES, RECORDS_SHA256,
)


GATES = {
    "recall_fppi10_vs_soft_pp": 0.50,
    "recall_fppi10_vs_soft_ci_low_pp": 0.0,
    "tiny_ap_vs_soft_pp": 0.30,
    "all_ap_vs_soft_pp": -0.10,
    "recall_fppi5_vs_soft_pp": -0.10,
    "recall_fppi10_vs_calibration_pp": 0.30,
    "recall_fppi10_vs_calibration_ci_low_pp": 0.0,
    "recall_fppi10_vs_no_observability_pp": 0.20,
    "recall_fppi10_vs_no_observability_ci_low_pp": 0.0,
}


def decide_rescue(report, controls, complete):
    def budget(items, label):
        return next(item for item in items if item["label"] == label)

    against_soft = budget(report["matched_fp_budgets"], "fppi_10")
    values = {
        "recall_fppi10_vs_soft_pp": against_soft["delta_recall_pp"]["small_all"],
        "recall_fppi10_vs_soft_ci_low_pp":
            against_soft["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0],
        "tiny_ap_vs_soft_pp": 100 * (
            report["coco_reference"]["sdcl"]["tiny"]["ap50_95"]
            - report["coco_reference"]["baseline"]["tiny"]["ap50_95"]),
        "all_ap_vs_soft_pp": 100 * (
            report["coco_reference"]["sdcl"]["all"]["ap50_95"]
            - report["coco_reference"]["baseline"]["all"]["ap50_95"]),
        "recall_fppi5_vs_soft_pp": budget(report["matched_fp_budgets"], "fppi_5")[
            "delta_recall_pp"]["small_all"],
    }
    for variant in ("calibration", "no_observability"):
        comparison = budget(controls[variant], "fppi_10")
        values[f"recall_fppi10_vs_{variant}_pp"] = comparison["delta_recall_pp"]["small_all"]
        values[f"recall_fppi10_vs_{variant}_ci_low_pp"] = comparison[
            "paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0]
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite decision metrics.")
    checks = [
        {"metric": key, "value_pp": values[key], "minimum_pp": minimum,
         "comparison": ">" if "ci_low" in key else ">=",
         "passed": values[key] > minimum if "ci_low" in key else values[key] >= minimum}
        for key, minimum in GATES.items()
    ]
    return {
        "decision": ("CONTINUE" if all(item["passed"] for item in checks) else "STOP")
                    if complete else "SMOKE_ONLY",
        "complete_formal_screen": bool(complete), "checks": checks,
        "rule": "All nine predeclared checks must pass, including learned-control and mechanism checks.",
        "scope": "One-seed investment screen of view-conditioned rescoring. "
                 "Passing is not proof of novelty or cross-seed/independent-data efficacy.",
        "stop_rule": "If STOP, stop this scorer recipe; keep the fixed Soft-NMS control. "
                     "Do not change opportunity thresholds, gain, epochs or add a rescue module.",
        "bootstrap_scope": "2000 paired image resamples; selected thresholds fixed. "
                           "No seed or threshold-selection uncertainty.",
    }


def diagnose_rescue(checkpoint=f"{RUN}/last.pt", output=None, limit=None):
    recipe = load_recipe()
    if limit is not None and not 1 <= limit <= EXPECTED_IMAGES:
        raise ValueError("Smoke limit must be in [1, 548].")
    checkpoint = project_path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != recipe.epochs or saved["recipe"] != asdict(recipe):
        raise ValueError("Evaluate fixed epoch 20 with the exact frozen recipe.")
    check_sources(saved["source_hashes"])
    cache = project_path(saved["cache_path"])
    if file_hash(cache / "manifest.json") != saved["cache_sha256"]:
        raise ValueError("Training cache manifest changed.")
    cache_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if (not cache_manifest["complete"] or cache_manifest["settings"]["split"] != "train"
            or cache_manifest["settings"]["source_hashes"] != saved["source_hashes"]
            or cache_manifest["settings"]["full_training_data"] != saved["full_training_data"]):
        raise ValueError("Invalid training provenance.")
    for chunk in cache_manifest["chunks"].values():
        if file_hash(cache / chunk["file"]) != chunk["sha256"]:
            raise ValueError("Training chunk changed.")
    dataset, strong = project_path("data/visdrone"), project_path(STRONG)
    if (file_hash(project_path(BASELINE)) != BASELINE_SHA256
            or file_hash(dataset / "dataset.yaml") != DATA_YAML_SHA256
            or file_hash(strong / "raw_views.jsonl") != RAW_VAL_SHA256):
        raise ValueError("Frozen detector/data/validation raw predictions changed.")
    all_records = load_records(dataset, "val")
    if len(all_records) != EXPECTED_IMAGES or record_hash(all_records) != RECORDS_SHA256:
        raise ValueError("Frozen validation data changed.")
    records = all_records[:limit] if limit is not None else all_records
    baseline_manifest = json.loads(
        (strong / "predictions/soft_nms/manifest.json").read_text(encoding="utf-8"))
    baseline_path = strong / "predictions/soft_nms/predictions.jsonl"
    if file_hash(baseline_path) != baseline_manifest["predictions_sha256"]:
        raise ValueError("Strong-control prediction hash mismatch.")
    baseline = {}
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        baseline[item["image"]] = np.asarray(item["boxes"], dtype=float).reshape(-1, 6)
    torch.set_num_threads(1)
    models = {}
    for variant in VARIANTS:
        model = RescueScorer()
        model.load_state_dict(saved["models"][variant])
        models[variant] = model.eval()
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(output or f"output/analysis/rescue-{now:%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.mkdir(parents=True)
    hashes = source_hashes()
    plan = {
        "recipe": asdict(recipe), "fixed_gates_pp": GATES, "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint), "training_cache_sha256": saved["cache_sha256"],
        "full_training_data": saved["full_training_data"], "validation_images": len(records),
        "raw_views_sha256": RAW_VAL_SHA256, "baseline_predictions_sha256": file_hash(baseline_path),
        "source_hashes": hashes, "created_at": now.isoformat(),
        "comparison": "Frozen Soft-NMS candidate set, identical boxes/views; only scores change.",
        "controls": list(VARIANTS), "selection": "One fixed epoch-20 run; no validation tuning.",
        "timing": "CPU one thread. Measures cached origin reconstruction, features and model "
                  "score correction separately; not an end-to-end deployment benchmark.",
    }
    save_json(output / "plan.json", plan)
    archive_source(output, hashes)
    expected = {record["stem"]: record for record in records}
    predictions = {name: {} for name in ("soft_nms", *VARIANTS)}
    timings = []
    mean, std = saved["mean"].numpy(), saved["std"].numpy()
    for model in models.values():
        with torch.inference_mode():
            model(torch.zeros(16, len(mean)))
    with (strong / "raw_views.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if raw["image"] not in expected:
                continue
            record = expected[raw["image"]]
            ignores = json.loads(
                (dataset / "metadata/val" / f"{record['stem']}.json").read_text(encoding="utf-8")
            )["ignore_boxes_xyxy"]
            started = perf_counter()
            candidates, origins, raw_scores = soft_nms_with_origins(
                raw["views"], record["size"], ignores)
            reconstruct_ms = (perf_counter() - started) * 1000
            if not np.array_equal(candidates, baseline[record["stem"]]):
                raise AssertionError("Reconstructed candidates differ from frozen Soft-NMS.")
            started = perf_counter()
            features = candidate_features(candidates, origins, raw_scores, raw["views"],
                                          raw["windows"], record["size"], ignores)
            feature_ms = (perf_counter() - started) * 1000
            predictions["soft_nms"][record["stem"]] = baseline[record["stem"]]
            row = {"image": record["stem"], "reconstruct_soft_nms_ms": reconstruct_ms,
                   "feature_ms": feature_ms}
            for variant in VARIANTS:
                started = perf_counter()
                with torch.inference_mode():
                    x = torch.from_numpy(normalized_features(features, mean, std, variant))
                    probabilities = models[variant](x).softmax(dim=1).numpy()
                predictions[variant][record["stem"]] = score_correction(candidates, probabilities, variant)
                row[f"{variant}_score_ms"] = (perf_counter() - started) * 1000
            timings.append(row)
            if len(timings) == 1 or len(timings) % 50 == 0 or len(timings) == len(records):
                print(f"Rescored cached validation images: {len(timings)}/{len(records)}.", flush=True)
    if any(set(values) != set(expected) for values in predictions.values()):
        raise ValueError("Validation capture missing or extra images.")
    write_csv(output / "timings.csv", timings)
    captures = {}
    common_settings = {
        "detector_checkpoint_sha256": BASELINE_SHA256, "records_sha256": record_hash(records),
        "raw_views_sha256": RAW_VAL_SHA256, "reference_size": 640, "final_max_det": 500,
    }
    for name, values in predictions.items():
        directory = output / "predictions" / name
        directory.mkdir(parents=True)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"image": record["stem"],
                                         "boxes": values[record["stem"]].tolist()}) + "\n")
        captures[name] = {
            "settings": common_settings, "rescorer": name,
            "checkpoint_sha256": None if name == "soft_nms" else plan["checkpoint_sha256"],
            "predictions_sha256": file_hash(directory / "predictions.jsonl"),
        }
        save_json(directory / "manifest.json", captures[name])
    if limit is None and captures["soft_nms"]["predictions_sha256"] != plan["baseline_predictions_sha256"]:
        raise AssertionError("Saved strong-control prediction file differs.")
    names = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(key): value for key, value in
             (names.items() if isinstance(names, dict) else enumerate(names))}
    report_dir = output / "rescue_vs_soft_nms"
    report_dir.mkdir()
    analyze_predictions(
        records, {"baseline": predictions["soft_nms"], "sdcl": predictions["rescue"]},
        {"baseline": captures["soft_nms"], "sdcl": captures["rescue"]}, names, report_dir,
        {"baseline": 960, "sdcl": 960}, reference_size=640,
        model_labels={"baseline": "Frozen Soft-NMS", "sdcl": "View-conditioned rescue scorer"},
        inference_description=plan["comparison"],
    )
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    controls = {}
    for variant in ("calibration", "no_observability"):
        print(f"Paired FP-budget comparison against {variant}.", flush=True)
        controls[variant], _ = recall_at_fp_budgets(
            records, {"baseline": predictions[variant], "sdcl": predictions["rescue"]}, 640)
        save_json(output / f"rescue_vs_{variant}.json", controls[variant])
    complete = limit is None and saved["full_training_data"] and len(records) == EXPECTED_IMAGES
    decision = decide_rescue(report, controls, complete)
    decision["reports"] = {"quality": str(report_dir / "report.json"),
                           **{name: str(output / f"rescue_vs_{name}.json") for name in controls}}
    decision["mean_cached_processing_ms"] = {
        key: float(np.mean([row[key] for row in timings]))
        for key in ("reconstruct_soft_nms_ms", "feature_ms", "rescue_score_ms")}
    decision["timing_note"] = plan["timing"]
    check_sources(hashes)
    verification = {
        "source_hashes_match": True,
        "model_checkpoint_hash_matches": file_hash(checkpoint) == plan["checkpoint_sha256"],
        "training_cache_manifest_matches": file_hash(cache / "manifest.json") == saved["cache_sha256"],
        "raw_predictions_match": file_hash(strong / "raw_views.jsonl") == RAW_VAL_SHA256,
        "baseline_predictions_match": limit is not None or captures["soft_nms"]["predictions_sha256"]
                                      == plan["baseline_predictions_sha256"],
        "all_final_caps_valid": all(len(values) <= 500 for mapping in predictions.values()
                                   for values in mapping.values()),
        "prediction_hashes_match": {
            name: file_hash(output / "predictions" / name / "predictions.jsonl")
                  == capture["predictions_sha256"] for name, capture in captures.items()},
    }
    if (not all(value for key, value in verification.items() if key != "prediction_hashes_match")
            or not all(verification["prediction_hashes_match"].values())):
        raise RuntimeError("Post-evaluation integrity failed.")
    save_json(output / "verification.json", verification)
    save_json(output / "decision.json", decision)
    lines = ["# View-conditioned rescue rescoring", "", f"Decision: **{decision['decision']}**", "",
             "| Check | Delta (pp) | Required | Pass |", "|---|---:|---:|---|"]
    for item in decision["checks"]:
        lines.append(f"| {item['metric']} | {item['value_pp']:.4f} | "
                     f"{item['comparison']} {item['minimum_pp']:.4f} | {item['passed']} |")
    lines.extend(["", decision["scope"], "", decision["stop_rule"], "",
                  decision["bootstrap_scope"], "", plan["timing"], "",
                  "COCO-style / IoF-ignore diagnostics, not official VisDrone AP."])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "decision": decision["decision"], "checks": decision["checks"],
            "complete_formal_screen": decision["complete_formal_screen"]}
