"""Predeclared localization screen against the frozen ordinary calibration control."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from time import perf_counter

import numpy as np
import torch
import yaml

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, coco_size_metrics, file_hash, match_at_threshold,
    load_records, recall_at_fp_budgets, record_hash, write_csv,
)
from sdcl.postprocess.localization import (
    CONTROL_SHA256, ActionScorer, action_menu, apply_actions, calibrated_candidates,
    check_sources, frozen_control, load_calibrator, load_recipe,
    normalized_actions, select_actions, source_hashes,
)
from sdcl.postprocess.rescue import STRONG, save_json
from sdcl.training.localization import RUN, verify_archive, verify_cache
from sdcl.training.rescue import archive_source


# Freeze these before fitting, including the computational-cost ceiling.
GATES = {
    "tiny_ap_vs_calibration_pp": (">=", 0.50),
    "small_all_ap_vs_calibration_pp": (">=", 0.30),
    "all_ap_vs_calibration_pp": (">=", 0.20),
    "recall_fppi5_vs_calibration_pp": (">=", -0.10),
    "recall_fppi10_vs_calibration_pp": (">=", -0.10),
    "small_all_ap_vs_vote_pp": (">=", 0.20),
    "small_all_ap_vs_absolute_pp": (">=", 0.20),
    "small_all_ap_vs_gain_always_pp": (">=", 0.20),
    "owned_small_all_iou_vs_absolute_pp": (">=", 0.50),
    "owned_small_all_iou_vs_absolute_ci_low_pp": (">", 0.0),
    "added_cached_mean_ms": ("<=", 15.0),
}
ARMS = ("calibration", "vote", "absolute", "gain_always", "gain")


def decide_localization(values, complete):
    if set(values) != set(GATES) or not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Missing/non-finite localization decision metrics.")
    checks = []
    for name, (comparison, threshold) in GATES.items():
        value = float(values[name])
        passed = value > threshold if comparison == ">" else (
            value <= threshold if comparison == "<=" else value >= threshold)
        checks.append({"metric": name, "value": value, "comparison": comparison,
                       "threshold": threshold, "unit": "ms" if name.endswith("_ms") else "pp",
                       "passed": bool(passed)})
    return {
        "decision": ("CONTINUE" if all(c["passed"] for c in checks) else "STOP")
                    if complete else "SMOKE_ONLY",
        "complete_formal_screen": bool(complete), "checks": checks,
        "rule": "All eleven checks must pass on full train and all 548 frozen val images, epoch20.",
        "scope": "One seed investment screen; passing does not prove novelty or generalization.",
        "stop_rule": "STOP this recipe; keep ordinary calibration. Do not tune peers, gain margin, "
                     "epochs, seeds or add modules after seeing results.",
        "bootstrap_scope": "2000 paired image resamples, seed0; FP thresholds held fixed. "
                           "No seed/threshold-selection uncertainty.",
        "protocol": "Project COCO-style / IoF-ignore diagnostics; not official VisDrone AP.",
    }


def owned_diagnostic(record, predictions):
    baseline = predictions["calibration"]
    gt = np.asarray([obj["box_xyxy"] for obj in record["objects"]]).reshape(-1, 4)
    classes = [obj["class_id"] for obj in record["objects"]]
    matched, _ = match_at_threshold(box_iou_matrix(gt, baseline[:, :4]), classes, baseline, 0.001)
    small = box_areas(gt) * (640 / max(record["size"]))**2 < 32**2
    indices = np.flatnonzero(small & (matched >= 0))
    row = {"image": record["stem"], "count": len(indices)}
    for name, values in predictions.items():
        # Preserve baseline identity even if another nearby object fits the chosen box better.
        overlap = box_iou_matrix(gt[indices], values[matched[indices], :4])
        row[f"{name}_iou_sum"] = float(np.diag(overlap).sum())
        old = np.diag(box_iou_matrix(gt[indices], baseline[matched[indices], :4]))
        current = np.diag(overlap)
        row[f"{name}_improved"] = int((current > old + 1e-9).sum())
        row[f"{name}_worsened"] = int((current < old - 1e-9).sum())
    return row


def summarize_owned(rows):
    counts = np.asarray([row["count"] for row in rows])
    difference = np.asarray([row["gain_iou_sum"] - row["absolute_iou_sum"] for row in rows])
    if not counts.sum():
        raise ValueError("No baseline-owned small GT; choose a larger smoke subset.")
    n = len(rows)
    samples = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=2000)
    denominator = samples @ counts
    valid = denominator > 0
    interval = np.quantile(100 * (samples[valid] @ difference) / denominator[valid], [.025, .975])
    return {
        "owned_small_all_gt": int(counts.sum()),
        "mean_iou": {name: float(sum(row[f"{name}_iou_sum"] for row in rows) / counts.sum())
                     for name in ARMS},
        "gain_vs_absolute_pp": float(100 * difference.sum() / counts.sum()),
        "gain_vs_absolute_paired_image_95ci_pp": interval.tolist(),
        "identity": "GT ownership frozen by calibration greedy IoU>=.5 matching at score>=.001.",
        "limitations": "Conditional on baseline-owned GT; excludes baseline IoU<.5 misses. "
                       "Does not establish efficacy on all GT or an independent test split.",
    }


def read_predictions(path):
    values = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["image"] in values:
                raise ValueError("Duplicate prediction image.")
            values[row["image"]] = np.asarray(row["boxes"], dtype=float).reshape(-1, 6)
    return values


def diagnose_localization(checkpoint=f"{RUN}/last.pt", output=None, limit=None):
    recipe = load_recipe()
    if limit is not None and not 1 <= limit <= 548:
        raise ValueError("Smoke validation limit must be in [1, 548].")
    control = frozen_control()
    checkpoint = project_path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != recipe.epochs or saved["recipe"] != asdict(recipe):
        raise ValueError("Evaluate fixed epoch20 with the exact recipe.")
    check_sources(saved["source_hashes"])
    verify_archive(checkpoint.parent, saved["source_hashes"], saved["source_archive_sha256"])
    cache = project_path(saved["cache_path"])
    verify_cache(cache, saved)
    dataset = project_path("data/visdrone")
    all_records = load_records(dataset, "val")
    if (len(all_records) != 548
            or record_hash(all_records) != control["frozen_validation"]["records_sha256"]
            or file_hash(dataset / "dataset.yaml") != control["frozen_validation"]["data_yaml_sha256"]):
        raise ValueError("Frozen validation data changed.")
    raw_path = project_path(STRONG) / "raw_views.jsonl"
    if file_hash(raw_path) != control["frozen_validation"]["raw_views_sha256"]:
        raise ValueError("Frozen validation views changed.")
    records = all_records[:limit] if limit is not None else all_records
    expected = {r["stem"]: r for r in records}
    baseline = read_predictions(project_path(control["prediction_capture"]["path"]))
    if set(baseline) != {r["stem"] for r in all_records}:
        raise ValueError("Frozen calibration image set differs.")
    calibrator = load_calibrator(control)
    models = {}
    for name in ("absolute", "gain"):
        models[name] = ActionScorer().eval()
        models[name].load_state_dict(saved["models"][name])
    mean, std = saved["mean"].numpy(), saved["std"].numpy()
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(output or f"output/analysis/localization-{now:%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.mkdir(parents=True)
    plan = {
        "recipe": asdict(recipe), "fixed_gates": GATES, "source_hashes": source_hashes(),
        "control_sha256": CONTROL_SHA256, "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint), "training_cache_sha256": saved["cache_sha256"],
        "full_training_data": saved["full_training_data"], "validation_images": len(records),
        "raw_views_sha256": file_hash(raw_path),
        "baseline_predictions_sha256": control["prediction_capture"]["sha256"],
        "created_at": now.isoformat(), "arms": list(ARMS),
        "timing": "CPU1 cached postprocessing, warmed; calibration reconstruction and new menu/"
                  "normalization/head/selection measured separately. Excludes image decode/"
                  "detector inference and excludes other experimental heads from candidate cost.",
        "comparison": "Same calibrated scores, classes, row order and count; only coordinates change. "
                      "No second NMS, validation fitting or annotation use in action selection.",
    }
    save_json(output / "plan.json", plan)
    archive_source(output, plan["source_hashes"])
    predictions = {name: {} for name in ARMS}
    timings, diagnostics, selections = [], [], []
    warmed = False
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if raw["image"] not in expected:
                continue
            record = expected[raw["image"]]
            if record["stem"] in predictions["gain"]:
                raise ValueError("Duplicate raw image.")
            ignores = json.loads(
                (dataset / "metadata/val" / f"{record['stem']}.json").read_text(encoding="utf-8")
            )["ignore_boxes_xyxy"]
            if not warmed:
                for _ in range(2):
                    c, o, s, f = calibrated_candidates(
                        raw["views"], raw["windows"], record["size"], ignores, calibrator)
                    _, _, feature = action_menu(c, o, s, f, raw["views"], record["size"], ignores)
                    with torch.inference_mode():
                        for model in models.values():
                            model(torch.from_numpy(normalized_actions(feature, mean, std)))
                warmed = True
            started = perf_counter()
            c, origins, scores, base = calibrated_candidates(
                raw["views"], raw["windows"], record["size"], ignores, calibrator)
            row = {"image": record["stem"], "calibration_cached_ms": (perf_counter() - started) * 1000}
            if not np.array_equal(c, baseline[record["stem"]]):
                raise AssertionError(f"Frozen calibration replay differs: {record['stem']}")
            started = perf_counter()
            boxes, valid, features = action_menu(c, origins, scores, base,
                                                raw["views"], record["size"], ignores)
            row["menu_ms"] = (perf_counter() - started) * 1000
            started = perf_counter()
            x = torch.from_numpy(normalized_actions(features, mean, std))
            row["normalization_ms"] = (perf_counter() - started) * 1000
            outputs = {}
            for name, model in models.items():
                started = perf_counter()
                with torch.inference_mode():
                    outputs[name] = model(x).numpy()
                row[f"{name}_head_ms"] = (perf_counter() - started) * 1000
            image_predictions = {"calibration": c}
            for name in ARMS[1:]:
                started = perf_counter()
                selected = select_actions(outputs["absolute" if name == "absolute" else "gain"], valid, name)
                image_predictions[name] = apply_actions(c, boxes, selected)
                row[f"{name}_selection_ms"] = (perf_counter() - started) * 1000
                selections.append({"image": record["stem"], "variant": name,
                                   **{f"action_{i}": int((selected == i).sum()) for i in range(6)}})
            row["gain_added_ms"] = sum(row[k] for k in (
                "menu_ms", "normalization_ms", "gain_head_ms", "gain_selection_ms"))
            timings.append(row)
            for name, values in image_predictions.items():
                predictions[name][record["stem"]] = values
            diagnostics.append(owned_diagnostic(record, image_predictions))
            if len(timings) == 1 or len(timings) % 50 == 0 or len(timings) == len(records):
                print(f"Replayed localization screen: {len(timings)}/{len(records)}.", flush=True)
    if any(set(mapping) != set(expected) for mapping in predictions.values()):
        raise ValueError("Validation view coverage differs.")
    write_csv(output / "timings.csv", timings)
    write_csv(output / "action_counts.csv", selections)
    write_csv(output / "owned_gt_per_image.csv", diagnostics)
    captures = {}
    for name, mapping in predictions.items():
        directory = output / "predictions" / name
        directory.mkdir(parents=True)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"image": record["stem"], "boxes": mapping[record["stem"]].tolist()}) + "\n")
        captures[name] = {
            "predictions_sha256": file_hash(directory / "predictions.jsonl"),
            "records_sha256": record_hash(records), "control_sha256": CONTROL_SHA256,
            "head_checkpoint_sha256": plan["checkpoint_sha256"] if name in ("absolute", "gain", "gain_always") else None,
            "description": plan["comparison"],
        }
        save_json(directory / "manifest.json", captures[name])
    if limit is None and captures["calibration"]["predictions_sha256"] != plan["baseline_predictions_sha256"]:
        raise AssertionError("Saved frozen calibration bytes differ.")
    names = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    metrics = {}
    for name in ARMS:
        print(f"COCO reference640 AP: {name}.", flush=True)
        metrics[name] = coco_size_metrics(records, predictions[name], names, reference_size=640)
        save_json(output / "coco_reference.json", metrics)
    budgets, curves = recall_at_fp_budgets(
        records, {"baseline": predictions["calibration"], "sdcl": predictions["gain"]},
        reference_size=640, budgets=(5, 10), bootstrap_samples=2000)
    save_json(output / "matched_fp_budgets.json", budgets)
    save_json(output / "fp_curves.json", curves)
    mechanism = summarize_owned(diagnostics)
    save_json(output / "owned_gt_mechanism.json", mechanism)
    timing = {k: {"mean": float(np.mean([r[k] for r in timings])),
                  "p95": float(np.quantile([r[k] for r in timings], .95))}
              for k in timings[0] if k != "image"}
    save_json(output / "processing_times.json", {"milliseconds": timing, "scope": plan["timing"]})
    values = {}
    for group in ("tiny", "small_all", "all"):
        values[f"{group}_ap_vs_calibration_pp"] = 100 * (
            metrics["gain"][group]["ap50_95"] - metrics["calibration"][group]["ap50_95"])
    for budget in (5, 10):
        item = next(r for r in budgets if r["label"] == f"fppi_{budget}")
        values[f"recall_fppi{budget}_vs_calibration_pp"] = item["delta_recall_pp"]["small_all"]
    for name in ("vote", "absolute", "gain_always"):
        values[f"small_all_ap_vs_{name}_pp"] = 100 * (
            metrics["gain"]["small_all"]["ap50_95"] - metrics[name]["small_all"]["ap50_95"])
    values["owned_small_all_iou_vs_absolute_pp"] = mechanism["gain_vs_absolute_pp"]
    values["owned_small_all_iou_vs_absolute_ci_low_pp"] = mechanism["gain_vs_absolute_paired_image_95ci_pp"][0]
    values["added_cached_mean_ms"] = timing["gain_added_ms"]["mean"]
    complete = limit is None and saved["full_training_data"] and len(records) == 548
    decision = decide_localization(values, complete)
    check_sources(plan["source_hashes"])
    frozen_control()
    verify_cache(cache, saved)
    verification = {
        "source_hashes_match": True, "train_cache_and_records_match": True,
        "calibration_replay_exact_all_evaluated_images": True,
        "score_class_count_order_preserved": all(
            np.array_equal(values[:, 4:], predictions["calibration"][stem][:, 4:])
            for mapping in predictions.values() for stem, values in mapping.items()),
        "raw_views_hash_match": file_hash(raw_path) == plan["raw_views_sha256"],
        "checkpoint_hash_match": file_hash(checkpoint) == plan["checkpoint_sha256"],
        "validation_records_hash_match": record_hash(load_records(dataset, "val")) == record_hash(all_records),
        "prediction_files_hash_match": all(
            file_hash(output / "predictions" / name / "predictions.jsonl") == capture["predictions_sha256"]
            for name, capture in captures.items()),
    }
    if not all(verification.values()):
        raise RuntimeError("Localization post-evaluation integrity failed.")
    save_json(output / "verification.json", verification)
    save_json(output / "decision.json", decision)
    lines = ["# Anchor-referenced localization gain screen", "", f"Decision: **{decision['decision']}**", "",
             "| Check | Value | Requirement | Pass |", "|---|---:|---:|---|"]
    for item in decision["checks"]:
        lines.append(f"| {item['metric']} | {item['value']:.4f} {item['unit']} | "
                     f"{item['comparison']} {item['threshold']:.4f} | {item['passed']} |")
    lines.extend(["", decision["scope"], "", decision["stop_rule"], "", decision["protocol"],
                  "", decision["bootstrap_scope"], "", plan["timing"]])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "decision": decision["decision"],
            "complete_formal_screen": complete, "checks": decision["checks"]}
