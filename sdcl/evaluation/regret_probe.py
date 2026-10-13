"""Fixed investment decision for a matched ranking-objective probe, not a novelty claim."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from time import perf_counter

import numpy as np
import torch
import yaml

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import HISTORY, checked_history
from sdcl.evaluation.localization_screen import owned_diagnostic
from sdcl.evaluation.size_analysis import (
    coco_size_metrics, file_hash, load_records, recall_at_fp_budgets, record_hash, write_csv,
)
from sdcl.postprocess.localization import (
    ActionScorer, action_menu, apply_actions, calibrated_candidates, frozen_control,
    load_calibrator, normalized_actions, select_actions,
)
from sdcl.postprocess.rescue import STRONG, save_json
from sdcl.training.localization import verify_archive
from sdcl.training.regret_probe import (
    ARMS, RUN, check_sources, checked_inputs, load_recipe, source_hashes,
)
from sdcl.training.rescue import archive_source


HISTORY_METRICS_SHA256 = "1ca67cf20d2e1b1a9cde8133bbd521fa4c2578c78d0b60610c0a73cc18902154"
GATES = {
    "tiny_ap_vs_calibration_pp": (">=", .50),
    "small_all_ap_vs_calibration_pp": (">=", .30),
    "all_ap_vs_calibration_pp": (">=", .20),
    "recall_fppi5_vs_calibration_pp": (">=", -.10),
    "recall_fppi10_vs_calibration_pp": (">=", -.10),
    "small_all_ap_vs_old_gain_pp": (">=", .20),
    "small_all_ap_vs_uniform_pp": (">=", .20),
    "owned_small_all_iou_vs_uniform_pp": (">=", .50),
    "owned_small_all_iou_vs_uniform_ci_low_pp": (">", 0.),
    "added_cached_mean_ms": ("<=", 15.),
}


def decide_probe(values, complete):
    if set(values) != set(GATES) or not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Missing/non-finite objective probe metrics.")
    checks = []
    for key, (operator, threshold) in GATES.items():
        value = float(values[key])
        passed = value > threshold if operator == ">" else (value <= threshold if operator == "<=" else value >= threshold)
        checks.append({"metric": key, "value": value, "comparison": operator,
            "threshold": threshold, "unit": "ms" if key.endswith("_ms") else "pp", "passed": bool(passed)})
    return {
        "decision": ("CONTINUE" if all(c["passed"] for c in checks) else "STOP") if complete else "SMOKE_ONLY",
        "complete_formal_screen": bool(complete), "checks": checks,
        "rule": "All ten objective-probe gates must pass, full train cache and 548val, fixed epoch20.",
        "scope": "One seed learnability screen. Regret weighting/ranking is not asserted novel. "
                 "Passing permits independent verification, not more epochs or a publication claim.",
        "stop_rule": "If STOP, finish this cached-feature/action-selection objective line. "
                     "Do not retune temperature, margin, epochs, seeds or select a different validation checkpoint.",
        "previous_localization_STOP": "Unchanged 7/11; this is a separately registered matched-objective probe.",
        "protocol": "COCO-style / IoF-ignore diagnostics, reference640; not official VisDrone AP.",
        "bootstrap": "2000 paired image samples seed0; FP thresholds fixed; no scene, seed or threshold-selection uncertainty.",
    }


def summarize_mechanism(rows):
    counts = np.asarray([r["count"] for r in rows])
    difference = np.asarray([r["regret_iou_sum"] - r["uniform_iou_sum"] for r in rows])
    if not counts.sum():
        raise ValueError("No owned small GT; enlarge smoke subset.")
    n = len(rows)
    samples = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=2000)
    denominator = samples @ counts
    eligible = denominator > 0
    interval = np.quantile(100 * (samples[eligible] @ difference) / denominator[eligible], [.025, .975])
    return {"owned_small_all": int(counts.sum()),
        "regret_vs_uniform_pp": float(100 * difference.sum() / counts.sum()),
        "paired_image_95ci_pp": interval.tolist(),
        "mean_iou": {name: sum(r[f"{name}_iou_sum"] for r in rows) / counts.sum()
                     for name in ("calibration", "old_gain", *ARMS)},
        "scope": "Same calibration-owned GT, score>=.001 and IoU>=.5; excludes unmatched GT."}


def diagnose_probe(checkpoint=f"{RUN}/last.pt", output=None, limit=None):
    recipe = load_recipe()
    if limit is not None and not 1 <= limit <= 548:
        raise ValueError("Smoke limit must be in [1,548].")
    control = frozen_control()
    checkpoint = project_path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != recipe.epochs or saved["recipe"] != asdict(recipe):
        raise ValueError("Only frozen epoch20 may be evaluated.")
    check_sources(saved["source_hashes"])
    verify_archive(checkpoint.parent, saved["source_hashes"], saved["source_archive_sha256"])
    data = project_path(saved["cache_path"])
    checked_inputs(data, saved)
    _, _, historical_predictions = checked_history()
    dataset = project_path("data/visdrone")
    all_records = load_records(dataset, "val")
    raw_path = project_path(STRONG) / "raw_views.jsonl"
    validation = control["frozen_validation"]
    if (len(all_records) != 548 or record_hash(all_records) != validation["records_sha256"]
            or file_hash(raw_path) != validation["raw_views_sha256"]):
        raise ValueError("Frozen validation/raw views changed.")
    records = all_records[:limit] if limit is not None else all_records
    expected = {r["stem"]: r for r in records}
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(output or f"output/analysis/regret-probe-{now:%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.mkdir(parents=True)
    plan = {"recipe": asdict(recipe), "fixed_gates": GATES, "source_hashes": source_hashes(),
        "checkpoint_sha256": file_hash(checkpoint), "cache_sha256": saved["cache_sha256"],
        "full_training_data": saved["full_training_data"], "validation_images": len(records),
        "raw_views_sha256": file_hash(raw_path), "records_sha256": record_hash(records),
        "timing": "Warmed CPU1 cached action-menu + normalization + regret-head + selection; "
                  "exclude calibration, other experimental heads, detector inference and decoding. Not deployment FPS.",
        "selection": "No GT in candidate selection; scores/classes/order/count fixed. No NMS or new views.",
        "created_at": now.isoformat()}
    save_json(output / "plan.json", plan)
    archive_source(output, plan["source_hashes"])
    calibrator = load_calibrator(control)
    models = {}
    for arm in ARMS:
        models[arm] = ActionScorer().eval()
        models[arm].load_state_dict(saved["models"][arm])
    mean, std = saved["mean"].numpy(), saved["std"].numpy()
    predictions = {name: {} for name in ("calibration", "old_gain", *ARMS)}
    timings, rows, selections = [], [], []
    warmed = False
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if raw["image"] not in expected:
                continue
            r = expected[raw["image"]]
            stem = r["stem"]
            if stem in predictions["regret"]:
                raise ValueError("Duplicate validation image.")
            ignores = json.loads((dataset / "metadata/val" / f"{stem}.json").read_text())["ignore_boxes_xyxy"]
            c, o, s, f = calibrated_candidates(raw["views"], raw["windows"], r["size"], ignores, calibrator)
            if not np.array_equal(c, historical_predictions["calibration"][stem]):
                raise AssertionError("Calibration replay differs.")
            if not warmed:
                for _ in range(2):
                    _, _, feature = action_menu(c, o, s, f, raw["views"], r["size"], ignores)
                    with torch.inference_mode():
                        for model in models.values():
                            model(torch.from_numpy(normalized_actions(feature, mean, std)))
                warmed = True
            tick = perf_counter()
            boxes, valid, feature = action_menu(c, o, s, f, raw["views"], r["size"], ignores)
            x = torch.from_numpy(normalized_actions(feature, mean, std))
            common_ms = (perf_counter() - tick) * 1000
            image = {"calibration": c, "old_gain": historical_predictions["gain"][stem]}
            timing = {"image": stem, "menu_normalization_ms": common_ms}
            for arm in ARMS:
                tick = perf_counter()
                with torch.inference_mode():
                    utility = models[arm](x).numpy()
                selected = select_actions(utility, valid, "absolute")
                image[arm] = apply_actions(c, boxes, selected)
                timing[f"{arm}_head_selection_ms"] = (perf_counter() - tick) * 1000
                selections.append({"image": stem, "arm": arm,
                    **{f"action_{i}": int((selected == i).sum()) for i in range(6)}})
            timing["regret_added_ms"] = common_ms + timing["regret_head_selection_ms"]
            timings.append(timing)
            for name, values in image.items():
                predictions[name][stem] = values
            rows.append(owned_diagnostic(r, image))
            if len(rows) == 1 or len(rows) % 50 == 0 or len(rows) == len(records):
                print(f"Replayed objective probe: {len(rows)}/{len(records)}.", flush=True)
    if any(set(mapping) != set(expected) for mapping in predictions.values()):
        raise ValueError("Missing validation images.")
    captures = {}
    for name, mapping in predictions.items():
        folder = output / "predictions" / name
        folder.mkdir(parents=True)
        with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for r in records:
                stream.write(json.dumps({"image": r["stem"], "boxes": mapping[r["stem"]].tolist()}) + "\n")
        captures[name] = {"predictions_sha256": file_hash(folder / "predictions.jsonl"),
                         "records_sha256": record_hash(records), "source_hashes": plan["source_hashes"]}
        save_json(folder / "manifest.json", captures[name])
    names = yaml.safe_load((dataset / "dataset.yaml").read_text())["names"]
    names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    metrics = {}
    if limit is None:
        history_path = project_path(HISTORY) / "coco_reference.json"
        if file_hash(history_path) != HISTORY_METRICS_SHA256:
            raise ValueError("Frozen historical AP report changed.")
        historical = json.loads(history_path.read_text())
        metrics.update(calibration=historical["calibration"], old_gain=historical["gain"])
    for name in predictions:
        if name not in metrics:
            print(f"Reference640 AP: {name}.", flush=True)
            metrics[name] = coco_size_metrics(records, predictions[name], names, reference_size=640)
    save_json(output / "coco_reference.json", metrics)
    budgets, _ = recall_at_fp_budgets(records,
        {"baseline": predictions["calibration"], "sdcl": predictions["regret"]},
        reference_size=640, budgets=(5, 10), bootstrap_samples=2000)
    save_json(output / "matched_fp_budgets.json", budgets)
    mechanism = summarize_mechanism(rows)
    save_json(output / "owned_gt_mechanism.json", mechanism)
    write_csv(output / "timings.csv", timings)
    write_csv(output / "action_counts.csv", selections)
    write_csv(output / "owned_gt_per_image.csv", rows)
    timing = {"mean_added_ms": float(np.mean([r["regret_added_ms"] for r in timings])),
              "p95_added_ms": float(np.quantile([r["regret_added_ms"] for r in timings], .95)),
              "scope": plan["timing"]}
    save_json(output / "processing_times.json", timing)
    values = {f"{group}_ap_vs_calibration_pp": 100 * (
        metrics["regret"][group]["ap50_95"] - metrics["calibration"][group]["ap50_95"])
        for group in ("tiny", "small_all", "all")}
    for budget in (5, 10):
        row = next(row for row in budgets if row["label"] == f"fppi_{budget}")
        values[f"recall_fppi{budget}_vs_calibration_pp"] = row["delta_recall_pp"]["small_all"]
    for name in ("old_gain", "uniform"):
        values[f"small_all_ap_vs_{name}_pp"] = 100 * (
            metrics["regret"]["small_all"]["ap50_95"] - metrics[name]["small_all"]["ap50_95"])
    values["owned_small_all_iou_vs_uniform_pp"] = mechanism["regret_vs_uniform_pp"]
    values["owned_small_all_iou_vs_uniform_ci_low_pp"] = mechanism["paired_image_95ci_pp"][0]
    values["added_cached_mean_ms"] = timing["mean_added_ms"]
    decision = decide_probe(values, limit is None and saved["full_training_data"])
    check_sources(plan["source_hashes"])
    checked_inputs(data, saved)
    verification = {"calibration_replay_exact": True, "source_and_train_cache_match": True,
        "same_scores_classes_order_count": all(np.array_equal(v[:, 4:], predictions["calibration"][stem][:, 4:])
            for mapping in predictions.values() for stem, v in mapping.items()),
        "checkpoint_hash_match": file_hash(checkpoint) == plan["checkpoint_sha256"],
        "validation_records_unchanged": record_hash(load_records(dataset, "val")) == record_hash(all_records),
        "raw_views_unchanged": file_hash(raw_path) == plan["raw_views_sha256"],
        "predictions_hashes_match": all(file_hash(output / "predictions" / name / "predictions.jsonl")
            == manifest["predictions_sha256"] for name, manifest in captures.items()),
        "calibration_capture_bytes_match": limit is not None or captures["calibration"]["predictions_sha256"]
            == control["prediction_capture"]["sha256"]}
    if not all(verification.values()):
        raise RuntimeError("Objective probe integrity failed.")
    save_json(output / "verification.json", verification)
    save_json(output / "decision.json", decision)
    lines = ["# Matched ranking-objective probe", "", f"Decision: **{decision['decision']}**", "",
             "| Check | Value | Requirement | Pass |", "|---|---:|---:|---|"]
    for row in decision["checks"]:
        lines.append(f"| {row['metric']} | {row['value']:.4f} {row['unit']} | "
                     f"{row['comparison']} {row['threshold']:.4f} | {row['passed']} |")
    lines.extend(["", decision["scope"], "", decision["stop_rule"], "", decision["protocol"],
                  "", decision["previous_localization_STOP"], "", plan["timing"]])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "decision": decision["decision"], "checks": decision["checks"],
            "complete_formal_screen": decision["complete_formal_screen"]}
