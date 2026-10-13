"""Label-assisted diagnostic only: fixed action-space potential and cached-pool misses."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from time import perf_counter
import zipfile

import numpy as np
import yaml

from sdcl.config import project_path
from sdcl.evaluation.localization_screen import read_predictions
from sdcl.evaluation.size_analysis import (
    GROUPS, box_areas, box_iou_matrix, coco_size_metrics, file_hash, group_mask,
    load_records, match_at_threshold, recall_at_fp_budgets, record_hash, write_csv,
)
from sdcl.postprocess.localization import (
    action_menu, action_targets, apply_actions, calibrated_candidates, frozen_control,
    load_calibrator, source_hashes,
)
from sdcl.postprocess.rescue import STRONG, clean_views, save_json
from sdcl.training.rescue import archive_source


HISTORY = "output/analysis/localization-recheck-20261013"
HISTORY_PLAN_SHA256 = "6718065ca8bf2b81e658315baf7a8f27212583dce71d46c79a700d21f523792c"
GATES = {
    "tiny_oracle_ap_vs_calibration_pp": (">=", 0.50),
    "small_all_oracle_ap_vs_calibration_pp": (">=", 0.30),
    "all_oracle_ap_vs_calibration_pp": (">=", 0.20),
    "owned_small_all_oracle_iou_vs_calibration_pp": (">=", 0.50),
    "owned_small_all_oracle_iou_ci_low_pp": (">", 0.0),
}
CAUSES = (
    "detected", "final_matching_competition", "final_below_operating_score",
    "fusion_or_cap_removal", "ignore_filter_removal", "raw_class_mismatch",
    "raw_localization_error", "no_sameclass_nearby_in_cached_pool",
)


def verify_dependencies(hashes, archive_path):
    """Allow independent new modules, but reject any changed historical dependency."""
    for name, digest in hashes.items():
        if file_hash(project_path(name)) != digest:
            raise ValueError(f"Historical diagnostic dependency changed: {name}")
    with zipfile.ZipFile(archive_path) as archive:
        if (set(archive.namelist()) != set(hashes)
                or any(hashlib.sha256(archive.read(name)).hexdigest() != digest
                       for name, digest in hashes.items())):
            raise ValueError("Historical source ZIP differs from the pinned plan.")


def checked_history():
    path = project_path(HISTORY)
    if file_hash(path / "plan.json") != HISTORY_PLAN_SHA256:
        raise ValueError("Pinned localization history plan changed.")
    plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
    verify_dependencies(plan["source_hashes"], path / "source.zip")
    if file_hash(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ValueError("Historical localization checkpoint changed.")
    decision = json.loads((path / "decision.json").read_text(encoding="utf-8"))
    if decision["decision"] != "STOP" or not decision["complete_formal_screen"]:
        raise ValueError("This diagnostic requires the completed localization STOP screen.")
    predictions = {}
    for name in ("calibration", "gain"):
        folder = path / "predictions" / name
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (file_hash(folder / "predictions.jsonl") != manifest["predictions_sha256"]
                or manifest["records_sha256"] != frozen_control()["frozen_validation"]["records_sha256"]):
            raise ValueError("Historical prediction capture changed.")
        predictions[name] = read_predictions(folder / "predictions.jsonl")
    return plan, decision, predictions


def select_owned_oracle(targets, owned, valid):
    """Never switch GT identity; invalid actions cannot win, ties prefer keep."""
    if targets.shape != valid.shape or owned.shape != (len(targets),) or not valid[:, 0].all():
        raise ValueError("Invalid oracle target/mask shape.")
    if not np.isfinite(targets).all():
        raise ValueError("Non-finite oracle targets.")
    selected = np.argmax(np.where(valid, targets, -np.inf), axis=1)
    selected[~owned] = 0
    return selected


def uncluttered_raw_pool(views, size):
    # Match clean_views clipping/validity, but deliberately omit only the ignore filter.
    values = np.concatenate([np.asarray(v, np.float32).reshape(-1, 6) for v in views]).copy()
    values = values[np.isfinite(values).all(1)]
    values[:, [0, 2]] = values[:, [0, 2]].clip(0, size[0])
    values[:, [1, 3]] = values[:, [1, 3]].clip(0, size[1])
    valid = ((values[:, 2] > values[:, 0]) & (values[:, 3] > values[:, 1])
             & (values[:, 4] >= .001) & (values[:, 4] <= 1)
             & (values[:, 5] >= 0) & (values[:, 5] == np.floor(values[:, 5])))
    return values[valid]


def best_same_iou(gt, classes, values):
    ious = box_iou_matrix(gt, values[:, :4])
    same = np.asarray(classes)[:, None] == values[None, :, 5]
    return np.where(same, ious, 0).max(1, initial=0), ious.max(1, initial=0), ious, same


def miss_partition(record, final, raw_clean, raw_before_ignore, threshold):
    gt = np.asarray([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
    classes = [o["class_id"] for o in record["objects"]]
    fbest, _, fiou, same = best_same_iou(gt, classes, final)
    rbest, rany, riou, rsame = best_same_iou(gt, classes, raw_clean)
    ibest, _, _, _ = best_same_iou(gt, classes, raw_before_ignore)
    matched, _ = match_at_threshold(fiou, classes, final, confidence=threshold)
    high_adequate = (same & (fiou >= .5) & (final[None, :, 4] >= threshold)).any(1)
    causes = []
    for index in range(len(gt)):
        if matched[index] >= 0:
            cause = "detected"
        elif high_adequate[index]:
            cause = "final_matching_competition"
        elif fbest[index] >= .5:
            cause = "final_below_operating_score"
        elif rbest[index] >= .5:
            cause = "fusion_or_cap_removal"
        elif ibest[index] >= .5:
            cause = "ignore_filter_removal"
        elif rany[index] >= .5:
            cause = "raw_class_mismatch"
        elif rbest[index] >= .1:
            cause = "raw_localization_error"
        else:
            cause = "no_sameclass_nearby_in_cached_pool"
        causes.append(cause)
    return causes, matched, {"raw_same_iou": rbest, "raw_any_iou": rany,
                             "final_same_iou": fbest, "raw_edges": rsame & (riou >= .5)}


def max_pool_matches(edges):
    """Geometric matching reference ignoring scores, duplicates, max500 and other groups."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not edges.shape[0] or not edges.shape[1]:
        return 0
    return int((maximum_bipartite_matching(csr_matrix(edges), perm_type="column") >= 0).sum())


def paired_owned_summary(rows):
    counts = np.asarray([r["owned_small_all"] for r in rows])
    difference = np.asarray([r["oracle_iou_sum"] - r["calibration_iou_sum"] for r in rows])
    if not counts.sum():
        raise ValueError("No baseline-owned small GT; increase the smoke subset.")
    n = len(rows)
    samples = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=2000)
    denominator = samples @ counts
    eligible = denominator > 0
    interval = np.quantile(100 * (samples[eligible] @ difference) / denominator[eligible], [.025, .975])
    sums = {name: sum(r[f"{name}_iou_sum"] for r in rows) for name in ("calibration", "oracle", "gain")}
    room = sums["oracle"] - sums["calibration"]
    return {
        "owned_small_all": int(counts.sum()),
        "mean_iou": {name: float(value / counts.sum()) for name, value in sums.items()},
        "oracle_vs_calibration_pp": float(100 * room / counts.sum()),
        "oracle_vs_calibration_paired_image_95ci_pp": interval.tolist(),
        "learned_gain_fraction_of_owned_iou_reference": float((sums["gain"] - sums["calibration"]) / room)
            if room > 0 else None,
        "scope": "Fixed baseline-owned GT only, IoU>=.5 at score>=.001. Not all GT or an AP upper bound.",
    }


def decide_headroom(values, full):
    if set(values) != set(GATES) or not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Missing/non-finite headroom metrics.")
    checks = [{"metric": name, "value_pp": float(values[name]), "comparison": operator,
               "threshold_pp": threshold,
               "passed": bool(values[name] > threshold if operator == ">" else values[name] >= threshold)}
              for name, (operator, threshold) in GATES.items()]
    passed = all(c["passed"] for c in checks)
    return {
        "decision": ("HEADROOM_ONLY" if passed else "STOP_CURRENT_ACTION_SPACE") if full else "SMOKE_ONLY",
        "full_validation": bool(full), "checks": checks, "training_authorized_by_diagnostic": False,
        "previous_localization_decision": "STOP; unchanged 7/11.",
        "rule": "All five predeclared diagnostic checks must pass; no fitting or threshold tuning.",
        "scope": "GT-assisted optimistic reference for one frozen six-action menu, not deployable efficacy, "
                 "novelty, a global AP optimum or a reason to resume the stopped recipe.",
        "next_action": "If HEADROOM_ONLY, test a distinct observable signal on disjoint training scenes "
                       "before designing a new bounded training run. Otherwise stop this action-space investment.",
    }


def diagnose_headroom(output=None, limit=None):
    if limit is not None and not 1 <= limit <= 548:
        raise ValueError("Smoke limit must be in [1,548].")
    control = frozen_control()
    history, previous, saved_predictions = checked_history()
    dataset = project_path("data/visdrone")
    all_records = load_records(dataset, "val")
    validation = control["frozen_validation"]
    if (len(all_records) != 548 or record_hash(all_records) != validation["records_sha256"]
            or file_hash(dataset / "dataset.yaml") != validation["data_yaml_sha256"]):
        raise ValueError("Frozen validation data changed.")
    raw_path = project_path(STRONG) / "raw_views.jsonl"
    if file_hash(raw_path) != validation["raw_views_sha256"]:
        raise ValueError("Cached raw views changed.")
    expected_stems = {r["stem"] for r in all_records}
    if any(set(mapping) != expected_stems for mapping in saved_predictions.values()):
        raise ValueError("Historical captures have the wrong image set.")
    records = all_records[:limit] if limit is not None else all_records
    expected = {r["stem"]: r for r in records}
    historical_budgets = json.loads((project_path(HISTORY) / "matched_fp_budgets.json").read_text())
    operating = {"floor": .001, **{item["label"]: item["models"]["baseline"]["confidence"]
                                  for item in historical_budgets}}
    # Confirm diagnostic operating points against the original strong control capture.
    identical, _ = recall_at_fp_budgets(all_records,
        {"baseline": saved_predictions["calibration"], "sdcl": saved_predictions["calibration"]},
        reference_size=640, budgets=(5, 10), bootstrap_samples=0)
    if any(item["models"]["baseline"] != historical_budgets[i]["models"]["baseline"]
           for i, item in enumerate(identical)):
        raise ValueError("Frozen calibration FP-budget operating points do not reproduce.")
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(output or f"output/analysis/headroom-{now:%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output}")
    output.mkdir(parents=True)
    hashes = source_hashes()
    plan = {
        "fixed_gates": GATES, "created_at": now.isoformat(), "source_hashes": hashes,
        "control_manifest_sha256": history["control_sha256"], "history_plan_sha256": HISTORY_PLAN_SHA256,
        "historical_input_hashes": {p.name: file_hash(p) for p in [project_path(HISTORY) / "decision.json",
            project_path(HISTORY) / "matched_fp_budgets.json"]},
        "raw_views_sha256": file_hash(raw_path), "records_sha256": record_hash(records),
        "validation_images": len(records), "fixed_operating_scores": operating,
        "oracle_rule": "Owned GT fixed by calibration IoU>=.5 greedy match at score>=.001; "
                       "choose highest same-GT IoU valid action; ties keep lowest index. Unowned rows stay unchanged.",
        "limitations": "GT-assisted diagnostics, never deployment predictions. Cached view pools are "
                       "already per-view NMS/conf/max500 filtered; absence is not proof of feature failure. "
                       "AP need not be globally maximized by per-owned-GT IoU, due to matching and area bins. "
                       "Miss causes are descriptive precedence rules, not identified causal write paths. "
                       "Repeatedly used val is not independent evidence.",
    }
    save_json(output / "plan.json", plan)
    archive_source(output, hashes)
    calibrator = load_calibrator(control)
    predictions = {name: {} for name in ("calibration", "gain", "gt_assisted_oracle")}
    rows, gt_rows, action_counts, timing_rows = [], [], np.zeros(6, dtype=int), []
    totals = {point: {group: dict.fromkeys(CAUSES, 0) for group in GROUPS} for point in operating}
    pool_counts = {group: {"gt": 0, "any_sameclass_coverage": 0, "max_one_to_one_matches": 0}
                   for group in ("all", "tiny", "small_all")}
    started = perf_counter()
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if raw["image"] not in expected:
                continue
            record = expected[raw["image"]]
            stem = record["stem"]
            if stem in predictions["calibration"]:
                raise ValueError("Duplicate raw image.")
            from sdcl.evaluation.slicing_screen import tile_windows
            if (raw["size"] != record["size"] or len(raw["views"]) != 5
                    or raw["windows"] != [[0, 0, *record["size"]],
                                           *[list(w) for w in tile_windows(*record["size"])]]):
                raise ValueError("Cached view dimensions/windows differ from the fixed policy.")
            ignores = json.loads((dataset / "metadata/val" / f"{stem}.json").read_text())["ignore_boxes_xyxy"]
            tick = perf_counter()
            c, origins, scores, base = calibrated_candidates(
                raw["views"], raw["windows"], record["size"], ignores, calibrator)
            if not np.array_equal(c, saved_predictions["calibration"][stem]):
                raise AssertionError("Frozen calibration replay differs.")
            actual_gain = saved_predictions["gain"][stem]
            if not np.array_equal(c[:, 4:], actual_gain[:, 4:]):
                raise ValueError("Historical gain scores/classes/count/order differ.")
            boxes, valid, _ = action_menu(c, origins, scores, base, raw["views"], record["size"], ignores)
            targets, owned = action_targets(record, c, boxes)
            selected = select_owned_oracle(targets, owned, valid)
            oracle = apply_actions(c, boxes, selected)
            action_counts += np.bincount(selected, minlength=6)
            for name, values in (("calibration", c), ("gain", actual_gain), ("gt_assisted_oracle", oracle)):
                predictions[name][stem] = values
            gt = np.asarray([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
            classes = [o["class_id"] for o in record["objects"]]
            areas = box_areas(gt) * (640 / max(record["size"]))**2
            owned_gt, _ = match_at_threshold(box_iou_matrix(gt, c[:, :4]), classes, c, .001)
            indices = np.flatnonzero((owned_gt >= 0) & group_mask(areas, "small_all"))
            image_row = {"image": stem, "owned_small_all": len(indices)}
            for name, values in (("calibration", c), ("oracle", oracle), ("gain", actual_gain)):
                ious = np.diag(box_iou_matrix(gt[indices], values[owned_gt[indices], :4]))
                image_row[f"{name}_iou_sum"] = float(ious.sum())
            rows.append(image_row)
            if image_row["oracle_iou_sum"] + 1e-7 < image_row["calibration_iou_sum"]:
                raise AssertionError("Oracle reduced same-owned-GT IoU.")
            clean, _ = clean_views(raw["views"], record["size"], ignores)
            before_ignore = uncluttered_raw_pool(raw["views"], record["size"])
            image_gt_rows = []
            for point, threshold in operating.items():
                causes, matched, info = miss_partition(record, c, clean, before_ignore, threshold)
                for group in GROUPS:
                    chosen = group_mask(areas, group)
                    for cause in CAUSES:
                        totals[point][group][cause] += int(sum(
                            cause == value and is_chosen for value, is_chosen in zip(causes, chosen)))
                if point == "floor":
                    for i, obj in enumerate(record["objects"]):
                        image_gt_rows.append({"image": stem, "object_id": obj.get("object_id", i),
                            "class_id": obj["class_id"], "reference_area": float(areas[i]),
                            "raw_best_sameclass_iou": float(info["raw_same_iou"][i]),
                            "raw_best_anyclass_iou": float(info["raw_any_iou"][i]),
                            "final_best_sameclass_iou": float(info["final_same_iou"][i]),
                            "floor_cause": causes[i]})
                    for group, counts in pool_counts.items():
                        chosen = group_mask(areas, group)
                        counts["gt"] += int(chosen.sum())
                        counts["any_sameclass_coverage"] += int((info["raw_same_iou"][chosen] >= .5).sum())
                        counts["max_one_to_one_matches"] += max_pool_matches(info["raw_edges"][chosen])
                else:
                    for row, cause in zip(image_gt_rows, causes):
                        row[f"{point}_cause"] = cause
            gt_rows.extend(image_gt_rows)
            timing_rows.append({"image": stem, "cached_diagnostic_ms": (perf_counter() - tick) * 1000})
            if len(rows) == 1 or len(rows) % 50 == 0 or len(rows) == len(records):
                print(f"Headroom diagnostic: {len(rows)}/{len(records)}, {perf_counter() - started:.1f}s.", flush=True)
    if any(set(mapping) != set(expected) for mapping in predictions.values()):
        raise ValueError("Incomplete diagnostic image coverage.")
    captures = {}
    for name, mapping in predictions.items():
        folder = output / "predictions" / name
        folder.mkdir(parents=True)
        with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"image": record["stem"], "boxes": mapping[record["stem"]].tolist()}) + "\n")
        captures[name] = {"predictions_sha256": file_hash(folder / "predictions.jsonl"),
                          "gt_assisted": name == "gt_assisted_oracle",
                          "warning": "DIAGNOSTIC ONLY; oracle uses validation GT and is not deployable."}
        save_json(folder / "manifest.json", captures[name])
    write_csv(output / "owned_gt_per_image.csv", rows)
    write_csv(output / "per_gt_bottlenecks.csv", gt_rows)
    write_csv(output / "processing_times.csv", timing_rows)
    for point, groups in totals.items():
        for group, counts in groups.items():
            total = sum(counts.values())
            expected_gt = sum(group_mask(box_areas([o["box_xyxy"] for o in r["objects"]])
                * (640 / max(r["size"]))**2, group).sum() for r in records)
            if total != expected_gt:
                raise AssertionError("Miss partition is not exhaustive.")
    bottlenecks = {point: {group: {"gt": sum(counts.values()), "counts": counts,
        "misses": sum(counts.values()) - counts["detected"],
        "share_of_misses": {cause: value / max(sum(counts.values()) - counts["detected"], 1)
                            for cause, value in counts.items() if cause != "detected"}}
        for group, counts in groups.items()} for point, groups in totals.items()}
    save_json(output / "bottlenecks.json", bottlenecks)
    save_json(output / "cached_pool_geometry.json", {
        "groups": pool_counts, "scope": "Score-free IoU>=.5 same-class bipartite matching on cached "
        "clean pools. Group-specific maxima independent, not jointly achievable; no max500 cap. "
        "Not deployable recall, pre-NMS detector recall, or predicted training benefit."})
    names = yaml.safe_load((dataset / "dataset.yaml").read_text())["names"]
    names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    metrics = {}
    for name, mapping in predictions.items():
        print(f"Reference640 AP for {name} (oracle is GT-assisted).", flush=True)
        metrics[name] = coco_size_metrics(records, mapping, names, reference_size=640)
        save_json(output / "coco_reference.json", metrics)
    if limit is None:
        original = json.loads((project_path(HISTORY) / "coco_reference.json").read_text())
        if any(metrics[name] != original[name] for name in ("calibration", "gain")):
            raise AssertionError("Historical AP not reproduced exactly.")
    mechanism = paired_owned_summary(rows)
    mechanism["action_counts"] = action_counts.tolist()
    save_json(output / "owned_gt_reference.json", mechanism)
    values = {f"{group}_oracle_ap_vs_calibration_pp": 100 * (
        metrics["gt_assisted_oracle"][group]["ap50_95"] - metrics["calibration"][group]["ap50_95"])
        for group in ("tiny", "small_all", "all")}
    values["owned_small_all_oracle_iou_vs_calibration_pp"] = mechanism["oracle_vs_calibration_pp"]
    values["owned_small_all_oracle_iou_ci_low_pp"] = mechanism["oracle_vs_calibration_paired_image_95ci_pp"][0]
    decision = decide_headroom(values, limit is None)
    if source_hashes() != hashes:
        raise ValueError("Diagnostic source changed during evaluation.")
    frozen_control()
    checked_history()
    verification = {
        "frozen_control_replayed_exactly": True, "history_dependencies_and_archive_match": True,
        "diagnostic_source_unchanged": True, "owned_gt_iou_nondecreasing": True,
        "miss_partitions_exhaustive": True, "previous_STOP_unchanged": previous == json.loads(
            (project_path(HISTORY) / "decision.json").read_text()),
        "same_scores_classes_count_order": all(np.array_equal(v[:, 4:], predictions["calibration"][stem][:, 4:])
            for mapping in predictions.values() for stem, v in mapping.items()),
        "records_unchanged": record_hash(load_records(dataset, "val")) == record_hash(all_records),
        "raw_views_unchanged": file_hash(raw_path) == plan["raw_views_sha256"],
        "prediction_hashes_match": all(file_hash(output / "predictions" / name / "predictions.jsonl")
            == capture["predictions_sha256"] for name, capture in captures.items()),
        "calibration_capture_bytes_match": limit is not None or captures["calibration"]["predictions_sha256"]
            == control["prediction_capture"]["sha256"],
    }
    if not all(verification.values()):
        raise RuntimeError("Diagnostic integrity check failed.")
    save_json(output / "verification.json", verification)
    save_json(output / "decision.json", decision)
    lines = ["# Fixed action-space headroom diagnostic", "", f"Decision: **{decision['decision']}**", "",
             "GT-assisted reference only; do not report oracle as a trained/deployable method.", "",
             "| Check | Value pp | Requirement | Pass |", "|---|---:|---:|---|"]
    for item in decision["checks"]:
        lines.append(f"| {item['metric']} | {item['value_pp']:.4f} | "
                     f"{item['comparison']} {item['threshold_pp']:.4f} | {item['passed']} |")
    lines.extend(["", decision["scope"], "", decision["next_action"], "", plan["limitations"], "",
                  "Previous localization STOP unchanged. No new training ran."])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "decision": decision["decision"], "checks": decision["checks"],
            "training_ran": False, "full_validation": limit is None}
