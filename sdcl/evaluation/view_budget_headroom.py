"""Finite view-omission inventory; all GT-dependent choices are diagnostic only."""

import json
from time import perf_counter

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import max_pool_matches
from sdcl.evaluation.native_censoring import pool_edges
from sdcl.evaluation.size_analysis import box_areas, file_hash, group_mask
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate25_view_budget_registration_20261014.json"
OUTPUT = "output/analysis/view-budget-headroom-fixed-20261014"
BASE = "output/analysis/class-choice-pilot-fixed-20261014"
NATIVE = "output/analysis/native-censoring-fixed-20261014"
GROUPS = ("all", "tiny", "small_all")
THRESHOLDS = (.5, .75)
ARMS = ("oracle", "global_empty", "global_density_one", "fixed_one")


def active_views(mask):
    return [0] + [v for v in range(1, 5) if mask & (1 << (v - 1))]


def control_masks(global_predictions, windows):
    center = (global_predictions[:, :2] + global_predictions[:, 2:4]) / 2
    counts = []
    mask = 0
    for view, (x1, y1, x2, y2) in enumerate(windows[1:], 1):
        inside = ((center >= [x1, y1]) & (center < [x2, y2])).all(1)
        mask |= (1 << (view - 1)) if (inside & (global_predictions[:, 4] >= .05)).any() else 0
        counts.append(int((inside & (global_predictions[:, 4] >= .25)).sum()))
    return {"global_empty": mask, "global_density_one": 15 ^ (1 << int(np.argmin(counts))),
            "fixed_one": 7}


def subset_inventory(edges, views, areas):
    result = {}
    full_support = {t: e.any(1) for t, e in edges.items()}
    full_capacity = {t: max_pool_matches(e) for t, e in edges.items()}
    for mask in range(16):
        keep = np.isin(views, active_views(mask))
        values = {}
        safe = True
        for threshold, edge in edges.items():
            remaining = edge[:, keep]
            support = remaining.any(1)
            capacity = max_pool_matches(remaining)
            lost = full_support[threshold] & ~support
            safe &= not lost.any() and capacity == full_capacity[threshold]
            values[str(threshold)] = {
                "all_max_matches": capacity,
                "lost_supported_gt": {g: int((lost & group_mask(areas, g)).sum()) for g in GROUPS},
            }
        result[mask] = {"views": active_views(mask), "safe": bool(safe), "metrics": values}
    chosen = min((mask for mask, row in result.items() if row["safe"]),
                 key=lambda mask: (len(result[mask]["views"]), mask))
    return chosen, result


def evaluate(output=OUTPUT):
    started = perf_counter()
    base, native = project_path(BASE), project_path(NATIVE)
    _, base_plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    _, plan = verify_record(native, "docs/experiments/candidate21_native_censoring_result_20261014.json")
    stems = plan["images"]
    if stems != base_plan["image_selection"]["holdout"][:64]:
        raise ValueError("Fixed64 image order changed.")
    raw_path = base / "raw_views.jsonl"
    if file_hash(raw_path) != plan["prior_raw_views_sha256"]:
        raise ValueError("Pinned full-image windows changed.")
    raw = {r["image"]: r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
    records = {r["stem"]: r for r in base_plan["records"]["holdout"]}
    manifest = json.loads((native / "frames.json").read_text())
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/evaluation/view_budget_headroom.py",
             "scripts/evaluation/diagnose_view_budget_headroom.py",
             "sdcl/evaluation/native_censoring.py", "sdcl/evaluation/headroom_screen.py",
             "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "images": stems,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate21_native_censoring_result_20261014.json")),
    })
    totals = {arm: {
        "forwards": 0, "images_with_omission": 0, "safe_images": 0,
        "lost_supported_gt": {str(t): dict.fromkeys(GROUPS, 0) for t in THRESHOLDS},
        "capacity_loss": dict.fromkeys((str(t) for t in THRESHOLDS), 0),
    } for arm in ARMS}
    frames, counts = {}, dict.fromkeys(GROUPS, 0)
    for index, stem in enumerate(stems, 1):
        snapshot = manifest[stem]
        path = native / snapshot["file"]
        if file_hash(path) != snapshot["sha256"]:
            raise ValueError("Frozen native predictions changed.")
        with np.load(path, allow_pickle=False) as c:
            pool, view = c["kept"].copy(), c["kept_view"].copy()
        masks = control_masks(pool[view == 0], raw[stem]["windows"])
        # Prediction-only controls are fixed before reading objects for the oracle.
        meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        if file_hash(meta_path) != records[stem]["metadata_sha256"]:
            raise ValueError("Frozen diagnostic labels changed.")
        meta = json.loads(meta_path.read_text())
        gt = np.asarray([o["box_xyxy"] for o in meta["valid_objects"]], float).reshape(-1, 4)
        classes = np.asarray([o["class_id"] for o in meta["valid_objects"]], int)
        areas = box_areas(gt) * (640 / max(meta["original_size"])) ** 2
        edges = {t: pool_edges(gt, classes, pool, .05, t) for t in THRESHOLDS}
        masks["oracle"], inventory = subset_inventory(edges, view, areas)
        for group in GROUPS:
            counts[group] += int(group_mask(areas, group).sum())
        for arm, mask in masks.items():
            row = inventory[mask]
            totals[arm]["forwards"] += len(row["views"])
            totals[arm]["images_with_omission"] += int(mask != 15)
            totals[arm]["safe_images"] += int(row["safe"])
            for threshold in THRESHOLDS:
                key = str(threshold)
                for group in GROUPS:
                    totals[arm]["lost_supported_gt"][key][group] += row["metrics"][key][
                        "lost_supported_gt"][group]
                totals[arm]["capacity_loss"][key] += inventory[15]["metrics"][key][
                    "all_max_matches"] - row["metrics"][key]["all_max_matches"]
        frames[stem] = {"chosen_masks": masks, "inventory": inventory}
        if index == 1 or index % 16 == 0:
            print(f"View-budget inventory: {index}/64, {perf_counter()-started:.1f}s", flush=True)
    metrics = {}
    for arm, values in totals.items():
        metrics[arm] = {
            **values, "forward_count_saving_percent": (1 - values["forwards"] / (5 * len(stems))) * 100,
            "lost_supported_gt_pp": {key: {g: n / max(counts[g], 1) * 100 for g, n in groups.items()}
                                     for key, groups in values["lost_supported_gt"].items()},
        }
    policy, oracle = registration["policy"], metrics["oracle"]
    checks = {
        "forward_saving_headroom": oracle["forward_count_saving_percent"] >=
        policy["oracle_forward_count_saving_percent_min"],
        "omission_not_isolated": oracle["images_with_omission"] >= policy["oracle_images_with_omission_min"],
    }
    verdict = {"decision": "BOUNDED_VIEW_POLICY_SCREEN_ALLOWED" if all(checks.values()) else
               "STOP_THIS_STRICT_EVIDENCE_PRESERVING_OMISSION_RECIPE",
               "checks": checks, "scope": "GT-assisted coverage inventory, not deployment, latency or AP."}
    save_json(output / "frames.json", frames)
    save_json(output / "metrics.json", {"gt": counts, "arms": metrics})
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": len(stems), "enumerated_subsets_per_image": 16,
        "zero_detector_forwards_zero_training": True, "no_val_no_testdev_no_threshold_search": True,
        "cache_has_diagnostic_ignore_filter": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
