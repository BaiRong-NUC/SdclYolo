"""GT-assisted audit of zero-target near misses within the original residual budget."""

import json
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold
from sdcl.postprocess.rescue import save_json
from sdcl.training.geometry_residual_confirmation import prepare
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate43_unowned_reachability_registration_20261014.json"
OUTPUT = "output/analysis/unowned-reachability-fixed-20261014"
GEOMETRY = "output/analysis/geometry-residual-confirmation-fixed-20261014"
THRESHOLDS = np.linspace(.5, .95, 10)
GROUPS = ("all", "tiny", "small_all")
POOLS = ("all_unused", "unowned", "unique_unowned")


def coordinate_bounds(regret, scale, size):
    regret, scale = np.asarray(regret), np.asarray(scale, float)
    delta = float(np.float32(.1))*scale
    lower, upper = regret[:, :4].astype(float)-delta, regret[:, :4].astype(float)+delta
    width, height = size
    maximum = np.array([width, height, width, height])
    lower = np.clip(lower, 0, maximum).astype(np.float32).astype(float)
    upper = np.clip(upper, 0, maximum).astype(np.float32).astype(float)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Nonfinite original coordinate budget.")
    if np.any(upper[:, :2] >= lower[:, 2:]):
        raise ValueError("Coordinate intervals permit crossed/degenerate boxes.")
    return lower, upper


def projected_iou(gt, lower, upper):
    """For ordered edge intervals, moving each edge toward GT never reduces IoU."""
    gt = np.asarray(gt, float).reshape(-1, 4)
    projected = np.clip(gt[:, None], lower[None], upper[None])
    lo = np.maximum(gt[:, None, :2], projected[:, :, :2])
    hi = np.minimum(gt[:, None, 2:], projected[:, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(-1)
    area = np.maximum(projected[:, :, 2:]-projected[:, :, :2], 0).prod(-1)
    return intersection/np.maximum(box_areas(gt)[:, None]+area-intersection, 1e-12)


def cardinality(edges):
    if not edges.size or not edges.any():
        return 0
    return int((maximum_bipartite_matching(csr_matrix(edges), perm_type="column") >= 0).sum())


def score_frame(gt, classes, prediction, regret, scale, active, owner, size):
    lower, upper = coordinate_bounds(regret, scale, size)
    if np.any(prediction[active, :4] < lower[active]) or np.any(prediction[active, :4] > upper[active]):
        raise ValueError("Frozen geometry escaped the original total correction budget.")
    best = projected_iou(gt, lower, upper)
    same = classes[:, None] == prediction[None, :, 5]
    possible = (best >= .5) & same
    unique = possible.sum(0) == 1
    high = active & (prediction[:, 4] >= .25)
    areas = box_areas(gt)*(640/max(size))**2
    current = box_iou_matrix(gt, prediction[:, :4])
    trace, results = {}, {g: {"gt": int(group_mask(areas, g).sum()), "thresholds": []} for g in GROUPS}
    for index, threshold in enumerate(THRESHOLDS):
        matches, used = match_at_threshold(current, classes, prediction, .25, float(threshold))
        missing = matches < 0
        unused = high & ~used
        pools = {"all_unused": unused, "unowned": unused & (owner < 0),
                 "unique_unowned": unused & (owner < 0) & unique}
        edges = (best >= threshold) & same
        trace[f"matches_{index}"] = matches
        trace[f"unused_{index}"] = unused
        for g, values in results.items():
            include = group_mask(areas, g)
            query = include & missing
            row = {"iou": float(threshold), "baseline_tp": int((include & ~missing).sum()),
                   **{name: cardinality(edges[np.ix_(query, members)]) for name, members in pools.items()}}
            if not row["unique_unowned"] <= row["unowned"] <= row["all_unused"] <= int(query.sum()):
                raise ValueError("Nested oracle pool cardinalities are inconsistent.")
            values["thresholds"].append(row)
    trace.update(max_reachable_iou=best, unique_at050=unique, classes=classes, areas=areas,
                 owner=owner, active=active, lower=lower, upper=upper)
    return results, trace


def summarize(frames):
    metrics = {}
    for g in GROUPS:
        rows = [f[g] for f in frames]
        gt = sum(r["gt"] for r in rows)
        thresholds = []
        for index, threshold in enumerate(THRESHOLDS):
            thresholds.append({"iou": float(threshold),
                               **{k: sum(r["thresholds"][index][k] for r in rows) for k in ("baseline_tp", *POOLS)}})
        metrics[g] = {"gt": gt, "thresholds": thresholds,
                      "mean_ten_threshold_headroom_pp": {
                          p: float(np.mean([t[p] for t in thresholds]))/max(gt, 1)*100 for p in POOLS},
                      "unique_unowned_images050": sum(r["thresholds"][0]["unique_unowned"] > 0 for r in rows),
                      "unique_unowned_fraction_of_all_unused050":
                      thresholds[0]["unique_unowned"]/max(thresholds[0]["all_unused"], 1)}
    return metrics


def decide(metrics, policy):
    tiny = metrics["tiny"]
    checks = {
        "tiny_unique_count": tiny["thresholds"][0]["unique_unowned"] >= policy["tiny_unique_additional_gt050_min"],
        "image_spread": tiny["unique_unowned_images050"] >= policy["tiny_unique_images050_min"],
        "ten_threshold_headroom": tiny["mean_ten_threshold_headroom_pp"]["unique_unowned"] >=
        policy["tiny_unique_mean_ten_threshold_headroom_min_pp"],
        "fraction_of_all_reachable": tiny["unique_unowned_fraction_of_all_unused050"] >=
        policy["tiny_unique_fraction_of_all_unused050_min"],
    }
    return {"decision": "BOUNDED_UNOWNED_SUPERVISION_CONTROL_ALLOWED" if all(checks.values()) else
            "STOP_THIS_BOUNDED_UNOWNED_SUPERVISION_ROUTE", "checks": checks,
            "scope": "GT-assisted target coverage lead only, not a feasible gain, AP bound or innovation."}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"frames").mkdir()
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/evaluation/unowned_reachability.py", "scripts/evaluation/diagnose_unowned_reachability.py",
             "sdcl/training/geometry_residual_confirmation.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    prior_record, _ = verify_record(project_path(GEOMETRY),
                                    "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json")
    frames, prior, heads, _, _ = prepare()
    frames = frames["holdout"]
    entries39 = json.loads((project_path(GEOMETRY)/"frames-seed0.json").read_text())
    save_json(output/"plan.json", {"registration": registration, "source_hashes": hashes, "heads": heads,
                                 "source_archive_sha256": file_hash(output/"source.zip"),
                                 "images": prior["image_selection"]["holdout"],
                                 "prior_result_sha256": file_hash(project_path(
                                     "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json"))})
    all_results, manifest = [], {}
    zero_target_rows = 0
    for f in frames:
        entry = entries39[f["stem"]]
        path = project_path(GEOMETRY)/entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Frozen geometry39 prediction trace changed.")
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["geometry"], f["seed0_geometry"]) or not np.array_equal(saved["regret"], f["reference"]):
                raise ValueError("Geometry38/39 or regret traces disagree.")
        no_owner = f["active"] & (f["owner"] < 0)
        if np.any(f["target"][no_owner] != 0):
            raise ValueError("Unowned rows no longer have zero original targets.")
        zero_target_rows += int(no_owner.sum())
        objects = f["metadata"]["valid_objects"]
        gt = np.array([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
        classes = np.array([o["class_id"] for o in objects], int)
        results, trace = score_frame(gt, classes, f["seed0_geometry"], f["reference"], f["scale"],
                                     f["active"], f["owner"], f["metadata"]["original_size"])
        all_results.append(results)
        path = output/"frames"/(f["stem"]+".npz")
        np.savez_compressed(path, gt=gt, **trace)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
    metrics = summarize(all_results)
    verdict = decide(metrics, registration["policy"])
    for name, value in (("frames", manifest), ("per_image", all_results), ("metrics", metrics), ("decision", verdict)):
        save_json(output/(name+".json"), value)
    save_json(output/"verification.json", {"images": len(frames), "detector_forwards": 0, "image_decodes": 0,
                                          "fitting_steps": 0, "active_unowned_zero_target_rows": zero_target_rows,
                                          "all64_geometry38_and39_predictions_identical": True,
                                          "budget_total_relative_to_original_regret_not_an_extra_step": True,
                                          "all_existing_geometry_coordinates_inside_budget": True,
                                          "current_matching_and_row_capacity_respected_in_oracle_graph": True,
                                          "all_GT_used_for_uniqueness_including_already_matched": True,
                                          "prior_STOP_preserved": prior_record["decision"]["decision"],
                                          "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                                          "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "metrics": metrics}
