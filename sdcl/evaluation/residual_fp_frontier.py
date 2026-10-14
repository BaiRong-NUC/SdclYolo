"""Attribute fixed-FP losses without changing the frozen residual predictions."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold, recall_at_fp_budgets,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate40_residual_fp_frontier_registration_20261014.json"
OUTPUT = "output/analysis/residual-fp-frontier-fixed-20261014"
PRIOR = "output/analysis/geometry-residual-confirmation-fixed-20261014"
CATEGORIES = ("same_row_geometry", "matching_competition", "budget_cutoff_only")
GROUPS = ("all", "tiny", "small_all")


def transitions(gt, classes, baseline, corrected, baseline_cutoff, corrected_cutoff):
    old_iou, new_iou = box_iou_matrix(gt, baseline[:, :4]), box_iou_matrix(gt, corrected[:, :4])
    old, _ = match_at_threshold(old_iou, classes, baseline, baseline_cutoff)
    common, _ = match_at_threshold(new_iou, classes, corrected, baseline_cutoff)
    new, _ = match_at_threshold(new_iou, classes, corrected, corrected_cutoff)
    lost = (old >= 0) & (new < 0)
    recovered = (old < 0) & (new >= 0)
    category = np.full(len(gt), -1, int)
    category[lost & (common >= 0)] = 2
    for g in np.flatnonzero(lost & (common < 0)):
        category[g] = 0 if new_iou[g, old[g]] < .5 else 1
    if not np.array_equal(category >= 0, lost):
        raise ValueError("Lost-GT attribution is not exhaustive.")
    return {"old": old, "common": common, "new": new, "lost": lost, "recovered": recovered,
            "category": category}


def route(metrics, policy):
    row = metrics[policy["primary_seed"]][policy["primary_group"]]
    count = row["lost"]
    name = max(CATEGORIES, key=lambda k: row["lost_categories"][k])
    fraction = row["lost_categories"][name]/max(count, 1)
    enough = count >= policy["lost_gt_min"]
    dominant = fraction >= policy["dominant_lost_category_fraction_min"]
    return {"decision": name.upper()+"_MECHANISM_SEARCH_LEAD" if enough and dominant else
            "NO_DOMINANT_FIXED_FP_LOSS_ROUTE",
            "checks": {"enough_actual_losses": enough, "dominant_category": dominant},
            "values": {"actual_lost_GT": count, "largest_category": name, "fraction": fraction},
            "scope": "Diagnostic lead only. No changed predictions, gate, STOP, training or official authorization."}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"frames").mkdir()
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/evaluation/residual_fp_frontier.py",
             "scripts/evaluation/diagnose_residual_fp_frontier.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    base = project_path(PRIOR)
    verify_record(base, "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json")
    frames, prior, heads = load_frames()
    frames = frames["holdout"]
    records = [{"id": i, "stem": f["stem"], "size": f["metadata"]["original_size"],
                "objects": f["metadata"]["valid_objects"]} for i, f in enumerate(frames, 1)]
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "heads": heads,
        "source_archive_sha256": file_hash(output/"source.zip"), "images": prior["image_selection"]["holdout"],
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json"))})
    metrics, manifest, cutoffs = {}, {}, {}
    for seed in ("0", "1", "2"):
        entries = json.loads((base/f"frames-seed{seed}.json").read_text())
        predictions = {"baseline": {}, "sdcl": {}}
        for f in frames:
            entry = entries[f["stem"]]
            path = base/entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Frozen residual predictions changed.")
            with np.load(path, allow_pickle=False) as c:
                predictions["baseline"][f["stem"]] = c["regret"].copy()
                predictions["sdcl"][f["stem"]] = c["geometry"].copy()
        budgets, _ = recall_at_fp_budgets(records, predictions, reference_size=640, budgets=(10,), bootstrap_samples=0)
        historical = json.loads((base/f"fixed-fp-budgets-seed{seed}.json").read_text())
        if budgets != historical:
            raise ValueError("Exact historical FP budget replay differs.")
        bcut = budgets[0]["models"]["baseline"]["confidence"]
        ncut = budgets[0]["models"]["sdcl"]["confidence"]
        cutoffs[seed] = {"baseline": bcut, "corrected": ncut}
        totals = {g: {"gt": 0, "baseline_tp": 0, "common_cutoff_tp": 0, "actual_corrected_tp": 0,
                      "lost": 0, "recovered": 0, "lost_categories": dict.fromkeys(CATEGORIES, 0)}
                  for g in GROUPS}
        for f in frames:
            gt = np.array([o["box_xyxy"] for o in f["metadata"]["valid_objects"]], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in f["metadata"]["valid_objects"]], int)
            areas = box_areas(gt)*(640/max(f["metadata"]["original_size"]))**2
            trace = transitions(gt, classes, predictions["baseline"][f["stem"]], predictions["sdcl"][f["stem"]], bcut, ncut)
            for g, row in totals.items():
                include = group_mask(areas, g)
                row["gt"] += int(include.sum())
                for key, name in (("old", "baseline_tp"), ("common", "common_cutoff_tp"), ("new", "actual_corrected_tp")):
                    row[name] += int((include & (trace[key] >= 0)).sum())
                for key in ("lost", "recovered"):
                    row[key] += int((include & trace[key]).sum())
                for k, name in enumerate(CATEGORIES):
                    row["lost_categories"][name] += int((include & (trace["category"] == k)).sum())
            path = output/"frames"/(seed+"-"+f["stem"]+".npz")
            np.savez_compressed(path, **trace, gt=gt, classes=classes, areas=areas)
            manifest[seed+":"+f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
        for row in totals.values():
            row["geometry_net_tp_at_common_cutoff"] = row["common_cutoff_tp"]-row["baseline_tp"]
            row["cutoff_only_net_tp"] = row["actual_corrected_tp"]-row["common_cutoff_tp"]
            row["actual_net_tp"] = row["actual_corrected_tp"]-row["baseline_tp"]
            if (sum(row["lost_categories"].values()) != row["lost"] or
                    row["recovered"]-row["lost"] != row["actual_net_tp"]):
                raise ValueError("FP-frontier counts do not reconcile.")
        metrics[seed] = totals
        print(f"Residual frontier seed{seed}: {json.dumps(totals['small_all'])}", flush=True)
    verdict = route(metrics, registration["routing_gates"])
    save_json(output/"metrics.json", metrics)
    save_json(output/"cutoffs.json", cutoffs)
    save_json(output/"frames.json", manifest)
    save_json(output/"decision.json", verdict)
    save_json(output/"verification.json", {
        "images": len(frames), "seeds": 3, "detector_forwards": 0, "image_decodes": 0, "fitting_steps": 0,
        "all_saved_fixed_FP_budgets_replayed_exactly": True, "lost_categories_exclusive_and_exhaustive": True,
        "common_cutoff_counterfactual_not_substituted_for_actual_FP_metric": True,
        "prior_STOPs_and_predictions_preserved": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "metrics": metrics, "cutoffs": cutoffs}
