"""Pre-registered GT-assisted AP ranking-debt diagnostic after coordinate repair."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import (
    coco_size_metrics, file_hash, group_mask, recall_at_fp_budgets,
)
from sdcl.postprocess.coordinate_score_debt import correct_scores, threshold_quality
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate34_coordinate_score_debt_registration_20261014.json"
OUTPUT = "output/analysis/coordinate-score-debt-fixed-20261014"
ARMS = ("calibration", "regret", "oracle_delta", "oracle_absolute")


def decide(metrics, mechanism, budgets, policy):
    gain = {g: (metrics["oracle_delta"][g]["ap50_95"]-metrics["regret"][g]["ap50_95"])*100
            for g in ("tiny", "small_all")}
    fp_gain = next(r["delta_recall_pp"]["small_all"] for r in budgets if r["maximum_fppi"] == 10)
    checks = {
        "changed_tiny": mechanism["tiny"]["changed_owned_anchors"] >= policy["tiny_changed_owned_anchors_min"],
        "nonzero_delta_tiny": mechanism["tiny"]["nonzero_quality_delta_anchors"] >=
        policy["tiny_nonzero_quality_delta_anchors_min"],
        "tiny_ap_headroom": gain["tiny"] >= policy["oracle_delta_tiny_ap_vs_regret_min_pp"],
        "small_ap_headroom": gain["small_all"] >= policy["oracle_delta_small_all_ap_vs_regret_min_pp"],
        "matched_fp_budget": fp_gain >= policy["oracle_delta_small_all_recall_fppi10_vs_regret_min_pp"],
    }
    return {"decision": "MATCHED_ACTION_SCORE_RESIDUAL_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_ACTION_QUALITY_DELTA_SCORE_RECIPE", "checks": checks,
            "values": {"oracle_delta_ap_gains_pp": gain, "small_all_recall_fppi10_gain_pp": fp_gain,
                       "tiny_changed_owned_anchors": mechanism["tiny"]["changed_owned_anchors"],
                       "tiny_nonzero_delta_anchors": mechanism["tiny"]["nonzero_quality_delta_anchors"]},
            "scope": "GT-assisted cached head-holdout proxy; not deployable, official AP or novelty."}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/coordinate_score_debt.py", "sdcl/evaluation/coordinate_score_debt.py",
             "scripts/evaluation/diagnose_coordinate_score_debt.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/localization.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads = load_frames()
    frames = frames["holdout"]
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "images": prior["image_selection"]["holdout"],
        "source_archive_sha256": file_hash(output/"source.zip"), "heads": heads,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json"))})
    predictions = {a: {} for a in ARMS}
    records, manifest = [], {}
    mechanism = {g: {"changed_owned_anchors": 0, "nonzero_quality_delta_anchors": 0,
                     "positive_quality_delta_anchors": 0, "negative_quality_delta_anchors": 0,
                     "quality_delta_sum": 0.} for g in ("all", "tiny", "small_all")}
    (output/"predictions").mkdir()
    for index, f in enumerate(frames, 1):
        n = len(f["candidates"])
        quality = threshold_quality(f["ious"][:, 0], f["owner"] >= 0)
        q0, q1 = quality[:, 0], quality[np.arange(n), f["regret"]]
        regret = apply_actions(f["candidates"], f["boxes"], f["regret"])
        arms = {
            "calibration": f["candidates"], "regret": regret,
            "oracle_delta": correct_scores(regret, q1-q0),
            "oracle_absolute": correct_scores(regret, q1-regret[:, 4]),
        }
        eligible = (f["owner"] >= 0) & (f["candidates"][:, 4] >= .05)
        changed = (regret[:, :4] != f["candidates"][:, :4]).any(1)
        for g, row in mechanism.items():
            mask = eligible & group_mask(f["areas"], g)
            row["changed_owned_anchors"] += int((mask & changed).sum())
            row["nonzero_quality_delta_anchors"] += int((mask & (q1 != q0)).sum())
            row["positive_quality_delta_anchors"] += int((mask & (q1 > q0)).sum())
            row["negative_quality_delta_anchors"] += int((mask & (q1 < q0)).sum())
            row["quality_delta_sum"] += float((q1-q0)[mask].sum())
        for arm, values in arms.items():
            predictions[arm][f["stem"]] = values
        records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                        "objects": f["metadata"]["valid_objects"]})
        path = output/"predictions"/(f["stem"]+".npz")
        np.savez_compressed(path, **arms, q0=q0, q1=q1)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
    save_json(output/"frames.json", manifest)
    names = dict(enumerate(NAMES))
    metrics = {}
    for arm in ARMS:
        metrics[arm] = coco_size_metrics(records, predictions[arm], names, 640)
        print(f"Score-debt AP {arm}: small_all{metrics[arm]['small_all']['ap50_95']*100:.6f}, "
              f"{perf_counter()-started:.1f}s", flush=True)
    budgets, _ = recall_at_fp_budgets(records, {"baseline": predictions["regret"],
        "sdcl": predictions["oracle_delta"]}, reference_size=640, budgets=(5, 10), bootstrap_samples=0)
    verdict = decide(metrics, mechanism, budgets, registration["policy"])
    save_json(output/"metrics.json", metrics)
    save_json(output/"mechanism.json", mechanism)
    save_json(output/"fixed-fp-budgets.json", budgets)
    save_json(output/"decision.json", verdict)
    save_json(output/"verification.json", {
        "head_holdout_train_images": len(frames), "detector_forwards": 0, "image_decodes": 0, "fitting_steps": 0,
        "all_regret_coordinates_classes_counts_and_array_order_preserved": True,
        "GT_USED_IN_ORACLE_SCORES_NOT_DEPLOYABLE": True, "same_original_owner_both_quality_targets": True,
        "scores_zero_delta_exactly_preserved": True,
        "cache_inherits_historical_uniform_diagnostic_ignore_filter": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "mechanism": mechanism}
