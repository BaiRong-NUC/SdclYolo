"""No-fitting screen for local same-class scale information in action quality."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.class_choice_signal import weighted_auc
from sdcl.evaluation.size_analysis import file_hash, group_mask
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.neighbor_scale import scale_context
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate32_neighbor_scale_signal_registration_20261014.json"
OUTPUT = "output/analysis/neighbor-scale-signal-fixed-20261014"
SCORES = ("local_scale", "global_scale", "reference_distance")
GROUPS = ("all", "tiny", "small_all")


def summarize(frames):
    result = {}
    for group in GROUPS:
        collected = {k: [] for k in ("labels", "weights", *SCORES)}
        anchors, positive_anchors = 0, 0
        for f in frames:
            reference = f["regret"]
            nominal = f["ious"][:, 0]
            delta = nominal-nominal[np.arange(len(reference)), reference, None]
            eligible = ((f["owner"] >= 0) & (f["candidates"][:, 4] >= .05) &
                        (f["valid"].sum(1) >= 2) & f["supported"] & group_mask(f["areas"], group))
            valid = f["valid"] & eligible[:, None] & (np.abs(delta) > .01)
            valid[np.arange(len(reference)), reference] = False
            weight = np.broadcast_to(1/np.maximum(valid.sum(1), 1)[:, None], valid.shape)
            labels = delta > .01
            anchors += int(valid.any(1).sum())
            positive_anchors += int((valid & labels).any(1).sum())
            collected["labels"].append(labels[valid])
            collected["weights"].append(weight[valid])
            for score in SCORES:
                relative = f[score]-f[score][np.arange(len(reference)), reference, None]
                collected[score].append(relative[valid])
        joined = {k: np.concatenate(v) for k, v in collected.items()}
        result[group] = {
            "context_supported_comparable_anchors": anchors, "anchors_with_better_actions": positive_anchors,
            "positive_actions": int(joined["labels"].sum()),
            "negative_actions": int((~joined["labels"]).sum()),
            **{s+"_weighted_auc": weighted_auc(joined["labels"], joined[s], joined["weights"]) for s in SCORES}}
    return result


def decide(metrics, policy):
    tiny = metrics["tiny"]
    local = tiny["local_scale_weighted_auc"]
    gain = {a: local-tiny[a+"_weighted_auc"] if
            local is not None and tiny[a+"_weighted_auc"] is not None else None
            for a in ("global_scale", "reference_distance")}
    checks = {
        "positive_actions": tiny["positive_actions"] >= policy["tiny_positive_actions_min"],
        "comparisons": tiny["positive_actions"]+tiny["negative_actions"] >= policy["tiny_total_comparisons_min"],
        "discrimination": local is not None and local >= policy["tiny_local_scale_auc_min"],
        "spatial_context": gain["global_scale"] is not None and gain["global_scale"] >
        policy["tiny_local_auc_vs_global_min_exclusive"],
        "geometric_control": gain["reference_distance"] is not None and gain["reference_distance"] >
        policy["tiny_local_auc_vs_distance_min_exclusive"],
    }
    return {"decision": "MATCHED_NEIGHBOR_SCALE_RESIDUAL_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_LOCAL_SCALE_MEDIAN_REPRESENTATION", "checks": checks,
            "values": {"tiny_local_scale_auc": local, "tiny_auc_gains": gain,
                       "tiny_positive_actions": tiny["positive_actions"]},
            "scope": "Reused train-head-holdout owned-action information proxy, not AP or novelty."}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/postprocess/neighbor_scale.py", "sdcl/evaluation/neighbor_scale_signal.py",
             "scripts/evaluation/diagnose_neighbor_scale_signal.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/evaluation/class_choice_signal.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads = load_frames()
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": prior["image_selection"],
        "source_archive_sha256": file_hash(output / "source.zip"), "heads": heads,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json"))})
    (output / "features").mkdir()
    manifest = {}
    for split, records in frames.items():
        for f in records:
            peer = apply_actions(f["candidates"], f["boxes"], f["regret"])
            context = scale_context(f["candidates"], f["boxes"], peer, f["regret"],
                                    f["metadata"]["original_size"])
            f.update(context)
            path = output / "features" / (f["stem"]+".npz")
            np.savez_compressed(path, **context)
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(),
                                    "sha256": file_hash(path), "split": split}
        print(f"Neighbor scale {split}: {len(records)} frames, {perf_counter()-started:.1f}s", flush=True)
    metrics = {k: summarize(v) for k, v in frames.items()}
    verdict = decide(metrics["holdout"], registration["policy"])
    save_json(output / "frames.json", manifest)
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "fit_pool_images_descriptive_only": 128, "head_holdout_images": 64, "detector_forwards": 0,
        "image_decodes": 0, "training_steps": 0, "GT_excluded_from_context_features": True,
        "same_original_GT_owner_all_actions": True, "each_comparable_anchor_total_weight1": True,
        "cache_inherits_historical_diagnostic_ignore_filter": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "metrics": metrics["holdout"]}
