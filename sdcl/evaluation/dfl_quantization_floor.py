"""Fixed cached screen of quantization-adjusted cross-view reliability."""

from itertools import combinations
import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.class_choice_signal import weighted_auc
from sdcl.evaluation.size_analysis import box_areas, file_hash, group_mask
from sdcl.postprocess.dfl_quantization_floor import distribution_reliability, original_coordinates
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate37_dfl_quantization_floor_registration_20261014.json"
OUTPUT = "output/analysis/dfl-quantization-floor-fixed-20261014"
NATIVE = "output/analysis/dfl-action-signal-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
SCORES = ("excess_variance", "raw_variance", "entropy", "consensus", "regret_agreement")
PAIRS = np.array(list(combinations(range(5), 2)))
GROUPS = ("all", "tiny", "small_all")


def paired_iou(left, right):
    intersection = np.maximum(np.minimum(left[..., 2:], right[..., 2:]) -
                              np.maximum(left[..., :2], right[..., :2]), 0).prod(-1)
    area_left = np.maximum(left[..., 2:]-left[..., :2], 0).prod(-1)
    area_right = np.maximum(right[..., 2:]-right[..., :2], 0).prod(-1)
    return intersection / np.maximum(area_left+area_right-intersection, 1e-12)


def geometry_scores(boxes, available, regret):
    overlap = paired_iou(boxes[:, None], boxes[None])
    valid = available[:, None] & available[None]
    valid &= ~np.eye(5, dtype=bool)[:, :, None]
    consensus = (overlap*valid).sum(1)/np.maximum(valid.sum(1), 1)
    return {"consensus": consensus, "regret_agreement": paired_iou(boxes, regret[None, :, :4])}


def decide(metrics, policy):
    tiny, small = metrics["tiny"], metrics["small_all"]
    auc = tiny["excess_variance_auc"]
    gains = {s: None if auc is None or tiny[s+"_auc"] is None else auc-tiny[s+"_auc"]
             for s in ("raw_variance", "consensus", "regret_agreement")}
    small_gain = (None if small["excess_variance_auc"] is None else
                  small["excess_variance_auc"]-small["raw_variance_auc"])
    checks = {
        "informative_tiny_rows": tiny["informative_rows"] >= policy["tiny_informative_rows_min"],
        "tiny_discrimination": auc is not None and auc >= policy["tiny_excess_auc_min"],
        **{"tiny_vs_"+s: gains[s] is not None and gains[s] >= policy["tiny_auc_vs_"+s+"_min"]
           for s in gains},
        "small_vs_raw": small_gain is not None and small_gain >= policy["small_all_auc_vs_raw_variance_min"],
    }
    return {
        "decision": ("MATCHED_EXCESS_VARIANCE_PILOT_ALLOWED" if all(checks.values()) else
                     "STOP_THIS_DFL_QUANTIZATION_FLOOR_RELIABILITY_RECIPE"),
        "checks": checks, "tiny_auc_gains": gains, "small_auc_gain_vs_raw": small_gain,
        "scope": "Conditional cached same-owner view ordering; no AP, deployment, novelty or detector training authorization.",
    }


def run(output=OUTPUT):
    tick = perf_counter()
    torch.set_num_threads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"frames").mkdir()
    paths = (REGISTRATION, "sdcl/postprocess/dfl_quantization_floor.py",
             "sdcl/evaluation/dfl_quantization_floor.py", "scripts/evaluation/diagnose_dfl_quantization_floor.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/class_choice_signal.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/localization.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, menu, heads = load_frames()
    frames = frames["holdout"]
    native = project_path(NATIVE)
    _, native_plan = verify_record(native, "docs/experiments/candidate24_dfl_action_signal_result_20261014.json")
    raw = project_path(RAW)
    verify_record(raw, "docs/experiments/candidate12_class_choice_result_20261014.json")
    if native_plan["images"] != menu["image_selection"]["holdout"]:
        raise ValueError("Historical native trace selection differs.")
    raw_path = raw/"raw_views.jsonl"
    if file_hash(raw_path) != native_plan["raw_views_sha256"]:
        raise ValueError("Historical raw windows changed.")
    windows = {r["image"]: r["windows"] for r in (json.loads(x) for x in raw_path.read_text().splitlines())}
    traces = json.loads((native/"frames.json").read_text())
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "heads": heads,
        "source_archive_sha256": file_hash(output/"source.zip"), "images": menu["image_selection"]["holdout"],
        "native_result_sha256": file_hash(project_path("docs/experiments/candidate24_dfl_action_signal_result_20261014.json")),
        "native_frames_manifest_sha256": file_hash(native/"frames.json"),
        "raw_windows_sha256": file_hash(raw_path)})
    accumulated = {g: {"label": [], "weight": [], "informative_rows": 0,
                      **{s: [] for s in SCORES}} for g in GROUPS}
    mechanism = {"available_owned_views": 0, "variance_sum": 0., "interpolation_floor_sum": 0.,
                 "zero_excess_sides": 0, "native_sides": 0}
    manifest = {}
    for index, f in enumerate(frames, 1):
        path = native/traces[f["stem"]]["file"]
        if file_hash(path) != traces[f["stem"]]["sha256"]:
            raise ValueError("Historical native distribution changed.")
        with np.load(path, allow_pickle=False) as cache:
            cached = {k: cache[k].copy() for k in cache.files}
        available = cached["available"]
        probability = cached["probability"].astype(float)
        # Unavailable trace rows are zero; use a harmless point mass and mask them.
        probability[..., 0] += (~available)[..., None]
        result = distribution_reliability(probability)
        boxes = np.stack([original_coordinates(result["mean"][v], cached["anchors"][v],
                          cached["strides"][v], window) for v, window in enumerate(windows[f["stem"]])])
        available = available & (np.stack([box_areas(b) for b in boxes]) > 0)
        scores = {s: result[s] for s in SCORES[:3]}
        regret = apply_actions(f["candidates"], f["boxes"], f["regret"])
        scores.update(geometry_scores(boxes, available, regret))
        gt = np.array([o["box_xyxy"] for o in f["metadata"]["valid_objects"]], float).reshape(-1, 4)
        owned = f["owner"] >= 0
        targets = np.zeros((len(owned), 4))
        targets[owned] = gt[f["owner"][owned]]
        quality = paired_iou(boxes, targets[None])
        eligible = owned & (f["candidates"][:, 4] >= .05) & (available.sum(0) >= 2)
        left, right = PAIRS.T
        difference = (quality[left]-quality[right]).T
        pair_valid = ((available[left] & available[right]).T & eligible[:, None] &
                      (np.abs(difference) > .01))
        weights = np.broadcast_to(1/np.maximum(pair_valid.sum(1), 1)[:, None], pair_valid.shape)
        active = available & eligible[None]
        mechanism["available_owned_views"] += int(active.sum())
        mechanism["variance_sum"] += float(result["variance"][active].sum())
        mechanism["interpolation_floor_sum"] += float(result["floor"][active].sum())
        mechanism["zero_excess_sides"] += int((result["excess"][active] <= 1e-12).sum())
        mechanism["native_sides"] += int(active.sum()*4)
        for g, values in accumulated.items():
            include = pair_valid & group_mask(f["areas"], g)[:, None]
            values["informative_rows"] += int(include.any(1).sum())
            label = (difference > 0)[include]
            values["label"].extend((label, ~label))
            values["weight"].extend((weights[include]*.5, weights[include]*.5))
            for s in SCORES:
                delta = (scores[s][left]-scores[s][right]).T[include]
                values[s].extend((delta, -delta))
        path = output/"frames"/(f["stem"]+".npz")
        np.savez_compressed(path, boxes=boxes, available=available, quality=quality,
                            eligible=eligible, pair_valid=pair_valid, **scores)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
        if index == 1 or index % 16 == 0:
            print(f"DFL quantization-floor screen: {index}/64, {perf_counter()-tick:.1f}s", flush=True)
    metrics = {}
    for g, values in accumulated.items():
        joined = {k: np.concatenate(v) for k, v in values.items() if k != "informative_rows"}
        metrics[g] = {"informative_rows": values["informative_rows"],
                      "unordered_informative_pairs": len(joined["label"])//2,
                      **{s+"_auc": weighted_auc(joined["label"], joined[s], joined["weight"]) for s in SCORES}}
    mechanism["floor_fraction_of_total_variance"] = (
        mechanism["interpolation_floor_sum"]/max(mechanism["variance_sum"], 1e-12))
    verdict = decide(metrics, registration["policy"])
    save_json(output/"frames.json", manifest)
    save_json(output/"metrics.json", metrics)
    save_json(output/"mechanism.json", mechanism)
    save_json(output/"decision.json", verdict)
    save_json(output/"verification.json", {
        "head_holdout_train_images": len(frames), "detector_forwards": 0, "image_decodes": 0, "fitting_steps": 0,
        "no_GT_in_reliability_scores": True, "native_means_reconstructed_float64_not_claimed_FP16_replay": True,
        "symmetric_pair_labels_and_unit_weight_per_informative_original": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-tick})
    return {"output": str(output), "decision": verdict, "metrics": metrics, "mechanism": mechanism}
