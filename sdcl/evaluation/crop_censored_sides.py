"""Screen spatially clipped coordinate evidence before any fusion evaluation."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, coco_size_metrics, file_hash, group_mask,
    match_at_threshold, recall_at_fp_budgets,
)
from sdcl.postprocess.correlated_error_fusion import source_order
from sdcl.postprocess.localization import action_feature_names, apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate49_crop_censored_sides_registration_20261014.json"
OUTPUT = "output/analysis/crop-censored-sides-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("regret", "point", "drop_whole", "drop_side", "one_sided")
GROUPS = ("all", "tiny", "small_all")


def censor_flags(boxes, available, windows, size):
    boxes = np.asarray(boxes, float)
    windows = np.asarray(windows, float)
    if boxes.shape[1:] != (5, 4) or available.shape != boxes.shape[:2] or windows.shape != (5, 4):
        raise ValueError("Expected five native views and original windows.")
    width, height = size
    internal = np.array([windows[:, 0] > 0, windows[:, 1] > 0,
                         windows[:, 2] < width, windows[:, 3] < height]).T
    return (boxes == windows[None]) & internal[None] & available[:, :, None]


def one_side(values, censored, reference, lower_side):
    values = np.asarray(values, float)
    censored = np.asarray(censored, bool)
    point, limits = values[~censored], values[censored]
    if not len(limits):
        return float(point.mean()) if len(point) else float(reference)
    if not len(point):
        return float(min(reference, limits.min()) if lower_side else max(reference, limits.max()))
    limits = np.sort(limits)
    if not lower_side:
        limits = limits[::-1]
    candidates = [float(point.mean()), float(reference), *limits.tolist()]
    for count in range(1, len(limits)+1):
        candidates.append(float((point.sum()+limits[:count].sum())/(len(point)+count)))
    def key(z):
        error = np.maximum(0, z-limits) if lower_side else np.maximum(0, limits-z)
        return float(((z-point)**2).sum()+(error**2).sum()), abs(z-reference), z
    return float(min(candidates, key=key))


def fuse(reference, native, available, flags, active):
    outputs = {a: reference.copy() for a in ARMS}
    fallback = dict.fromkeys(ARMS[1:], 0)
    for row in np.flatnonzero(active):
        values = native[row, available[row]]
        censored = flags[row, available[row]]
        coordinates = {"point": values.mean(0)}
        whole = ~censored.any(1)
        coordinates["drop_whole"] = values[whole].mean(0) if whole.any() else reference[row, :4].copy()
        coordinates["drop_side"], coordinates["one_sided"] = reference[row, :4].copy(), reference[row, :4].copy()
        for side in range(4):
            good = ~censored[:, side]
            if good.any():
                coordinates["drop_side"][side] = values[good, side].mean()
            coordinates["one_sided"][side] = one_side(values[:, side], censored[:, side], reference[row, side], side < 2)
        for arm, box in coordinates.items():
            if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                fallback[arm] += 1
                continue
            outputs[arm][row, :4] = box
    return outputs, fallback


def mechanism_decision(metrics, policy):
    tiny = metrics["tiny"]
    checks = {
        "owned_rows": tiny["affected_owned_rows"] >= policy["tiny_affected_owned_rows_min"],
        "images": tiny["affected_owned_images"] >= policy["tiny_affected_owned_images_min"],
        "sides": tiny["censored_side_observations"] >= policy["tiny_censored_side_observations_min"],
        "valid_inequality": tiny["inequality_valid_fraction"] is not None and tiny["inequality_valid_fraction"] >= policy["tiny_inequality_valid_fraction_min"],
        "strict_extension": tiny["strict_extension_fraction"] is not None and tiny["strict_extension_fraction"] >= policy["tiny_strict_extension_fraction_min"],
    }
    return {"decision": "FIXED_CROP_FUSION_EVALUATION_ALLOWED" if all(checks.values()) else "STOP_THIS_EXACT_CROP_CENSORING_RECIPE",
            "checks": checks, "scope": "GT-assisted crop-boundary mechanism/coverage only, not AP, deployability or innovation."}


def efficacy_decision(metrics, budgets, policy):
    gains = {g: 100*(metrics["one_sided"][g]["ap50_95"]-metrics["regret"][g]["ap50_95"]) for g in GROUPS}
    comparisons = {a: 100*(metrics["one_sided"]["small_all"]["ap50_95"]-metrics[a]["small_all"]["ap50_95"])
                   for a in ("point", "drop_whole", "drop_side")}
    fp = budgets["one_sided"][0]["delta_recall_pp"]["small_all"]
    checks = {g+"_AP": v >= policy[g+"_AP_vs_regret_min_pp"] for g, v in gains.items()}
    checks.update({a: v >= policy["small_all_AP_vs_"+a+"_min_pp"] for a, v in comparisons.items()})
    checks["fixedFPPI10"] = fp >= policy["small_all_fixedFPPI10_vs_regret_min_pp"]
    return {"decision": "BOUNDED_CROP_CENSORING_CONFIRMATION_ALLOWED" if all(checks.values()) else "STOP_THIS_ONE_SIDED_EQUAL_WEIGHT_CROP_FUSION_RECIPE",
            "checks": checks, "AP_gains_vs_regret_pp": gains, "small_AP_gains_vs_controls_pp": comparisons,
            "small_fixedFPPI10_recall_gain_pp": fp,
            "scope": "Reused train-scene diagnostic efficacy only; no official AP or novelty proof."}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    if output.exists():
        raise FileExistsError("Refuse an existing crop-censoring experiment output.")
    registration = json.loads(project_path(REGISTRATION).read_text())
    frames, prior, heads = load_frames()
    frames = frames["holdout"]
    if len(frames) != 64:
        raise ValueError("Expected exactly original64 headholdout frames.")
    _, raw_plan = verify_record(project_path(RAW), "docs/experiments/candidate12_class_choice_result_20261014.json")
    raw_path = project_path(RAW)/"raw_views.jsonl"
    raw = {r["image"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    _, mean, std, _ = frozen_heads()
    columns = [action_feature_names().index("view_"+str(v)) for v in range(5)]
    paths = (REGISTRATION, "sdcl/evaluation/crop_censored_sides.py",
             "scripts/evaluation/diagnose_crop_censored_sides.py", "tests/test_crop_censored_sides.py",
             "sdcl/postprocess/correlated_error_fusion.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/evaluation/localization_official_audit.py", "sdcl/evaluation/size_analysis.py",
             "sdcl/postprocess/localization.py", "sdcl/training/marginal_class_intervention.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    archive_source(output, hashes)
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "source_archive_sha256": file_hash(output/"source.zip"),
        "heads": heads, "images": [f["stem"] for f in frames], "raw_views_sha256": file_hash(raw_path),
        "source_columns": columns, "source_mean": mean[columns].tolist(), "source_std": std[columns].tolist(),
    })
    totals = {g: dict.fromkeys(("owned_rows", "affected_owned_rows", "affected_owned_images",
                               "censored_side_observations", "inequality_valid", "strict_extension"), 0) for g in GROUPS}
    manifest, per_image = {}, []
    all_active, high_active = 0, 0
    for f in frames:
        native, available = source_order(f["features"], mean, std, columns, f["boxes"], f["valid"])
        size = f["metadata"]["original_size"]
        windows = raw[f["stem"]]["windows"]
        flags = censor_flags(native, available, windows, size)
        active = (f["candidates"][:, 4] >= .05) & (available.sum(1) >= 2) & flags.any((1, 2))
        reference = apply_actions(f["candidates"], f["boxes"], f["regret"])
        objects = f["metadata"]["valid_objects"]
        gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
        classes = np.asarray([o["class_id"] for o in objects], int)
        matches, _ = match_at_threshold(box_iou_matrix(gt, reference[:, :4]), classes, reference, .05)
        owner = np.full(len(reference), -1, int)
        owner[matches[matches >= 0]] = np.flatnonzero(matches >= 0)
        targets = np.zeros((len(reference), 4))
        targets[owner >= 0] = gt[owner[owner >= 0]]
        areas = np.zeros(len(reference))
        areas[owner >= 0] = box_areas(gt[owner[owner >= 0]])*(640/max(size))**2
        sign = np.array([-1, -1, 1, 1])
        signed_extension = (targets[:, None]-native)*sign
        trace = {}
        for group in GROUPS:
            owned = (owner >= 0) & group_mask(areas, group)
            selected = owned & active
            observations = flags & selected[:, None, None]
            values = {
                "owned_rows": int(owned.sum()), "affected_owned_rows": int(selected.sum()),
                "affected_owned_images": int(selected.any()), "censored_side_observations": int(observations.sum()),
                "inequality_valid": int(((signed_extension >= 0) & observations).sum()),
                "strict_extension": int(((signed_extension > 0) & observations).sum()),
            }
            trace[group] = values
            for key, value in values.items():
                totals[group][key] += value
        all_active += int(active.sum())
        high_active += int((active & (reference[:, 4] >= .25)).sum())
        path = output/"frames"/(f["stem"]+".npz")
        np.savez_compressed(path, reference=reference, native=native, available=available,
                            flags=flags, active=active, owner=owner, targets=targets, areas=areas)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
        per_image.append(trace)
        f.update(reference=reference, native=native, available=available, flags=flags, active=active)
    for group, values in totals.items():
        denominator = values["censored_side_observations"]
        values["inequality_valid_fraction"] = values["inequality_valid"]/denominator if denominator else None
        values["strict_extension_fraction"] = values["strict_extension"]/denominator if denominator else None
    mechanism = mechanism_decision(totals, registration["mechanism_policy"])
    save_json(output/"frames.json", manifest)
    save_json(output/"per_image.json", per_image)
    save_json(output/"mechanism.json", totals)
    save_json(output/"mechanism_decision.json", mechanism)
    decision = mechanism
    evaluation = None
    if mechanism["decision"] == "FIXED_CROP_FUSION_EVALUATION_ALLOWED":
        records, predictions, prediction_manifest = [], {a: {} for a in ARMS}, {}
        (output/"predictions").mkdir()
        for index, f in enumerate(frames, 1):
            arms, fallback = fuse(f["reference"], f["native"], f["available"], f["flags"], f["active"])
            for a, values in arms.items():
                if not np.array_equal(values[:, 4:], f["reference"][:, 4:]) or not np.array_equal(values[~f["active"]], f["reference"][~f["active"]]):
                    raise ValueError("Fusion changed a frozen invariant.")
                predictions[a][f["stem"]] = values
            path = output/"predictions"/(f["stem"]+".npz")
            np.savez_compressed(path, **arms)
            prediction_manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path), "fallback": fallback}
            records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                            "objects": f["metadata"]["valid_objects"]})
        metrics = {a: coco_size_metrics(records, predictions[a], dict(enumerate(NAMES)), 640) for a in ARMS}
        budgets = {}
        for a in ARMS[1:]:
            budgets[a], _ = recall_at_fp_budgets(records, {"baseline": predictions["regret"], "sdcl": predictions[a]},
                                                reference_size=640, budgets=(10,), bootstrap_samples=0)
        evaluation = {"metrics": metrics, "fixedFP_budgets": budgets}
        decision = efficacy_decision(metrics, budgets, registration["efficacy_policy"])
        save_json(output/"prediction_manifest.json", prediction_manifest)
        save_json(output/"evaluation.json", evaluation)
    verification = {
        "headholdout_images": 64, "detector_forwards": 0, "image_decodes": 0, "fitting_steps": 0,
        "active_rows": all_active, "active_rows_at_original_conf025": high_active,
        "GT_not_used_in_flags_active_mask_or_fusion": True, "GT_matching_before_eligibility": True,
        "source_view_identity_exact_replay": True, "AP_evaluated": evaluation is not None,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started,
    }
    save_json(output/"decision.json", decision)
    save_json(output/"verification.json", verification)
    result = {
        "experiment": registration["experiment"], "date": "2026-10-14", "output": str(output),
        "decision": decision, "mechanism": totals, "evaluation": evaluation, "verification": verification,
        "artifacts_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    save_json(output/"result.json", result)
    return result
