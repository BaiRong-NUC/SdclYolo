"""Conditional native semantic/geometric pairing test without learning or inference."""

import hashlib
import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate45_native_semantic_coupling_registration_20261014.json"
OUTPUT = "output/analysis/native-semantic-coupling-fixed-20261014"
PRIOR = "output/analysis/native-spatial-field-fixed-20261014"
SCORES = ("coupling", "shuffled", "marginal_product", "semantic_mean", "geometry_mean", "confidence")
GROUPS = ("all", "tiny", "small_all")
TOLERANCES = np.array([.03, .05, .05])


def features(slots, confidence, stem):
    slots = np.asarray(slots, float)
    if slots.shape != (len(confidence), 8, 8) or not np.isfinite(slots).all():
        raise ValueError("Expected finite cached eight-neighbor/eight-channel slots.")
    mask = slots[:, :, 7] > 0
    count = mask.sum(1)
    semantic, geometry = slots[:, :, 4], slots[:, :, 6]
    pmean = (semantic*mask).sum(1)/np.maximum(count, 1)
    gmean = (geometry*mask).sum(1)/np.maximum(count, 1)
    shuffled = semantic.copy()
    permutations = np.tile(np.arange(8), (len(slots), 1))
    for row in range(len(slots)):
        valid = np.flatnonzero(mask[row])
        seed = int.from_bytes(hashlib.sha256(f"{stem}:{row}:semantic-coupling".encode()).digest()[:8], "little")
        order = np.random.default_rng(seed).permutation(valid)
        permutations[row, valid] = order
        shuffled[row, valid] = semantic[row, order]
    product = pmean*gmean
    # Center explicitly so constant fields do not acquire subtraction noise.
    return {"coupling": ((semantic-pmean[:, None])*(geometry-gmean[:, None])*mask).sum(1)/np.maximum(count, 1),
            "shuffled": ((shuffled-pmean[:, None])*(geometry-gmean[:, None])*mask).sum(1)/np.maximum(count, 1),
            "marginal_product": product, "semantic_mean": pmean, "geometry_mean": gmean,
            "confidence": np.asarray(confidence, float)}, count, permutations


def conditional_pairs(classes, labels, eligible, values):
    controls = np.column_stack([values[k] for k in ("confidence", "semantic_mean", "geometry_mean")])
    positive, negative, weights = [], [], []
    for row in np.flatnonzero(eligible & labels):
        other = np.flatnonzero(eligible & ~labels & (classes == classes[row]))
        delta = np.abs(controls[other]-controls[row])
        keep = (delta <= TOLERANCES).all(1)
        other, delta = other[keep], delta[keep]
        if len(other):
            distance = ((delta/TOLERANCES)**2).sum(1)
            chosen = other[np.argsort(distance, kind="stable")[:4]]
            positive.extend([row]*len(chosen))
            negative.extend(chosen.tolist())
            weights.extend([1/len(chosen)]*len(chosen))
    return np.asarray(positive, int), np.asarray(negative, int), np.asarray(weights, float)


def pair_totals(positive, negative, weight, values):
    return {a: float((weight*((values[a][positive] > values[a][negative]) +
                             .5*(values[a][positive] == values[a][negative]))).sum()) for a in SCORES}


def summarize(per_image):
    draws = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    result = {}
    for group in GROUPS:
        rows = [r[group] for r in per_image]
        denominator = np.array([r["weight"] for r in rows])
        numerator = {a: np.array([r["numerators"][a] for r in rows]) for a in SCORES}
        total = float(denominator.sum())
        concordance = {a: float(v.sum()/total) if total else None for a, v in numerator.items()}
        base = draws@denominator
        sampled = (draws@numerator["coupling"])[base > 0]/base[base > 0]
        interval = np.quantile(sampled, [.025, .975]).tolist() if len(sampled) else [None, None]
        result[group] = {"eligible_tp_queries": sum(r["eligible_tp"] for r in rows),
                         "eligible_fp_rows": sum(r["eligible_fp"] for r in rows),
                         "paired_tp_queries": sum(r["paired_tp"] for r in rows),
                         "pair_count": sum(r["pairs"] for r in rows),
                         "represented_images": sum(r["paired_tp"] > 0 for r in rows),
                         "concordance": concordance, "image_bootstrap_coupling_95ci": interval}
    return result


def decide(metrics, policy):
    tiny = metrics["tiny"]
    score = tiny["concordance"]
    coupling = score["coupling"]
    def difference(baseline):
        return None if coupling is None or score[baseline] is None else coupling-score[baseline]
    lower = tiny["image_bootstrap_coupling_95ci"][0]
    checks = {"pairs": tiny["pair_count"] >= policy["tiny_pair_count_min"],
              "tp_queries": tiny["paired_tp_queries"] >= policy["tiny_paired_tp_queries_min"],
              "images": tiny["represented_images"] >= policy["tiny_represented_images_min"],
              "concordance": coupling is not None and coupling >= policy["tiny_coupling_concordance_min"],
              "vs_marginal_product": difference("marginal_product") is not None and
              difference("marginal_product") >= policy["tiny_coupling_minus_marginal_product_min"],
              "vs_shuffle": difference("shuffled") is not None and difference("shuffled") >= policy["tiny_coupling_minus_shuffle_min"],
              "image_interval": lower is not None and lower > policy["tiny_image_bootstrap_coupling_lower95_min_exclusive"]}
    return {"decision": "MATCHED_NATIVE_COUPLING_LEARNABILITY_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_NATIVE_SEMANTIC_GEOMETRY_COUPLING_RECIPE", "checks": checks,
            "coupling_minus_marginal_product": difference("marginal_product"),
            "coupling_minus_shuffle": difference("shuffled"),
            "scope": "Matched-pair discrimination only, not globalAUROC/AP, conditional information proof or innovation."}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"frames").mkdir()
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/evaluation/native_semantic_coupling.py", "scripts/evaluation/diagnose_native_semantic_coupling.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/postprocess/localization.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    base = project_path(PRIOR)
    verify_record(base, "docs/experiments/candidate42_native_spatial_field_result_20261014.json")
    frames, prior, heads = load_frames()
    frames = frames["holdout"]
    manifest42 = json.loads((base/"features.json").read_text())
    save_json(output/"plan.json", {"registration": registration, "source_hashes": hashes, "heads": heads,
                                 "source_archive_sha256": file_hash(output/"source.zip"),
                                 "images": prior["image_selection"]["holdout"],
                                 "prior_result_sha256": file_hash(project_path(
                                     "docs/experiments/candidate42_native_spatial_field_result_20261014.json"))})
    manifest, per_image = {}, []
    for f in frames:
        entry = manifest42[f["stem"]]
        path = base/entry["file"]
        if entry["split"] != "holdout" or file_hash(path) != entry["sha256"]:
            raise ValueError("Frozen native semantic input changed.")
        with np.load(path, allow_pickle=False) as c:
            slots, active = c["slots"].copy(), c["active"].copy()
        prediction = apply_actions(f["candidates"], f["boxes"], f["regret"])
        values, count, permutations = features(slots, prediction[:, 4], f["stem"])
        objects = f["metadata"]["valid_objects"]
        gt = np.array([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
        gt_classes = np.array([o["class_id"] for o in objects], int)
        matches, labels = match_at_threshold(box_iou_matrix(gt, prediction[:, :4]), gt_classes, prediction, .25)
        areas = box_areas(prediction[:, :4])*(640/max(f["metadata"]["original_size"]))**2
        classes = prediction[:, 5].astype(int)
        eligible = active & (prediction[:, 4] >= .25) & (count >= 4)
        trace = {"labels": labels, "matches": matches, "classes": classes, "areas": areas,
                 "eligible": eligible, "neighbor_count": count, "permutations": permutations,
                 **{a+"_score": v for a, v in values.items()}}
        totals = {}
        for group in GROUPS:
            include = eligible & group_mask(areas, group)
            positive, negative, weight = conditional_pairs(classes, labels, include, values)
            trace.update({group+"_positive": positive, group+"_negative": negative, group+"_weight": weight})
            totals[group] = {"eligible_tp": int((include & labels).sum()), "eligible_fp": int((include & ~labels).sum()),
                             "paired_tp": len(np.unique(positive)), "pairs": len(positive),
                             "weight": float(weight.sum()), "numerators": pair_totals(positive, negative, weight, values)}
        path = output/"frames"/(f["stem"]+".npz")
        np.savez_compressed(path, **trace)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path),
                              "source_feature_sha256": entry["sha256"]}
        per_image.append(totals)
    metrics = summarize(per_image)
    verdict = decide(metrics, registration["policy"])
    for name, value in (("frames", manifest), ("per_image", per_image), ("metrics", metrics), ("decision", verdict)):
        save_json(output/(name+".json"), value)
    save_json(output/"verification.json", {"images": 64, "detector_forwards": 0, "image_decodes": 0, "fitting_steps": 0,
                                          "GT_only_in_TP_FP_labels_not_features_or_pairing_distances": True,
                                          "all_GT_matching_before_native_eligibility_filter": True,
                                          "original_regret_predictions_not_changed": True,
                                          "predicted_area_groups_include_FP_rows": True,
                                          "shuffle_preserves_marginal_semantic_geometric_multisets": True,
                                          "fixed_three_marginal_tolerance_pairs_without_coupling_selection": True,
                                          "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                                          "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "metrics": metrics}
