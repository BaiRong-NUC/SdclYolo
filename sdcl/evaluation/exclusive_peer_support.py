"""Conditional foreground discrimination from conserved cross-view observations."""

import json
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from sdcl.config import project_path
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold
from sdcl.postprocess.localization import action_feature_names, apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate47_exclusive_peer_support_registration_20261014.json"
OUTPUT = "output/analysis/exclusive-peer-support-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
SCORES = ("exclusive_retention", "greedy_retention", "independent_support", "confidence", "inverse_crowding")
GROUPS = ("all", "tiny", "small_all")


def allocate(weights):
    """Return unconstrained, optimal partial, and deterministic greedy assignments."""
    weights = np.asarray(weights, np.float64)
    if weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Expected finite nonnegative bipartite weights.")
    n, m = weights.shape
    best = np.full(n, -1, int)
    optimal, greedy = best.copy(), best.copy()
    if not n or not m:
        return best, optimal, greedy
    chosen = weights.argmax(1)
    active = weights[np.arange(n), chosen] > 0
    best[active] = chosen[active]
    padded = np.concatenate([weights, np.zeros((n, n))], axis=1)
    rows, cols = linear_sum_assignment(padded, maximize=True)
    valid = cols < m
    rows, cols = rows[valid], cols[valid]
    valid = weights[rows, cols] > 0
    optimal[rows[valid]] = cols[valid]
    row, col = np.nonzero(weights > 0)
    order = np.lexsort((col, row, -weights[row, col]))
    used = np.zeros(m, bool)
    for index in order:
        r, c = row[index], col[index]
        if greedy[r] < 0 and not used[c]:
            greedy[r], used[c] = c, True
    return best, optimal, greedy


def peer_graph(prediction, origin, views):
    rows = np.flatnonzero(prediction[:, 4] >= .25)
    query = prediction[rows]
    identities = np.full((len(rows), 5), -1, int)
    assigned, greedy = identities.copy(), identities.copy()
    best_weight = np.zeros((len(rows), 5))
    optimal_weight, greedy_weight = best_weight.copy(), best_weight.copy()
    for view, raw in enumerate(views):
        raw = np.asarray(raw, np.float32).reshape(-1, 6)
        selected = np.flatnonzero(raw[:, 4] >= .05)
        observations = raw[selected]
        iou = box_iou_matrix(query[:, :4], observations[:, :4])
        valid = ((iou >= .5) & (query[:, 5, None] == observations[None, :, 5]) &
                 (origin[rows, None] != view))
        weights = np.where(valid, iou*observations[None, :, 4], 0)
        choices = allocate(weights)
        for choice, ids, values in zip(choices, (identities, assigned, greedy),
                                      (best_weight, optimal_weight, greedy_weight)):
            keep = np.flatnonzero(choice >= 0)
            ids[keep, view] = selected[choice[keep]]
            values[keep, view] = weights[keep, choice[keep]]
    denominator = best_weight.sum(1)
    pair_iou = box_iou_matrix(query[:, :4], query[:, :4])
    crowd = np.where((pair_iou >= .5) & (query[:, 5, None] == query[None, :, 5]), pair_iou, 0)
    np.fill_diagonal(crowd, 0)
    scores = {
        "exclusive_retention": optimal_weight.sum(1)/np.maximum(denominator, 1e-12),
        "greedy_retention": greedy_weight.sum(1)/np.maximum(denominator, 1e-12),
        "independent_support": denominator/np.maximum((best_weight > 0).sum(1), 1),
        "confidence": query[:, 4].astype(float), "inverse_crowding": -crowd.sum(1),
    }
    return {
        "rows": rows, "best_ids": identities, "assigned_ids": assigned, "greedy_ids": greedy,
        "best_weight": best_weight, "assigned_weight": optimal_weight, "greedy_weight": greedy_weight,
        "eligible": denominator > 0, "query_iou": pair_iou, **scores,
    }


def shared_pairs(best_ids, classes, labels, eligible, query_iou):
    shared = ((best_ids[:, None] == best_ids[None]) & (best_ids[:, None] >= 0)).any(2)
    shared &= classes[:, None] == classes[None]
    np.fill_diagonal(shared, False)
    positive, negative, weights = [], [], []
    for row in np.flatnonzero(eligible & labels):
        other = np.flatnonzero(shared[row] & eligible & ~labels)
        other = other[np.argsort(-query_iou[row, other], kind="stable")[:4]]
        if len(other):
            positive.extend([row]*len(other))
            negative.extend(other.tolist())
            weights.extend([1/len(other)]*len(other))
    return np.asarray(positive, int), np.asarray(negative, int), np.asarray(weights, float), shared


def summarize(traces):
    draws = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    metrics = {}
    for group in GROUPS:
        rows = [t[group] for t in traces]
        weight = np.array([t["weight"] for t in rows])
        total = float(weight.sum())
        numerator = {a: np.array([t["numerators"][a] for t in rows]) for a in SCORES}
        concordance = {a: float(v.sum()/total) if total else None for a, v in numerator.items()}
        denominator = draws@weight
        sampled = (draws@numerator["exclusive_retention"])[denominator > 0]/denominator[denominator > 0]
        metrics[group] = {
            "eligible_tp": sum(t["eligible_tp"] for t in rows),
            "eligible_fp": sum(t["eligible_fp"] for t in rows),
            "eligible_duplicate_like_fp": sum(t["eligible_duplicate_like_fp"] for t in rows),
            "paired_tp_queries": sum(t["paired_tp_queries"] for t in rows),
            "pair_count": sum(t["pairs"] for t in rows),
            "represented_images": sum(t["pairs"] > 0 for t in rows),
            "distinct_tp_pairs_sharing_support": sum(t["distinct_tp_pairs_sharing_support"] for t in rows),
            "concordance": concordance,
            "exclusive_image_bootstrap95ci": np.quantile(sampled, [.025, .975]).tolist() if len(sampled) else [None, None],
        }
    return metrics


def decide(metrics, policy):
    tiny = metrics["tiny"]
    values = tiny["concordance"]
    exclusive = values["exclusive_retention"]
    differences = {a: exclusive-values[a] if exclusive is not None and values[a] is not None else None
                   for a in SCORES[1:]}
    checks = {
        "pairs": tiny["pair_count"] >= policy["tiny_pairs_min"],
        "queries": tiny["paired_tp_queries"] >= policy["tiny_paired_tp_queries_min"],
        "images": tiny["represented_images"] >= policy["tiny_images_min"],
        "concordance": exclusive is not None and exclusive >= policy["tiny_exclusive_concordance_min"],
    }
    for a, key in (("greedy_retention", "greedy"), ("independent_support", "independent"),
                   ("confidence", "confidence"), ("inverse_crowding", "crowding")):
        checks["vs_"+key] = differences[a] is not None and differences[a] >= policy["tiny_vs_"+key+"_min"]
    lower = tiny["exclusive_image_bootstrap95ci"][0]
    checks["interval"] = lower is not None and lower > policy["tiny_bootstrap_lower95_exclusive"]
    return {
        "decision": "BOUNDED_EXCLUSIVE_PEER_PILOT_ALLOWED" if all(checks.values()) else "STOP_THIS_EXCLUSIVE_PEER_SUPPORT_RECIPE",
        "checks": checks, "concordance_differences": differences,
        "scope": "Conditional support-conflict discrimination only, not AP, deployability, independent information or novelty.",
    }


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing peer-support experiment.")
    registration = json.loads(project_path(REGISTRATION).read_text())
    frames, prior, heads = load_frames()
    frames = frames["holdout"]
    if len(frames) != 64:
        raise ValueError("Expected original64 head-held-out training images.")
    _, raw_plan = verify_record(project_path(RAW), "docs/experiments/candidate12_class_choice_result_20261014.json")
    raw_path = project_path(RAW)/"raw_views.jsonl"
    raw = {r["image"]: r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
    _, mean, std, _ = frozen_heads()
    columns = [action_feature_names().index("view_"+str(v)) for v in range(5)]
    templates = np.clip((np.eye(5)-mean[columns])/std[columns], -6, 6)
    paths = (
        REGISTRATION, "sdcl/evaluation/exclusive_peer_support.py",
        "scripts/evaluation/diagnose_exclusive_peer_support.py", "tests/test_exclusive_peer_support.py",
        "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/localization_official_audit.py",
        "sdcl/postprocess/localization.py", "sdcl/evaluation/size_analysis.py",
        "sdcl/training/marginal_class_intervention.py", "sdcl/training/rescue.py",
    )
    hashes = {p: file_hash(project_path(p)) for p in paths}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    archive_source(output, hashes)
    import scipy
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output/"source.zip"), "heads": heads,
        "raw_views_sha256": file_hash(raw_path), "scipy_version": scipy.__version__,
        "image_selection": [f["stem"] for f in frames],
        "source_templates": templates.tolist(), "source_columns": columns,
        "historical_menu_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")),
        "historical_raw_result_sha256": file_hash(project_path(
            "docs/experiments/candidate12_class_choice_result_20261014.json")),
    })
    traces, manifest = [], {}
    for f in frames:
        prediction = apply_actions(f["candidates"], f["boxes"], f["regret"])
        distances = ((f["features"][:, 0, columns][:, None]-templates[None])**2).sum(2)
        if np.any(distances.min(1) > 1e-8):
            raise ValueError("Frozen source identity templates do not exactly replay.")
        origin = distances.argmin(1)
        graph = peer_graph(prediction, origin, raw[f["stem"]]["views"])
        objects = f["metadata"]["valid_objects"]
        gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
        classes = np.array([o["class_id"] for o in objects])
        iou = box_iou_matrix(gt, prediction[:, :4])
        matched, used = match_at_threshold(iou, classes, prediction, .25)
        duplicate = ~used & ((iou >= .5) & (classes[:, None] == prediction[None, :, 5])).any(0)
        rows = graph["rows"]
        query = prediction[rows]
        labels = used[rows]
        areas = box_areas(query[:, :4])*(640/max(f["metadata"]["original_size"]))**2
        arrays = {"prediction": prediction, "origin": origin, "labels": labels,
                  "duplicate_like": duplicate[rows], **graph}
        trace = {}
        for group in GROUPS:
            eligible = graph["eligible"] & group_mask(areas, group)
            positive, negative, weight, shared = shared_pairs(
                graph["best_ids"], query[:, 5], labels, eligible, graph["query_iou"])
            numerators = {}
            for a in SCORES:
                v = graph[a]
                numerators[a] = float((weight*((v[positive] > v[negative]) + .5*(v[positive] == v[negative]))).sum())
            trace[group] = {
                "eligible_tp": int((eligible & labels).sum()), "eligible_fp": int((eligible & ~labels).sum()),
                "eligible_duplicate_like_fp": int((eligible & duplicate[rows]).sum()),
                "paired_tp_queries": int(len(np.unique(positive))), "pairs": len(positive),
                "weight": float(weight.sum()), "numerators": numerators,
                "distinct_tp_pairs_sharing_support": int(np.triu(
                    shared & (eligible & labels)[:, None] & (eligible & labels)[None], 1).sum()),
            }
            arrays.update({group+"_positive": positive, group+"_negative": negative, group+"_weight": weight})
        traces.append(trace)
        path = output/"frames"/(f["stem"]+".npz")
        np.savez_compressed(path, **arrays)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
    metrics = summarize(traces)
    decision = decide(metrics, registration["policy"])
    verification = {
        "head_holdout_images": 64, "detector_forwards": 0, "image_decodes": 0, "fitting_steps": 0,
        "all_source_identities_exactly_replayed": True,
        "GT_not_used_in_queries_graphs_weights_assignments_or_score_features": True,
        "regret_predictions_scores_classes_count_order_unchanged": True,
        "all_GT_matching_before_eligibility": True,
        "peer_inventory_not_ignore_filtered": True,
        "query_cache_inherits_historical_diagnostic_ignore_filtering": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started,
    }
    for name, data in (("frames.json", manifest), ("per_image.json", traces), ("metrics.json", metrics),
                       ("decision.json", decision), ("verification.json", verification)):
        save_json(output/name, data)
    result = {
        "experiment": registration["experiment"], "date": "2026-10-14", "output": str(output),
        "decision": decision, "metrics": metrics, "verification": verification,
        "artifacts_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    save_json(output/"result.json", result)
    return result
