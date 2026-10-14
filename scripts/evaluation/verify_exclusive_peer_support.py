"""Audit candidate47 with independent scalar matching and LP optimality checks."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.localization import apply_actions
from sdcl.training.side_composition_pilot import load_frames


OUTPUT = "output/analysis/exclusive-peer-support-fixed-20261014"
NAMES = ("exclusive_retention", "greedy_retention", "independent_support", "confidence", "inverse_crowding")
GROUPS = ("all", "tiny", "small_all")


def overlaps(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    result = np.zeros((len(first), len(second)))
    for i, a in enumerate(first):
        if not len(second):
            continue
        b = np.asarray(second)
        intersection = (np.minimum(a[2], b[:, 2])-np.maximum(a[0], b[:, 0])).clip(0)
        intersection *= (np.minimum(a[3], b[:, 3])-np.maximum(a[1], b[:, 1])).clip(0)
        aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
        ba = (b[:, 2]-b[:, 0]).clip(0)*(b[:, 3]-b[:, 1]).clip(0)
        result[i] = intersection/np.maximum(aa+ba-intersection, 1e-12)
    return result


def lp_value(weights):
    row, col = np.nonzero(weights)
    if not len(row):
        return 0.
    n, m = weights.shape
    incidence = coo_matrix((np.ones(2*len(row)),
                            (np.r_[row, n+col], np.tile(np.arange(len(row)), 2))),
                           shape=(n+m, len(row))).tocsr()
    solution = linprog(-weights[row, col], A_ub=incidence, b_ub=np.ones(n+m),
                      bounds=(0, 1), method="highs")
    assert solution.success, solution.message
    return -solution.fun


def verify():
    torch.set_num_threads(1)
    output = project_path(OUTPUT)
    read = lambda name: json.loads((output/name).read_text())
    result, plan, manifest = read("result.json"), read("plan.json"), read("frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest, name
    with zipfile.ZipFile(output/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            assert file_hash(project_path(name)) == digest, name
    frames, _, heads = load_frames()
    frames = frames["holdout"]
    assert heads == plan["heads"]
    assert [f["stem"] for f in frames] == plan["image_selection"]
    raw_path = project_path("output/analysis/class-choice-pilot-fixed-20261014/raw_views.jsonl")
    assert file_hash(raw_path) == plan["raw_views_sha256"]
    raw = {r["image"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    templates = np.asarray(plan["source_templates"])
    per_image = []
    lp_checks = 0
    for frame in frames:
        artifact = output/manifest[frame["stem"]]["file"]
        assert file_hash(artifact) == manifest[frame["stem"]]["sha256"]
        with np.load(artifact, allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        prediction = apply_actions(frame["candidates"], frame["boxes"], frame["regret"])
        np.testing.assert_array_equal(prediction, saved["prediction"])
        origin = np.array([
            min(range(5), key=lambda v: sum(float(x-y)**2 for x, y in zip(
                row[0, plan["source_columns"]], templates[v])))
            for row in frame["features"]])
        np.testing.assert_array_equal(origin, saved["origin"])
        rows = np.array([i for i, p in enumerate(prediction) if p[4] >= .25], int)
        np.testing.assert_array_equal(rows, saved["rows"])
        query = prediction[rows].astype(float)
        best_ids = np.full((len(query), 5), -1, int)
        best_weights, allocated_weights, greedy_weights = (np.zeros((len(query), 5)) for _ in range(3))
        for v, raw_view in enumerate(raw[frame["stem"]]["views"]):
            observations = np.asarray(raw_view, np.float32).reshape(-1, 6)
            ids = [i for i, p in enumerate(observations) if p[4] >= .05]
            values = observations[ids]
            iou = overlaps(query[:, :4], values[:, :4])
            weights = np.zeros_like(iou)
            for r in range(len(query)):
                for c in range(len(values)):
                    if origin[rows[r]] != v and query[r, 5] == values[c, 5] and iou[r, c] >= .5:
                        weights[r, c] = iou[r, c]*values[c, 4]
                if len(values):
                    chosen = int(np.argmax(weights[r]))
                    if weights[r, chosen] > 0:
                        best_ids[r, v] = ids[chosen]
                        best_weights[r, v] = weights[r, chosen]
            assigned = saved["assigned_ids"][:, v]
            used_columns = set()
            for r, identity in enumerate(assigned):
                if identity >= 0:
                    assert int(identity) in ids and int(identity) not in used_columns
                    c = ids.index(int(identity))
                    assert weights[r, c] > 0
                    used_columns.add(int(identity))
                    allocated_weights[r, v] = weights[r, c]
            optimum = lp_value(weights)
            np.testing.assert_allclose(allocated_weights[:, v].sum(), optimum, atol=1e-8, rtol=1e-10)
            lp_checks += 1
            taken_rows, taken_cols = set(), set()
            expected_greedy = np.full(len(query), -1, int)
            edges = sorted([(float(weights[r, c]), r, c) for r, c in zip(*np.nonzero(weights))],
                           key=lambda e: (-e[0], e[1], e[2]))
            for value, r, c in edges:
                if r not in taken_rows and c not in taken_cols:
                    taken_rows.add(r)
                    taken_cols.add(c)
                    expected_greedy[r] = ids[c]
                    greedy_weights[r, v] = value
            np.testing.assert_array_equal(saved["greedy_ids"][:, v], expected_greedy)
        for name, value in (("best_ids", best_ids), ("best_weight", best_weights),
                            ("assigned_weight", allocated_weights), ("greedy_weight", greedy_weights)):
            np.testing.assert_array_equal(saved[name], value)
        all_iou = overlaps(query[:, :4], query[:, :4])
        np.testing.assert_array_equal(saved["query_iou"], all_iou)
        denominator = best_weights.sum(1)
        crowding = np.array([
            -sum(all_iou[i, j] for j in range(len(query))
                 if i != j and query[i, 5] == query[j, 5] and all_iou[i, j] >= .5)
            for i in range(len(query))])
        scores = {
            "exclusive_retention": allocated_weights.sum(1)/np.maximum(denominator, 1e-12),
            "greedy_retention": greedy_weights.sum(1)/np.maximum(denominator, 1e-12),
            "independent_support": denominator/np.maximum((best_weights > 0).sum(1), 1),
            "confidence": query[:, 4], "inverse_crowding": crowding,
        }
        for name, value in scores.items():
            np.testing.assert_allclose(saved[name], value, atol=1e-14, rtol=0)
        objects = frame["metadata"]["valid_objects"]
        gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
        gt_class = [o["class_id"] for o in objects]
        label_iou = overlaps(gt, prediction[:, :4])
        taken_gt, taken_dt = set(), set()
        for dt in sorted(range(len(prediction)), key=lambda i: (-prediction[i, 4], i)):
            if prediction[dt, 4] < .25:
                continue
            eligible = [g for g in range(len(gt)) if g not in taken_gt and
                        gt_class[g] == prediction[dt, 5] and label_iou[g, dt] >= .5]
            if eligible:
                chosen = min(eligible, key=lambda g: (-label_iou[g, dt], g))
                taken_gt.add(chosen)
                taken_dt.add(dt)
        labels = np.array([int(i) in taken_dt for i in rows])
        duplicate = np.array([int(i) not in taken_dt and any(
            gt_class[g] == prediction[i, 5] and label_iou[g, i] >= .5 for g in range(len(gt))) for i in rows])
        np.testing.assert_array_equal(saved["labels"], labels)
        np.testing.assert_array_equal(saved["duplicate_like"], duplicate)
        area = ((query[:, 2]-query[:, 0])*(query[:, 3]-query[:, 1]) *
                (640/max(frame["metadata"]["original_size"]))**2)
        trace = {}
        for group in GROUPS:
            eligible = denominator > 0
            if group != "all":
                eligible &= area < (16 if group == "tiny" else 32)**2
            positive, negative, weight = [], [], []
            shared = np.zeros((len(query), len(query)), bool)
            for i in range(len(query)):
                for j in range(len(query)):
                    shared[i, j] = i != j and query[i, 5] == query[j, 5] and any(
                        best_ids[i, v] >= 0 and best_ids[i, v] == best_ids[j, v] for v in range(5))
            for i in np.flatnonzero(labels & eligible):
                others = [j for j in range(len(query)) if eligible[j] and not labels[j] and shared[i, j]]
                others = sorted(others, key=lambda j: (-all_iou[i, j], j))[:4]
                for j in others:
                    positive.append(i)
                    negative.append(j)
                    weight.append(1/len(others))
            for suffix, expected in (("positive", positive), ("negative", negative), ("weight", weight)):
                np.testing.assert_array_equal(saved[group+"_"+suffix], expected)
            trace[group] = {
                "weight": sum(weight), "pairs": len(positive), "paired_tp_queries": len(set(positive)),
                "eligible_tp": int((labels & eligible).sum()),
                "eligible_fp": int((~labels & eligible).sum()),
                "eligible_duplicate_like_fp": int((duplicate & eligible).sum()),
                "distinct_tp_pairs_sharing_support": sum(
                    shared[i, j] for i in range(len(query)) for j in range(i+1, len(query))
                    if labels[i] and labels[j] and eligible[i] and eligible[j]),
                "numerators": {a: sum(
                    w*(float(scores[a][i] > scores[a][j])+.5*float(scores[a][i] == scores[a][j]))
                    for i, j, w in zip(positive, negative, weight)) for a in NAMES},
            }
        per_image.append(trace)
    expected_traces = read("per_image.json")
    for actual, expected in zip(per_image, expected_traces):
        for group in GROUPS:
            for key, value in actual[group].items():
                if key == "numerators":
                    for name, number in value.items():
                        assert number == expected[group][key][name]
                else:
                    assert value == expected[group][key]
    metrics = read("metrics.json")
    draws = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    for group in GROUPS:
        traces = [t[group] for t in per_image]
        total = sum(t["weight"] for t in traces)
        for name in NAMES:
            actual = sum(t["numerators"][name] for t in traces)/total if total else None
            assert actual == metrics[group]["concordance"][name]
        counts = np.array([t["weight"] for t in traces])
        denominator = draws@counts
        numerator = draws@np.array([t["numerators"]["exclusive_retention"] for t in traces])
        interval = np.quantile(numerator[denominator > 0]/denominator[denominator > 0], [.025, .975]).tolist()
        assert interval == metrics[group]["exclusive_image_bootstrap95ci"]
    verification = {
        "all64_original_regret_predictions_and_native_source_identities_replayed": True,
        "all_raw_peer_edge_weights_and_unconstrained_choices_independently_recomputed": True,
        "optimal_allocations_feasible_and_objectives_equal_independent_linear_programs": lp_checks,
        "all_greedy_assignments_independently_recomputed": True,
        "all_TP_FP_duplicate_labels_and_scores_recomputed": True,
        "all192_group_image_pair_selections_numerators_and_bootstrap_intervals_recomputed": True,
        "all_source_artifact_archive_and_raw_view_hashes_verified": True,
        "verifier_sha256": file_hash(Path(__file__)),
        "limits": "Reused detector-training scenes and inherited query ignore filtering; no AP/latency/novelty proof.",
    }
    (output/"independent_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
