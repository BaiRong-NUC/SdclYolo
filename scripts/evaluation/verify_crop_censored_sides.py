"""Audit candidate49 without calling its flag, fusion or matching routines."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash
from sdcl.postprocess.localization import apply_actions
from sdcl.training.side_composition_pilot import load_frames


OUTPUT = "output/analysis/crop-censored-sides-fixed-20261014"
ARMS = ("regret", "point", "drop_whole", "drop_side", "one_sided")
BOUNDS = {"all": (0, np.inf), "tiny": (0, 256), "small": (256, 1024),
          "small_all": (0, 1024), "medium": (1024, 9216), "large": (9216, np.inf)}


def owners(predictions, objects, confidence):
    gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
    overlaps = np.zeros((len(predictions), len(gt)))
    for i, p in enumerate(predictions):
        intersection = np.maximum(np.minimum(p[2:4], gt[:, 2:])-np.maximum(p[:2], gt[:, :2]), 0).prod(1)
        union = np.maximum(p[2:4]-p[:2], 0).prod()+np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)-intersection
        overlaps[i] = intersection/np.maximum(union, 1e-12)
    result, used = np.full(len(predictions), -1, int), set()
    for i in sorted(range(len(predictions)), key=lambda i: (-float(predictions[i, 4]), i)):
        if predictions[i, 4] < confidence:
            continue
        eligible = [g for g, o in enumerate(objects)
                    if g not in used and o["class_id"] == int(predictions[i, 5]) and overlaps[i, g] >= .5]
        if eligible:
            g = min(eligible, key=lambda g: (-overlaps[i, g], g))
            result[i] = g
            used.add(g)
    return result


def root_coordinate(values, censored, reference, left):
    point, limits = values[~censored], values[censored]
    if not len(point):
        return min(reference, min(limits)) if left else max(reference, max(limits))
    low, high = float(min(values)), float(max(values))
    # Independent monotone derivative root, rather than enumerating active sets.
    for _ in range(90):
        middle = (low+high)/2
        derivative = (middle-point).sum()
        derivative += np.maximum(middle-limits, 0).sum() if left else np.minimum(middle-limits, 0).sum()
        if derivative > 0:
            high = middle
        else:
            low = middle
    return (low+high)/2


def fixed_fp(frames, predictions):
    entries, totals = [], dict.fromkeys(BOUNDS, 0)
    for f in frames:
        objects = f["metadata"]["valid_objects"]
        area = np.array([np.prod(np.subtract(o["box_xyxy"][2:], o["box_xyxy"][:2]))
                         for o in objects])*(640/max(f["metadata"]["original_size"]))**2
        for g, (lo, hi) in BOUNDS.items():
            totals[g] += int(((area >= lo) & (area < hi)).sum())
        p = predictions[f["stem"]]
        owner = owners(p, objects, .001)
        entries.extend((float(row[4]), int(owner[i]), float(area[owner[i]]) if owner[i] >= 0 else -1.)
                       for i, row in enumerate(p))
    entries.sort(key=lambda e: -e[0])
    false, true = 0, dict.fromkeys(BOUNDS, 0)
    selected = {"confidence": float(np.nextafter(entries[0][0], np.inf)), "actual_fp": 0,
                "actual_fppi": 0., "recall": dict.fromkeys(BOUNDS, 0.)}
    start = 0
    while start < len(entries):
        end = start+1
        while end < len(entries) and entries[end][0] == entries[start][0]:
            end += 1
        for _, owner, area in entries[start:end]:
            false += owner < 0
            for g, (lo, hi) in BOUNDS.items():
                true[g] += owner >= 0 and lo <= area < hi
        if false > 10*len(frames):
            break
        selected = {"confidence": entries[start][0], "actual_fp": false,
                    "actual_fppi": false/len(frames),
                    "recall": {g: true[g]/n if n else None for g, n in totals.items()}}
        start = end
    return selected


def assert_tree(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            assert_tree(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_tree(a, e)
    elif expected is None or isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def verify():
    torch.set_num_threads(1)
    output = project_path(OUTPUT)
    read = lambda name: json.loads((output/name).read_text())
    result, plan = read("result.json"), read("plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest, name
    with zipfile.ZipFile(output/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            assert file_hash(project_path(name)) == digest, name
    frames, _, heads = load_frames()
    frames = frames["holdout"]
    assert len(frames) == 64 and [f["stem"] for f in frames] == plan["images"]
    assert heads == plan["heads"]
    raw_path = project_path("output/analysis/class-choice-pilot-fixed-20261014/raw_views.jsonl")
    assert file_hash(raw_path) == plan["raw_views_sha256"]
    raw = {r["image"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    templates = np.clip((np.eye(5)-np.array(plan["source_mean"]))/np.array(plan["source_std"]), -6, 6)
    manifest, predictions_manifest = read("frames.json"), read("prediction_manifest.json")
    totals = {g: dict.fromkeys(("owned_rows", "affected_owned_rows", "affected_owned_images",
                               "censored_side_observations", "inequality_valid", "strict_extension"), 0)
              for g in ("all", "tiny", "small_all")}
    per_image, records, predictions = [], [], {a: {} for a in ARMS}
    roots, active_count, high_count = 0, 0, 0
    for index, f in enumerate(frames, 1):
        stem = f["stem"]
        path = output/manifest[stem]["file"]
        assert file_hash(path) == manifest[stem]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        reference = apply_actions(f["candidates"], f["boxes"], f["regret"])
        native, available = np.zeros((len(reference), 5, 4)), np.zeros((len(reference), 5), bool)
        for row, slot in zip(*np.nonzero(f["valid"][:, :5])):
            observed = f["features"][row, slot, plan["source_columns"]]
            distance = ((templates-observed)**2).sum(1)
            view = int(np.argmin(distance))
            assert distance[view] <= 1e-8 and not available[row, view]
            native[row, view], available[row, view] = f["boxes"][row, slot], True
        flags = np.zeros_like(native, bool)
        width, height = f["metadata"]["original_size"]
        for view, window in enumerate(raw[stem]["windows"]):
            for side, bound in enumerate(window):
                inside = bound > 0 if side < 2 else bound < (width if side == 2 else height)
                flags[:, view, side] = available[:, view] & (native[:, view, side] == bound) & inside
        active = (f["candidates"][:, 4] >= .05) & (available.sum(1) >= 2) & flags.any((1, 2))
        active_count += int(active.sum())
        high_count += int((active & (reference[:, 4] >= .25)).sum())
        objects = f["metadata"]["valid_objects"]
        owner = owners(reference, objects, .05)
        targets, area = np.zeros((len(reference), 4)), np.zeros(len(reference))
        for row in np.flatnonzero(owner >= 0):
            targets[row] = objects[owner[row]]["box_xyxy"]
            area[row] = np.prod(targets[row, 2:]-targets[row, :2])*(640/max(width, height))**2
        rebuilt = dict(reference=reference, native=native, available=available, flags=flags,
                       active=active, owner=owner, targets=targets, areas=area)
        for k, value in rebuilt.items():
            np.testing.assert_array_equal(value, saved[k], err_msg=f"{stem}:{k}")
        trace = {}
        for g in totals:
            lo, hi = BOUNDS[g]
            owned = (owner >= 0) & (area >= lo) & (area < hi)
            selected = owned & active
            observations = flags & selected[:, None, None]
            extension = (targets[:, None]-native)*[-1, -1, 1, 1]
            values = dict(owned_rows=int(owned.sum()), affected_owned_rows=int(selected.sum()),
                          affected_owned_images=int(selected.any()), censored_side_observations=int(observations.sum()),
                          inequality_valid=int(((extension >= 0) & observations).sum()),
                          strict_extension=int(((extension > 0) & observations).sum()))
            trace[g] = values
            for k, value in values.items():
                totals[g][k] += value
        per_image.append(trace)
        path = output/predictions_manifest[stem]["file"]
        assert file_hash(path) == predictions_manifest[stem]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            predicted = {a: z[a].copy() for a in ARMS}
        expected = {a: reference.copy() for a in ARMS}
        fallbacks = dict.fromkeys(ARMS[1:], 0)
        for row in np.flatnonzero(active):
            v, c = native[row, available[row]], flags[row, available[row]]
            whole = ~c.any(1)
            boxes = {"point": v.mean(0), "drop_whole": v[whole].mean(0) if whole.any() else reference[row, :4].copy(),
                     "drop_side": reference[row, :4].copy(), "one_sided": np.zeros(4)}
            for side in range(4):
                if (~c[:, side]).any():
                    boxes["drop_side"][side] = v[~c[:, side], side].mean()
                boxes["one_sided"][side] = root_coordinate(v[:, side], c[:, side], reference[row, side], side < 2)
                roots += 1
            for a, box in boxes.items():
                if np.isfinite(box).all() and np.all(box[2:] > box[:2]):
                    expected[a][row, :4] = box
                else:
                    fallbacks[a] += 1
        assert fallbacks == predictions_manifest[stem]["fallback"]
        for a in ARMS:
            np.testing.assert_allclose(expected[a], predicted[a], atol=1e-9, rtol=1e-12)
            np.testing.assert_array_equal(predicted[a][:, 4:], reference[:, 4:])
            np.testing.assert_array_equal(predicted[a][~active], reference[~active])
            predictions[a][stem] = predicted[a]
        records.append({"id": index, "stem": stem, "size": [width, height], "objects": objects})
    assert per_image == read("per_image.json")
    for values in totals.values():
        n = values["censored_side_observations"]
        values["inequality_valid_fraction"] = values["inequality_valid"]/n if n else None
        values["strict_extension_fraction"] = values["strict_extension"]/n if n else None
    assert_tree(totals, read("mechanism.json"))
    tiny, mechanism_policy = totals["tiny"], plan["registration"]["mechanism_policy"]
    mechanism_checks = {
        "owned_rows": tiny["affected_owned_rows"] >= mechanism_policy["tiny_affected_owned_rows_min"],
        "images": tiny["affected_owned_images"] >= mechanism_policy["tiny_affected_owned_images_min"],
        "sides": tiny["censored_side_observations"] >= mechanism_policy["tiny_censored_side_observations_min"],
        "valid_inequality": tiny["inequality_valid_fraction"] >= mechanism_policy["tiny_inequality_valid_fraction_min"],
        "strict_extension": tiny["strict_extension_fraction"] >= mechanism_policy["tiny_strict_extension_fraction_min"],
    }
    assert mechanism_checks == read("mechanism_decision.json")["checks"] and all(mechanism_checks.values())
    evaluation = read("evaluation.json")
    assert_tree(evaluation, result["evaluation"])
    fixed = {a: fixed_fp(frames, predictions[a]) for a in ARMS}
    for a in ARMS:
        print(f"Rechecking saved AP: {a}", flush=True)
        assert_tree(coco_size_metrics(records, predictions[a], dict(enumerate(NAMES)), 640),
                    evaluation["metrics"][a])
        if a != "regret":
            reported = evaluation["fixedFP_budgets"][a][0]
            assert_tree(fixed["regret"], reported["models"]["baseline"])
            assert_tree(fixed[a], reported["models"]["sdcl"])
            for g in BOUNDS:
                np.testing.assert_allclose(100*(fixed[a]["recall"][g]-fixed["regret"]["recall"][g]),
                                           reported["delta_recall_pp"][g], atol=1e-12)
    policy, decision = plan["registration"]["efficacy_policy"], read("decision.json")
    checks = {}
    for g in ("all", "tiny", "small_all"):
        gain = 100*(evaluation["metrics"]["one_sided"][g]["ap50_95"]-evaluation["metrics"]["regret"][g]["ap50_95"])
        checks[g+"_AP"] = gain >= policy[g+"_AP_vs_regret_min_pp"]
    for a in ("point", "drop_whole", "drop_side"):
        gain = 100*(evaluation["metrics"]["one_sided"]["small_all"]["ap50_95"]-evaluation["metrics"][a]["small_all"]["ap50_95"])
        checks[a] = gain >= policy["small_all_AP_vs_"+a+"_min_pp"]
    checks["fixedFPPI10"] = 100*(fixed["one_sided"]["recall"]["small_all"]-fixed["regret"]["recall"]["small_all"]) >= policy["small_all_fixedFPPI10_vs_regret_min_pp"]
    assert checks == decision["checks"] and not all(checks.values())
    assert decision["decision"] == "STOP_THIS_ONE_SIDED_EQUAL_WEIGHT_CROP_FUSION_RECIPE"
    assert active_count == result["verification"]["active_rows"] == 239
    assert high_count == result["verification"]["active_rows_at_original_conf025"] == 114
    verified = {
        "archived_and_current_sources_and_all128_frame_prediction_files_hashed": True,
        "all64_source_identities_exact_boundary_flags_active_masks_GT_owners_and_mechanism_rebuilt": True,
        "all_five_fusion_arms_fallbacks_and_frozen_scores_classes_inactive_rows_checked": True,
        "independent_derivative_root_coordinate_checks": roots,
        "all_five_AP_results_recomputed_from_saved_predictions_using_same_COCO_evaluator": True,
        "all_five_fixedFPPI10_thresholds_and_recalls_independently_rebuilt_with_whole_ties": True,
        "all_seven_efficacy_gates_recomputed": True,
        "verifier_sha256": file_hash(Path(__file__)),
        "limits": "Reused64 detector-training scenes and inherited ignore-filtered query pool; same AP evaluator, no independent AP implementation or official VisDrone evaluation.",
    }
    (output/"independent_verification.json").write_text(json.dumps(verified, indent=2)+"\n", encoding="utf-8")
    return verified


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
