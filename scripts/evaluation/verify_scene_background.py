"""Independent cached-statistic, convex-fit and decision audit for candidate55."""

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator


ROOT = "output/analysis/scene-background-fixed-20261015"
ARMS = ("row", "whole", "masked", "shuffled")
LIMITS = dict(all=math.inf, tiny=256., small_all=1024.)


def check(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b:
            check(a[key], b[key])
    elif isinstance(b, (str, bool)) or b is None:
        assert a == b
    else:
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-10)


def independent_statistics(logits, boxes, width, height, saved_masks, coordinate_bounds, diagnostics):
    ratio = 960/max(width, height)
    rw, rh = round(width*ratio), round(height*ratio)
    left, top = round((960-rw)/2-.1), round((960-rh)/2-.1)
    whole, masked, coverage, masks, fallback = [], [], [], [], []
    start = 0
    for stride in (8, 16, 32):
        side = 960//stride
        points = np.array([[(x+.5)*stride, (y+.5)*stride] for y in range(side) for x in range(side)])
        valid = ((points[:, 0] >= left) & (points[:, 0] < left+rw) &
                 (points[:, 1] >= top) & (points[:, 1] < top+rh))
        pool = valid.copy()
        sure, possible = np.zeros(len(points), bool), np.zeros(len(points), bool)
        for index, b in enumerate(boxes):
            if b[4] >= .25:
                scaled = (b[:4].astype(np.float32)*ratio).astype(float)
                pool &= ~((points[:, 0] >= scaled[0]+left) & (points[:, 0] <= scaled[2]+left) &
                          (points[:, 1] >= scaled[1]+top) & (points[:, 1] <= scaled[3]+top))
                if coordinate_bounds is not None:
                    lower, upper = coordinate_bounds
                    lo = (lower[index]*ratio).astype(float)+[left, top, left, top]
                    hi = (upper[index]*ratio).astype(float)+[left, top, left, top]
                    sure |= ((points[:, 0] >= hi[0]) & (points[:, 0] <= lo[2]) &
                             (points[:, 1] >= hi[1]) & (points[:, 1] <= lo[3]))
                    possible |= ((points[:, 0] >= lo[0]) & (points[:, 0] <= hi[2]) &
                                 (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[3]))
        stored = saved_masks[start:start+side*side, 1]
        np.testing.assert_array_equal(saved_masks[start:start+side*side, 0], valid)
        if coordinate_bounds is None:
            np.testing.assert_array_equal(stored, pool)
        else:
            minimum, maximum = valid & ~possible, valid & ~sure
            assert not (minimum & ~stored).any() and not (stored & ~maximum).any()
            diagnostics["ambiguous_point_memberships"] += int((minimum ^ maximum).sum())
            diagnostics["inverse_mapping_point_disagreements"] += int((pool != stored).sum())
            pool = stored.copy()
        coverage.append(float(np.count_nonzero(pool)/np.count_nonzero(valid)))
        fallback.append(bool(np.count_nonzero(pool) < 32))
        chosen = valid if fallback[-1] else pool
        masks.append(np.stack((valid, pool, chosen), axis=1))
        values = logits[start:start+side*side].astype(float)
        for destination, mask in ((whole, valid), (masked, chosen)):
            ordered = np.sort(values[mask], axis=0)
            quantiles = []
            for q in (.5, .9, .99):
                position = q*(len(ordered)-1)
                low, high = math.floor(position), math.ceil(position)
                quantiles.append(ordered[low]+(position-low)*(ordered[high]-ordered[low]))
            destination.append(np.column_stack((*quantiles, (ordered >= math.log(.05/.95)).mean(0))))
        start += side*side
    return np.array(whole), np.array(masked), np.array(coverage), np.concatenate(masks), np.array(fallback)


def supervision(reference, metadata, threshold):
    gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
    cls = np.array([o["class_id"] for o in metadata["valid_objects"]])
    area = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)
    ignored = np.zeros(len(reference), bool)
    for b in metadata["ignore_boxes_xyxy"]:
        overlap = np.maximum(np.minimum(reference[:, 2:4], b[2:])-np.maximum(reference[:, :2], b[:2]), 0).prod(1)
        ignored |= overlap/np.maximum(area, 1e-12) > .5
    intersection = np.maximum(np.minimum(gt[:, None, 2:], reference[None, :, 2:4])-
                              np.maximum(gt[:, None, :2], reference[None, :, :2]), 0).prod(2)
    gt_area = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)
    iou = intersection/np.maximum(gt_area[:, None]+area[None]-intersection, 1e-12)
    tp, used = np.zeros(len(reference), bool), set()
    for row in sorted(range(len(reference)), key=lambda k: (-reference[k, 4], k)):
        if ignored[row] or reference[row, 4] < threshold:
            continue
        eligible = [g for g in range(len(gt)) if g not in used and cls[g] == reference[row, 5] and iou[g, row] >= .5]
        if eligible:
            winner = max(eligible, key=lambda g: (iou[g, row], -g))
            used.add(winner)
            tp[row] = True
    duplicate = ~tp & ((iou >= .5) & (cls[:, None] == reference[None, :, 5])).any(0)
    return tp, duplicate, ignored


def verify():
    torch.set_num_threads(1)
    root = project_path(ROOT)
    read = lambda p: json.loads(p.read_text())
    result, plan, manifest = read(root/"result.json"), read(root/"plan.json"), read(root/"frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
        for name, item in plan["installed_sources"].items():
            assert file_hash(Path(item["path"])) == hashlib.sha256(z.read("installed/"+name)).hexdigest() == item["sha256"]
    prior = project_path("output/analysis/class-choice-pilot-fixed-20261014")
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"raw_views.jsonl") == plan["raw_views_sha256"]
    selected = project_path("output/analysis/action-order-sensitivity-fixed-20261014/plan.json")
    assert file_hash(selected) == plan["selection_plan_sha256"]
    assert read(selected)["image_selection"] == plan["image_selection"]
    assert file_hash(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")) == plan["strong_control_manifest_sha256"]
    calibrator = load_calibrator(frozen_control())
    raw = {r["image"]: r for r in map(json.loads, (prior/"raw_views.jsonl").read_text().splitlines())}
    records = {r["stem"]: r for items in plan["records"].values() for r in items}
    frames = dict(train=[], holdout=[])
    checked_statistics, fallbacks = 0, 0
    rounding = dict(ambiguous_point_memberships=0, inverse_mapping_point_disagreements=0)
    for split, stems in plan["image_selection"].items():
        for stem in stems:
            entry = manifest[stem]
            assert entry["split"] == split
            assert file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(root/entry["inputs_file"]) == entry["inputs_sha256"]
            record = records[stem]
            assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == record["image_sha256"] == entry["image_sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == record["metadata_sha256"] == entry["metadata_sha256"]
            metadata = read(meta_path)
            with np.load(root/entry["file"], allow_pickle=False) as z:
                f = {k: z[k].copy() for k in z.files}
            item = raw[stem]
            views = [np.array(v, float).reshape(-1, 6) for v in item["views"]]
            reference, _, _, original = calibrated_candidates(views, item["windows"], metadata["original_size"], [], calibrator)
            np.testing.assert_array_equal(reference, f["reference"])
            classes = reference[:, 5].astype(int)
            np.testing.assert_array_equal(np.column_stack((original, np.eye(10)[classes])), f["base"])
            for view, window in enumerate(item["windows"]):
                x1, y1, x2, y2 = window
                predictions = views[view].copy()
                bounds = None
                if view:
                    global_coordinates = predictions[:, :4].astype(np.float32)
                    mapped = global_coordinates.astype(float)
                    previous_value = np.nextafter(global_coordinates, np.float32(-np.inf)).astype(float)
                    next_value = np.nextafter(global_coordinates, np.float32(np.inf)).astype(float)
                    offset = np.array([x1, y1, x1, y1])
                    lower = np.nextafter(((mapped+previous_value)/2-offset).astype(np.float32), np.float32(-np.inf))
                    upper = np.nextafter(((mapped+next_value)/2-offset).astype(np.float32), np.float32(np.inf))
                    extent = np.array([x2-x1, y2-y1, x2-x1, y2-y1], np.float32)
                    bounds = (np.maximum(lower, 0), np.minimum(upper, extent))
                    predictions[:, [0, 2]] -= x1
                    predictions[:, [1, 3]] -= y1
                stats = independent_statistics(f["native_logits"][view], predictions, x2-x1, y2-y1,
                                               f["masks"][view], bounds, rounding)
                for name, values in zip(("scene_whole", "scene_masked", "fraction", "masks", "fallback"), stats):
                    check(values, f[name][view])
                checked_statistics += 2*3*10*4
                fallbacks += int(stats[-1].sum())
            tp, _, ignored = supervision(reference, metadata, .05)
            np.testing.assert_array_equal(tp, f["fit_labels"])
            np.testing.assert_array_equal((reference[:, 4] >= .05) & ~ignored, f["fit_eligible"])
            f.update(stem=stem, metadata=metadata)
            f["whole"] = f["scene_whole"][:, :, classes, :].transpose(2, 0, 1, 3).reshape(len(reference), 60)
            f["masked"] = f["scene_masked"][:, :, classes, :].transpose(2, 0, 1, 3).reshape(len(reference), 60)
            f["fraction"] = np.broadcast_to(f["fraction"].ravel(), (len(reference), 15)).copy()
            del f["native_logits"], f["masks"], f["fallback"]
            frames[split].append(f)
    stats = read(root/"normalization.json")
    centers, scales = {}, {}
    for name in ("base", "fraction"):
        values = np.concatenate([f[name][f["fit_eligible"]] for f in frames["train"]])
        centers[name], scales[name] = values.mean(0), np.maximum(values.std(0), 1e-6)
    mixed = np.concatenate([f[name][f["fit_eligible"]] for f in frames["train"] for name in ("whole", "masked")])
    for name in ("whole", "masked"):
        centers[name], scales[name] = mixed.mean(0), np.maximum(mixed.std(0), 1e-6)
    check({k: v.tolist() for k, v in centers.items()}, stats["mean"])
    check({k: v.tolist() for k, v in scales.items()}, stats["std"])
    with np.load(root/"coefficients.npz", allow_pickle=False) as z:
        coefficients = {k: z[k].copy() for k in z.files}
    for split, fs in frames.items():
        order = sorted(range(len(fs)), key=lambda i: hashlib.sha256(("candidate55-scene-shuffle:"+fs[i]["stem"]).encode()).hexdigest())
        mapping = dict(zip(order, order[1:]+order[:1]))
        for i, f in enumerate(fs):
            other = fs[mapping[i]]
            assert other["stem"] == manifest[f["stem"]]["shuffled_stem"] and other["stem"] != f["stem"]
            classes = f["reference"][:, 5].astype(int)
            shuffled = other["scene_masked"][:, :, classes, :].transpose(2, 0, 1, 3).reshape(len(classes), 60)
            base = (f["base"]-centers["base"])/scales["base"]
            whole = (f["whole"]-centers["whole"])/scales["whole"]
            masked = (f["masked"]-centers["masked"])/scales["masked"]
            coverage = (f["fraction"]-centers["fraction"])/scales["fraction"]
            shuffled = (shuffled-centers["masked"])/scales["masked"]
            zero = np.zeros_like(masked)
            rebuilt = dict(row=np.column_stack((base, zero, np.zeros_like(coverage), zero)),
                           whole=np.column_stack((base, whole, coverage, zero)),
                           masked=np.column_stack((base, whole, coverage, masked)),
                           shuffled=np.column_stack((base, whole, coverage, shuffled)))
            with np.load(root/manifest[f["stem"]]["inputs_file"], allow_pickle=False) as z:
                for name, value in rebuilt.items():
                    check(value, z[name])
            f["inputs"] = rebuilt
    stationarity = {}
    for arm in ARMS:
        x = np.concatenate([f["inputs"][arm][f["fit_eligible"]] for f in frames["train"]])
        y = np.concatenate([f["fit_labels"][f["fit_eligible"]] for f in frames["train"]])
        p = np.concatenate([f["reference"][f["fit_eligible"], 4] for f in frames["train"]])
        offset = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
        theta = coefficients[arm]
        logits = offset+x@theta[:-1]+theta[-1]
        residual = (expit(logits)-y)/len(y)
        gradient = np.r_[x.T@residual+.01*theta[:-1], residual.sum()]
        objective = float(np.mean(np.logaddexp(0, logits)-y*logits)+.005*(theta[:-1]@theta[:-1]))
        check(objective, result["training"][arm]["objective"])
        check(float(abs(gradient).max()), result["training"][arm]["gradient_inf"])
        assert abs(gradient).max() <= 1e-5 and result["training"][arm]["success"]
        stationarity[arm] = float(abs(gradient).max())
    traces = []
    labels, scores = {g: [] for g in LIMITS}, {g: {a: [] for a in ARMS} for g in LIMITS}
    for f in frames["holdout"]:
        reference = f["reference"]
        tp, duplicate, ignored = supervision(reference, f["metadata"], .25)
        areas = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)*(640/max(f["metadata"]["original_size"]))**2
        p = np.clip(reference[:, 4], 1e-6, 1-1e-6)
        offset = np.log(p/(1-p))
        logits = {a: offset+f["inputs"][a]@t[:-1]+t[-1] for a, t in coefficients.items()}
        with np.load(root/manifest[f["stem"]]["inputs_file"], allow_pickle=False) as z:
            for arm in ARMS:
                check(logits[arm], z[arm+"_logits"])
        trace = {}
        for group, limit in LIMITS.items():
            population = (reference[:, 4] >= .25) & ~ignored & (areas < limit)
            selected = population & ~duplicate
            labels[group].extend(tp[selected])
            for arm in ARMS:
                scores[group][arm].extend(logits[arm][selected])
            trace[group] = dict(rows=int(selected.sum()), tp=int((selected & tp).sum()), fp=int((selected & ~tp).sum()),
                duplicate_excluded=int((population & duplicate).sum()), images=int(selected.any()),
                loss_sums={a: math.fsum(np.logaddexp(0, v[selected])-tp[selected]*v[selected]) for a, v in logits.items()})
        traces.append(trace)
    for actual, expected in zip(traces, read(root/"per_image.json"), strict=True):
        check(actual, expected)
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    metrics = {}
    for group in LIMITS:
        total = {k: sum(t[group][k] for t in traces) for k in ("rows", "tp", "fp", "duplicate_excluded", "images")}
        total["NLL"] = {a: math.fsum(t[group]["loss_sums"][a] for t in traces)/total["rows"] for a in ARMS}
        y = np.array(labels[group], bool)
        total["AUROC"] = {a: float((rankdata(v)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
                          for a, v in scores[group].items()}
        samples = []
        for weight in draws:
            denominator = math.fsum(w*t[group]["rows"] for w, t in zip(weight, traces))
            assert denominator > 0
            samples.append(math.fsum(w*(t[group]["loss_sums"]["whole"]-t[group]["loss_sums"]["masked"])
                                     for w, t in zip(weight, traces))/denominator)
        total["whole_minus_masked_NLL_bootstrap95ci"] = np.percentile(samples, [2.5, 97.5]).tolist()
        metrics[group] = total
    check(metrics, result["metrics"])
    tiny, policy = metrics["tiny"], plan["registration"]["policy"]
    checks = dict(numerical=True, tp=tiny["tp"] >= policy["tiny_tp_min"], fp=tiny["fp"] >= policy["tiny_nonduplicate_fp_min"],
        images=tiny["images"] >= policy["tiny_images_min"],
        vs_row=tiny["NLL"]["row"]-tiny["NLL"]["masked"] >= policy["tiny_NLL_gain_vs_row_min"],
        vs_whole=tiny["NLL"]["whole"]-tiny["NLL"]["masked"] >= policy["tiny_NLL_gain_vs_whole_min"],
        vs_shuffle=tiny["NLL"]["shuffled"]-tiny["NLL"]["masked"] >= policy["tiny_NLL_gain_vs_shuffle_min"],
        AUROC=tiny["AUROC"]["masked"]-tiny["AUROC"]["whole"] >= policy["tiny_AUROC_gain_vs_whole_min"],
        interval=tiny["whole_minus_masked_NLL_bootstrap95ci"][0] > policy["tiny_NLL_gain_vs_whole_bootstrap_lower_min_exclusive"],
        all_NLL=metrics["all"]["NLL"]["whole"]-metrics["all"]["NLL"]["masked"] >= policy["all_NLL_gain_vs_whole_min"])
    assert checks == result["decision"]["checks"] and not all(checks.values())
    assert result["decision"]["decision"] == "STOP_THIS_SCENE_BACKGROUND_QUANTILE_LINEAR_RECIPE"
    audit = dict(all192_sources_frames_inputs_image_metadata_and_selection_hashes_verified=True,
                 all_original_calibration_predictions_and_features_rebuilt_with_shared_frozen_readout=True,
                 independently_rebuilt_scalar_statistics=checked_statistics, fallback_level_pools=fallbacks,
                 global192_prediction_masks_independently_rebuilt=True,
                 local768_saved_masks_consistent_with_conservative_FP32_mapping_intervals=True,
                 local_mask_rounding_diagnostics=rounding,
                 all_statistics_recomputed_from_saved_logits_and_verified_masks=True,
                 all_normalization_inputs_shuffles_fit_labels_eval_matching_NLL_AUROC_bootstrap_and_ten_gates_rebuilt=True,
                 four_convex_fit_stationarity_gradient_inf=stationarity,
                 extra_detector_forwards=0, extra_image_decodes=0, verifier_sha256=file_hash(Path(__file__)),
                 verifier_repairs="First attempted exact inverse mapping/float64 box arithmetic; corrected originalFP32 multiplication and audited local coordinate intervals without changing primary sources or numerical tolerances.",
                 limits="Local masks at ambiguous FP32 mapping boundaries are not uniquely independently reconstructed; stored memberships satisfy conservative coordinate bounds and feed statistic replay. Saved detector logits hashed, not re-extracted. Calibration uses shared frozen implementation. Train scenes repeatedly viewed; no AP, generalization or novelty proof.")
    destination = root/"independent_verification.json"
    assert not destination.exists()
    destination.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
