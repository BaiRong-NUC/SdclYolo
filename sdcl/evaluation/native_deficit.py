"""Uncensored native geometry/class deficit inventory to route future research."""

import json
from time import perf_counter

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate36_native_deficit_registration_20261014.json"
OUTPUT = "output/analysis/native-deficit-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
THRESHOLDS = (.5, .75)
CATEGORIES = ("native_support_not_realized", "nonargmax_class_suppression",
              "low_true_class_confidence", "class_evidence_almost_absent", "geometry_absent")
SUPPORTS = ("geometry", "true_prob001", "true_prob005", "correct_argmax005")


def native_boxes(extractor, image, window):
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy

    prediction, _ = extractor.predict(image)
    decoded = extractor.decoded[0].T
    if decoded.shape != (18900, 14):
        raise ValueError("Unexpected frozen square960 native grid.")
    shape = tuple(int(x*extractor.head.stride[0]) for x in extractor.raw["feats"][0].shape[2:])
    coordinates = xywh2xyxy(decoded[:, :4]).clone()
    coordinates = scale_boxes(shape, coordinates, image.shape[:2])
    # Historical tile mapping converts scaled FP16 coordinates to FP32 before
    # adding original-image offsets; preserve that order at large coordinates.
    coordinates = coordinates.float()
    x1, y1, x2, y2 = window
    coordinates[:, [0, 2]] = coordinates[:, [0, 2]].clamp(0, x2-x1)+x1
    coordinates[:, [1, 3]] = coordinates[:, [1, 3]].clamp(0, y2-y1)+y1
    probabilities = decoded[:, 4:14]
    valid = (torch.isfinite(coordinates).all(1) & torch.isfinite(probabilities).all(1) &
             (coordinates[:, 2:] > coordinates[:, :2]).all(1))
    return prediction["control"], coordinates[valid].float(), probabilities[valid].float(), int(valid.sum())


def native_support(coordinates, probabilities, gt, classes):
    gt = torch.as_tensor(gt, dtype=torch.float32, device=coordinates.device).reshape(-1, 4)
    classes = torch.as_tensor(classes, dtype=torch.long, device=coordinates.device)
    supports = {str(t): {k: np.zeros(len(gt), bool) for k in SUPPORTS} for t in THRESHOLDS}
    max_iou = np.zeros(len(gt), np.float32)
    max_score = {str(t): np.zeros(len(gt), np.float32) for t in THRESHOLDS}
    if not len(coordinates):
        return supports, max_iou, max_score
    native_area = (coordinates[:, 2:]-coordinates[:, :2]).prod(1)
    native_max, native_class = probabilities.max(1)
    for start in range(0, len(gt), 64):
        target = gt[start:start+64]
        target_class = classes[start:start+64]
        lo = torch.maximum(target[:, None, :2], coordinates[None, :, :2])
        hi = torch.minimum(target[:, None, 2:], coordinates[None, :, 2:])
        intersection = (hi-lo).clamp_min(0).prod(-1)
        area = (target[:, 2:]-target[:, :2]).clamp_min(0).prod(1)
        overlap = intersection/(area[:, None]+native_area[None]-intersection).clamp_min(1e-12)
        true_score = probabilities[:, target_class].T
        same = native_class[None] == target_class[:, None]
        stop = start+len(target)
        max_iou[start:stop] = overlap.max(1).values.cpu().numpy()
        for threshold in THRESHOLDS:
            key = str(threshold)
            geometry = overlap >= threshold
            values = {
                "geometry": geometry.any(1),
                "true_prob001": (geometry & (true_score >= .001)).any(1),
                "true_prob005": (geometry & (true_score >= .05)).any(1),
                "correct_argmax005": (geometry & same & (native_max[None] >= .05)).any(1),
            }
            for name, value in values.items():
                supports[key][name][start:stop] = value.cpu().numpy()
            max_score[key][start:stop] = torch.where(geometry, true_score, 0).max(1).values.cpu().numpy()
    return supports, max_iou, max_score


def exclusive_categories(support):
    n = len(support["geometry"])
    result = np.full(n, 4, int)
    result[support["geometry"]] = 3
    result[support["true_prob001"]] = 2
    result[support["true_prob005"]] = 1
    result[support["correct_argmax005"]] = 0
    return result


def route(metrics, policy):
    primary = metrics["0.5"]["tiny"]
    high = metrics["0.75"]["tiny"]
    misses = max(primary["missed_gt"], 1)
    geometry = primary["categories"]["geometry_absent"]
    classification = sum(primary["categories"][k] for k in CATEGORIES[1:4])
    downstream = primary["categories"][CATEGORIES[0]]
    checks = {
        "geometry": (geometry >= policy["geometry_absent_tiny_min_count"] and
                     geometry/misses >= policy["geometry_absent_tiny_miss_fraction_min"] and
                     high["categories"]["geometry_absent"]/max(high["missed_gt"], 1) >=
                     policy["geometry_absent_tiny_miss_fraction_at075_min"]),
        "classification": (classification >= policy["classification_deficit_tiny_min_count"] and
                           classification/misses >= policy["classification_deficit_tiny_miss_fraction_min"]),
        "downstream": (downstream >= policy["downstream_native_support_tiny_min_count"] and
                       downstream/misses >= policy["downstream_native_support_tiny_miss_fraction_min"]),
    }
    if checks["geometry"] and checks["classification"]:
        chosen = "geometry" if geometry >= classification else "classification"
    else:
        chosen = "geometry" if checks["geometry"] else "classification" if checks["classification"] else None
    return {"decision": "BOUNDED_NATIVE_"+chosen.upper()+"_FEASIBILITY_SCREEN_ALLOWED" if chosen else
            "NO_DOMINANT_NATIVE_REPRESENTATION_ROUTE", "checks": checks,
            "values": {"tiny_misses050": primary["missed_gt"], "geometry_absent": geometry,
                       "classification_deficit": classification, "downstream_support": downstream,
                       "primary_fractions": {"geometry": geometry/misses, "classification": classification/misses,
                                             "downstream": downstream/misses}},
            "scope": "Annotation-assisted bottleneck routing, not AP, deployment, novelty or full training authorization."}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Available CUDA index required.")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/evaluation/native_deficit.py", "scripts/evaluation/diagnose_native_deficit.py",
             "sdcl/training/class_margin_probe.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/evaluation/slicing_screen.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    import ultralytics.utils.nms as installed_nms
    import ultralytics.utils.ops as installed_ops
    installed = {str(p): file_hash(p) for p in (installed_nms.__file__, installed_ops.__file__)}
    archive_source(output, hashes)
    frames, menu, heads = load_frames()
    frames = frames["holdout"]
    base = project_path(RAW)
    _, plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    records = {r["stem"]: r for r in plan["records"]["holdout"]}
    raw_path = base/"raw_views.jsonl"
    raw = {r["image"]: r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "images": menu["image_selection"]["holdout"],
        "source_archive_sha256": file_hash(output/"source.zip"), "heads": heads,
        "installed_sources": installed,
        "raw_views_sha256": file_hash(raw_path), "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json"))})
    metrics = {str(t): {g: {"gt": 0, "missed_gt": 0, "categories": dict.fromkeys(CATEGORIES, 0)}
                       for g in ("all", "tiny", "small_all")} for t in THRESHOLDS}
    manifest, forwards = {}, 0
    (output/"frames").mkdir()
    extractor = FrozenViews(device)
    try:
        with torch.inference_mode():
            for index, f in enumerate(frames, 1):
                stem = f["stem"]
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                if file_hash(image_path) != records[stem]["image_sha256"]:
                    raise ValueError("Frozen native audit image changed.")
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("Unreadable native audit image.")
                objects = f["metadata"]["valid_objects"]
                gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
                classes = np.asarray([o["class_id"] for o in objects], int)
                areas = box_areas(gt)*(640/max(f["metadata"]["original_size"]))**2
                perview, valid_native, max_iou, max_score = [], [], [], []
                for view, window in enumerate(raw[stem]["windows"]):
                    x1, y1, x2, y2 = window
                    kept, coordinates, probability, count = native_boxes(extractor, image[y1:y2, x1:x2].copy(), window)
                    mapped = map_tile_predictions(kept, window) if view else kept
                    if not np.array_equal(mapped, np.asarray(raw[stem]["views"][view], np.float32).reshape(-1, 6)):
                        raise ValueError("Frozen five-view raw replay differs.")
                    supports, iou, score = native_support(coordinates, probability, gt, classes)
                    perview.append(supports)
                    valid_native.append(count)
                    max_iou.append(iou)
                    max_score.append(score)
                    forwards += 1
                reference = apply_actions(f["candidates"], f["boxes"], f["regret"])
                reference_iou = box_iou_matrix(gt, reference[:, :4])
                traces = {}
                for threshold in THRESHOLDS:
                    key = str(threshold)
                    combined = {name: np.stack([v[key][name] for v in perview]).any(0) for name in SUPPORTS}
                    category = exclusive_categories(combined)
                    matched, _ = match_at_threshold(reference_iou, classes, reference, .25, threshold)
                    missing = matched < 0
                    for group, totals in metrics[key].items():
                        include = group_mask(areas, group)
                        totals["gt"] += int(include.sum())
                        totals["missed_gt"] += int((include & missing).sum())
                        for c, name in enumerate(CATEGORIES):
                            totals["categories"][name] += int((include & missing & (category == c)).sum())
                    traces.update({key+"_"+name: np.stack([v[key][name] for v in perview]) for name in SUPPORTS})
                    traces[key+"_category"] = category
                    traces[key+"_regret_matched"] = matched
                    traces[key+"_native_max_true_score"] = np.stack([v[key] for v in max_score])
                path = output/"frames"/(stem+".npz")
                np.savez_compressed(path, **traces, gt=gt, classes=classes, areas=areas,
                                    native_max_iou=np.stack(max_iou), valid_native_counts=valid_native)
                manifest[stem] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path),
                                  "image_sha256": records[stem]["image_sha256"]}
                if index == 1 or index%16 == 0:
                    print(f"Uncensored native deficit: {index}/64, {perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    for row in metrics.values():
        for values in row.values():
            if sum(values["categories"].values()) != values["missed_gt"]:
                raise ValueError("Native deficit partition is not exhaustive/disjoint.")
    verdict = route(metrics, registration["policy"])
    save_json(output/"frames.json", manifest)
    save_json(output/"metrics.json", metrics)
    save_json(output/"decision.json", verdict)
    save_json(output/"verification.json", {
        "head_holdout_train_images": 64, "detector_forwards_excluding_warmup": forwards, "fitting_steps": 0,
        "raw_all_five_view_kept_replay_exact": True, "uncensored18900_native_anchors_before_confidence": True,
        "all_ten_native_probabilities": True, "native_inventory_not_modified_using_GT": True,
        "categories_partition_all_regret_misses_exactly": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "installed_sources_unchanged": all(file_hash(p) == h for p, h in installed.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "metrics": metrics}
