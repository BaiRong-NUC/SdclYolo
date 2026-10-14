"""Inventory pre/post-NMS evidence on unchanged frozen views before training."""

import json
from time import perf_counter

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import max_pool_matches
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold,
)
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.rescue import clean_views, save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate21_native_censoring_registration_20261014.json"
OUTPUT = "output/analysis/native-censoring-fixed-20261014"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
GROUPS = ("all", "tiny", "small_all")
POOLS = ("dense", "uncapped", "kept")
POINTS = ((.05, .5), (.25, .5), (.05, .75), (.25, .75))


def native_inventory(decoded, input_shape, original_shape, control):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy

    if decoded.shape[1] != 14 or decoded.shape[-1] >= 30000:
        raise ValueError("Unexpected native output or max_nms would add another censor.")
    values = decoded[0].T
    score, cls = values[:, 4:14].max(1, keepdim=True)
    before = torch.cat((xywh2xyxy(values[:, :4]), score, cls.float()), 1)
    before = before[score[:, 0] > .001]
    before[:, :4] = scale_boxes(input_shape, before[:, :4], original_shape)
    all_kept, indices = non_max_suppression(
        decoded.clone(), .001, .7, max_det=30000, return_idxs=True)
    after = all_kept[0]
    after[:, :4] = scale_boxes(input_shape, after[:, :4], original_shape)
    after_np = after.float().cpu().numpy()
    if not np.array_equal(after_np[:500], control):
        raise ValueError("Uncapped NMS prefix differs from original capped replay.")
    index = indices[0].reshape(-1).long()
    if len(index):
        selected = torch.cat((xywh2xyxy(values[index, :4]), score[index], cls[index].float()), 1)
        selected[:, :4] = scale_boxes(input_shape, selected[:, :4], original_shape)
        if not torch.equal(selected, after):
            raise ValueError("Decoded anchor coordinates do not match the kept native rows.")
    return before.float().cpu().numpy(), after_np


def pool_edges(gt, classes, pool, confidence, threshold):
    ious = box_iou_matrix(gt, pool[:, :4])
    return ((ious >= threshold) & (classes[:, None] == pool[None, :, 5]) &
            (pool[None, :, 4] >= confidence))


def image_metrics(record, baseline, pools, windows):
    gt = np.asarray([o["box_xyxy"] for o in record["objects"]], float).reshape(-1, 4)
    classes = np.asarray([o["class_id"] for o in record["objects"]], int)
    areas = box_areas(gt) * (640 / max(record["size"])) ** 2
    matched, _ = match_at_threshold(
        box_iou_matrix(gt, baseline[:, :4]), classes, baseline, .25)
    results, view_results = {}, {}
    for confidence, threshold in POINTS:
        key = f"conf{confidence:g}_iou{threshold:g}"
        edges = {name: pool_edges(gt, classes, pool, confidence, threshold)
                 for name, (pool, _) in pools.items()}
        support = {name: values.any(1) for name, values in edges.items()}
        newly = (matched < 0) & ~support["kept"] & support["dense"]
        point = {}
        for group in GROUPS:
            selected = group_mask(areas, group)
            point[group] = {
                "gt": int(selected.sum()),
                "baseline_missed": int(((matched < 0) & selected).sum()),
                "baseline_missed_new_dense_support": int((newly & selected).sum()),
                **{name + "_supported_gt": int(values[selected].any(1).sum())
                   for name, values in edges.items()},
                **{name + "_max_matches": max_pool_matches(values[selected])
                   for name, values in edges.items()},
            }
        results[key] = point
        if confidence == .05 and threshold == .5:
            counts = {g: dict.fromkeys(
                ("fully_visible_opportunities", "missing_kept", "explained_by_nms",
                 "explained_by_cap", "no_qualifying_dense_argmax"), 0) for g in GROUPS}
            for view, window in enumerate(windows):
                low = np.maximum(gt[:, :2], window[:2])
                high = np.minimum(gt[:, 2:], window[2:])
                fully = np.maximum(high-low, 0).prod(1) / np.maximum(box_areas(gt), 1e-12) >= .9
                available = {name: edge[:, pools[name][1] == view].any(1)
                             for name, edge in edges.items()}
                missing = fully & ~available["kept"]
                for group in GROUPS:
                    selected = group_mask(areas, group)
                    counts[group]["fully_visible_opportunities"] += int((selected & fully).sum())
                    counts[group]["missing_kept"] += int((selected & missing).sum())
                    counts[group]["explained_by_nms"] += int(
                        (selected & missing & available["dense"] & ~available["uncapped"]).sum())
                    counts[group]["explained_by_cap"] += int(
                        (selected & missing & available["uncapped"]).sum())
                    counts[group]["no_qualifying_dense_argmax"] += int(
                        (selected & missing & ~available["dense"]).sum())
            view_results = counts
    return results, view_results


def decide(metrics, policy):
    primary = metrics["conf0.05_iou0.5"]
    tiny, small = primary["tiny"], primary["small_all"]
    values = {
        "tiny_baseline_missed_new_dense_support_count": tiny["baseline_missed_new_dense_support"],
        "tiny_max_matching_gain_pp": (tiny["dense_max_matches"] - tiny["kept_max_matches"]) /
        max(tiny["gt"], 1) * 100,
        "small_all_max_matching_gain_pp": (small["dense_max_matches"] - small["kept_max_matches"]) /
        max(small["gt"], 1) * 100,
    }
    checks = {
        "new_missed_target_support": values["tiny_baseline_missed_new_dense_support_count"] >=
        policy["tiny_baseline_missed_new_dense_support_count_min"],
        "unique_tiny_capacity": values["tiny_max_matching_gain_pp"] >=
        policy["tiny_dense_vs_kept_max_matching_gain_min_pp"],
        "unique_small_all_capacity": values["small_all_max_matching_gain_pp"] >=
        policy["small_all_dense_vs_kept_max_matching_gain_min_pp"],
    }
    return {"decision": "CENSORING_RETRIEVAL_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_NATIVE_CENSORING_RECIPE", "checks": checks, "values": values,
            "scope": "GT-assisted candidate inventory only; not AP/AR or evidence of a new fusion method."}


def evaluate(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior = project_path(PRIOR)
    _, plan = verify_record(prior, "docs/experiments/candidate12_class_choice_result_20261014.json")
    stems = plan["image_selection"]["holdout"][:64]
    if len(stems) != 64:
        raise ValueError("Fixed image list changed.")
    snapshots = {r["stem"]: r for r in plan["records"]["holdout"]}
    frames = json.loads((prior / "frames.json").read_text())
    raw = {r["image"]: r for r in
           (json.loads(line) for line in (prior / "raw_views.jsonl").read_text().splitlines())}
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    import ultralytics.utils.nms as installed_nms
    paths = (REGISTRATION, "sdcl/evaluation/native_censoring.py",
             "scripts/evaluation/diagnose_native_censoring.py",
             "sdcl/training/class_margin_probe.py", "sdcl/evaluation/headroom_screen.py",
             "sdcl/postprocess/rescue.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_plan_sha256": file_hash(prior / "plan.json"),
        "prior_raw_views_sha256": file_hash(prior / "raw_views.jsonl"),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate12_class_choice_result_20261014.json")),
        "installed_nms_path": installed_nms.__file__,
        "installed_nms_sha256": file_hash(installed_nms.__file__),
        "images": stems, "detector_sha256": plan["detector_sha256"],
    })
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    extractor = FrozenViews(device)
    totals, view_totals, outputs = {}, {}, {}
    forwards, views_over500, pool_counts = 0, 0, dict.fromkeys(POOLS, 0)
    try:
        with torch.inference_mode():
            for index, stem in enumerate(stems, 1):
                meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                if (file_hash(meta_path) != snapshots[stem]["metadata_sha256"] or
                        file_hash(image_path) != snapshots[stem]["image_sha256"]):
                    raise ValueError("Fixed train image/metadata changed.")
                meta = json.loads(meta_path.read_text())
                record = {"size": meta["original_size"], "objects": meta["valid_objects"]}
                image = cv2.imread(str(image_path))
                if image is None or list(image.shape[1::-1]) != record["size"]:
                    raise ValueError("Image dimensions changed.")
                windows = raw[stem]["windows"]
                inventories = {name: [] for name in POOLS}
                for view, window in enumerate(windows):
                    x1, y1, x2, y2 = window
                    crop = image[y1:y2, x1:x2].copy()
                    prediction, _ = extractor.predict(crop)
                    kept = prediction["control"]
                    input_shape = tuple(int(x * extractor.head.stride[0])
                                        for x in extractor.raw["feats"][0].shape[2:])
                    before, uncapped = native_inventory(
                        extractor.decoded, input_shape, crop.shape[:2], kept)
                    views_over500 += int(len(uncapped) > 500)
                    forwards += 1
                    for name, values in (("dense", before), ("uncapped", uncapped), ("kept", kept)):
                        mapped = map_tile_predictions(values, window) if view else values
                        if name == "kept" and not np.array_equal(
                                mapped, np.asarray(raw[stem]["views"][view], np.float32).reshape(-1, 6)):
                            raise ValueError("Frozen kept raw-view replay differs; do not change control.")
                        inventories[name].append(mapped[mapped[:, 4] >= .05])
                # All native inventory construction is annotation-free; this filter is diagnostic only.
                pools = {name: clean_views(views, record["size"], meta["ignore_boxes_xyxy"])
                         for name, views in inventories.items()}
                cached = prior / frames[stem]["file"]
                if file_hash(cached) != frames[stem]["sha256"]:
                    raise ValueError("Frozen baseline frame changed.")
                with np.load(cached, allow_pickle=False) as saved:
                    baseline = saved["baseline"]
                metrics, views = image_metrics(record, baseline, pools, windows)
                for point, groups in metrics.items():
                    totals.setdefault(point, {})
                    for group, counts in groups.items():
                        destination = totals[point].setdefault(group, dict.fromkeys(counts, 0))
                        for key, value in counts.items():
                            destination[key] += value
                for group, counts in views.items():
                    destination = view_totals.setdefault(group, dict.fromkeys(counts, 0))
                    for key, value in counts.items():
                        destination[key] += value
                target = output / "frames" / f"{stem}.npz"
                np.savez_compressed(target, **{
                    key: value for name, (pool, origin) in pools.items()
                    for key, value in ((name, pool), (name + "_view", origin))})
                for name, (pool, _) in pools.items():
                    pool_counts[name] += len(pool)
                outputs[stem] = {"file": target.relative_to(output).as_posix(), "sha256": file_hash(target),
                                 "metrics": metrics, "views": views}
                if index == 1 or index % 16 == 0 or index == 64:
                    print(f"Native-censoring inventory: {index}/64, "
                          f"{perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    verdict = decide(totals, registration["policy"])
    save_json(output / "frames.json", outputs)
    save_json(output / "metrics.json", totals)
    save_json(output / "view_missingness.json", view_totals)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": 64, "detector_forwards_excluding_warmup": forwards,
        "views_exceeding500_after_nms": views_over500,
        "stored_pool_rows_at_native_conf005_after_diagnostic_ignore": pool_counts,
        "kept_original_and_uncapped_prefix_replay_exact": True,
        "kept_anchor_coordinates_match_decoded_exact": True,
        "no_fitting_no_threshold_search": True, "no_val_no_testdev": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "installed_nms_unchanged": file_hash(installed_nms.__file__) ==
        json.loads((output / "plan.json").read_text())["installed_nms_sha256"],
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": totals,
            "view_missingness": view_totals}
