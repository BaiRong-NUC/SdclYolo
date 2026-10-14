"""Audit native full class scores on frozen final detections before any fitting."""

import json
from time import perf_counter

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.class_choice_signal import weighted_auc
from sdcl.evaluation.class_conflict_preflight import owned_oracle
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold,
)
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.class_conflict import apply_labels, class_evidence, vote_labels
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.rescue import archive_source, checked_data


REGISTRATION = "docs/experiments/candidate15_detection_soft_evidence_registration_20261014.json"
OUTPUT = "output/analysis/detection-soft-evidence-fixed-20261014"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("uniform", "hard_vote", "soft_vote", "hard_oracle", "soft_oracle")
GROUPS = ("all", "tiny", "small_all")


def posterior_evidence(baseline, views, posterior, hard_valid):
    n = len(baseline)
    evidence = np.zeros((n, 10, 5), np.float32)
    valid = hard_valid.copy()
    for view, (boxes, probability) in enumerate(zip(views, posterior)):
        boxes = np.asarray(boxes).reshape(-1, 6)
        probability = np.asarray(probability)
        if probability.shape != (len(boxes), 10) or not np.isfinite(probability).all():
            raise ValueError("Native per-view probabilities differ from raw boxes.")
        if not len(boxes):
            continue
        overlap = box_iou_matrix(baseline[:, :4], boxes[:, :4])
        quality = np.where((overlap >= .7) & (boxes[None, :, 4] >= .05),
                           overlap * boxes[None, :, 4], 0)
        best = quality.argmax(1)
        active = quality[np.arange(n), best] > 0
        selected = probability[best]
        evidence[:, :, view] = selected * overlap[np.arange(n), best, None] * active[:, None]
        top3 = np.argsort(-selected, axis=1, kind="stable")[:, :3]
        for rank in range(3):
            cls = top3[:, rank]
            eligible = active & (selected[np.arange(n), cls] >= .05)
            valid[np.flatnonzero(eligible), cls[eligible]] = True
    return valid, evidence


def native_view(extractor, image):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    result, _ = extractor.predict(image)
    original = result["control"]
    boxes, indices = non_max_suppression(
        extractor.decoded.clone(), .001, .7, max_det=500, return_idxs=True)
    native = boxes[0]
    shape = tuple(int(x * extractor.head.stride[0]) for x in extractor.raw["feats"][0].shape[2:])
    native[:, :4] = scale_boxes(shape, native[:, :4], image.shape[:2])
    if not np.array_equal(native.float().cpu().numpy(), original):
        raise ValueError("Native kept-anchor replay differs from original inference.")
    kept = indices[0].reshape(-1).long()
    probability = extractor.decoded[0, 4:14, kept].T.float().cpu().numpy()
    if len(original):
        row = np.arange(len(original))
        if (not np.array_equal(probability.argmax(1), original[:, 5].astype(int))
                or not np.array_equal(probability[row, original[:, 5].astype(int)], original[:, 4])):
            raise ValueError("Native posterior row does not belong to its kept detection.")
    return original, probability


def decide(metrics, count, auc, policy):
    delta = (metrics["soft_oracle"]["small_all_recall_percent"]
             - metrics["hard_oracle"]["small_all_recall_percent"])
    checks = {
        "new_class_actions": count >= policy["new_owned_correct_class_actions_at025_min"],
        "extra_action_headroom": delta >= policy["soft_oracle_vs_hard_oracle_small_all_recall_min_pp"],
        "prediction_discrimination": auc is not None and auc >= policy["aggregate_gap_correctness_auc_min"],
    }
    return {
        "decision": "TINY_SOFT_INTERVENTION_PILOT_ALLOWED" if all(checks.values()) else
        "STOP_THIS_NATIVE_SOFT_CLASS_MENU",
        "checks": checks, "new_owned_correct_class_actions_at025": count,
        "soft_oracle_vs_hard_oracle_small_all_recall_pp": delta,
        "aggregate_gap_correctness_auc": auc,
        "scope": "Final-box diagnostic, not official AP/AR, independent confirmation or innovation.",
    }


def evaluate(device="0", output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    precursor = json.loads(project_path(
        "docs/experiments/candidate14_soft_class_information_result_20261014.json").read_text())
    if precursor["decision"]["decision"] != "RAW_POSTERIOR_AUDIT_ONLY":
        raise ValueError("Frozen class-information screen has not passed.")
    precursor_dir = project_path("output/analysis/soft-class-information-fixed-20261014")
    for name, digest in precursor["artifacts_sha256"].items():
        if file_hash(precursor_dir / name) != digest:
            raise ValueError("Soft class-information evidence changed.")
    prior = project_path(PRIOR)
    prior_result = json.loads(project_path(
        "docs/experiments/candidate12_class_choice_result_20261014.json").read_text())
    for name, digest in prior_result["artifacts_sha256"].items():
        if file_hash(prior / name) != digest:
            raise ValueError("Frozen class-choice pilot inputs changed.")
    plan = json.loads((prior / "plan.json").read_text())
    verify_dependencies(plan["source_hashes"], prior / "source.zip")
    _, checkpoint, all_records = checked_data()
    lookup = {r["stem"]: r for r in all_records}
    records = [lookup[s] for s in plan["image_selection"]["holdout"]]
    if len(records) != 128:
        raise ValueError("Fixed head-held-out image list changed.")
    frames = json.loads((prior / "frames.json").read_text())
    raw = {}
    with (prior / "raw_views.jsonl").open() as stream:
        for line in stream:
            value = json.loads(line)
            raw[value["image"]] = value
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    paths = (REGISTRATION, "sdcl/evaluation/detection_soft_evidence.py",
             "scripts/evaluation/diagnose_detection_soft_evidence.py",
             "sdcl/training/class_margin_probe.py", "sdcl/postprocess/class_margin_probe.py",
             "sdcl/postprocess/class_conflict.py", "sdcl/evaluation/class_conflict_preflight.py",
             "sdcl/evaluation/class_choice_signal.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "candidate12_plan_sha256": file_hash(prior / "plan.json"),
        "candidate12_record_sha256": file_hash(project_path(
            "docs/experiments/candidate12_class_choice_result_20261014.json")),
        "candidate14_record_sha256": file_hash(project_path(
            "docs/experiments/candidate14_soft_class_information_result_20261014.json")),
        "detector_sha256": file_hash(checkpoint), "images": [r["stem"] for r in records],
        "installed_nms_sha256": file_hash(__import__(
            "ultralytics.utils.nms", fromlist=["non_max_suppression"]).__file__),
    })
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    extractor = FrozenViews(device)
    totals = {a: {g: {"gt": 0, "tp": 0} for g in GROUPS} for a in ARMS}
    fp, changes = dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0)
    new_count, audit_vectors, index_manifest = 0, [], {}
    try:
        with torch.inference_mode():
            for index, record in enumerate(records, 1):
                stem = record["stem"]
                if (file_hash(record["image"]) != record["image_sha256"] or
                        file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json"))
                        != record["metadata_sha256"]):
                    raise ValueError("Training image or annotation changed.")
                image = cv2.imread(record["image"])
                if image is None or list(image.shape[1::-1]) != record["size"]:
                    raise ValueError("Invalid fixed training image.")
                windows = raw[stem]["windows"]
                views, probability = [], []
                for view, window in enumerate(windows):
                    x1, y1, x2, y2 = window
                    boxes, posterior = native_view(extractor, image[y1:y2, x1:x2].copy())
                    if view:
                        boxes = map_tile_predictions(boxes, window)
                    expected = np.asarray(raw[stem]["views"][view], np.float32).reshape(-1, 6)
                    if not np.array_equal(boxes, expected):
                        raise ValueError(f"Frozen raw-view replay differs at {stem}, view {view}.")
                    views.append(boxes)
                    probability.append(posterior)
                with np.load(prior / frames[stem]["file"], allow_pickle=False) as cached:
                    baseline = cached["baseline"]
                hard_valid, hard_score, _, _ = class_evidence(
                    baseline, views, windows, record["size"])
                soft_valid, soft_score = posterior_evidence(
                    baseline, views, probability, hard_valid)
                hard_oracle, _ = owned_oracle(record, baseline, hard_valid)
                soft_oracle, _ = owned_oracle(record, baseline, soft_valid)
                gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
                cls = np.array([o["class_id"] for o in record["objects"]], int)
                iou = box_iou_matrix(gt, baseline[:, :4])
                geometry = baseline.copy()
                geometry[:, 5] = 0
                owned, _ = match_at_threshold(iou, np.zeros(len(cls), int), geometry, .001)
                owner = np.full(len(baseline), -1, int)
                mask = owned >= 0
                owner[owned[mask]] = cls[mask]
                original = baseline[:, 5].astype(int)
                rows = np.arange(len(baseline))
                at_point = baseline[:, 4] >= .25
                active = (owner >= 0) & at_point
                correct = owner[active] == original[active]
                aggregated = soft_score.sum(2)
                ranked = np.argsort(-aggregated, axis=1, kind="stable")
                gap = aggregated[rows, ranked[:, 0]] - aggregated[rows, ranked[:, 1]]
                audit_vectors.append((correct, gap[active]))
                newly = active & (owner != original)
                candidates = np.flatnonzero(newly)
                new_count += int((soft_valid[candidates, owner[candidates]]
                                  & ~hard_valid[candidates, owner[candidates]]).sum())
                predictions = {
                    "uniform": baseline,
                    "hard_vote": apply_labels(baseline, vote_labels(baseline, hard_valid, hard_score)),
                    "soft_vote": apply_labels(baseline, vote_labels(baseline, soft_valid, soft_score)),
                    "hard_oracle": apply_labels(baseline, hard_oracle),
                    "soft_oracle": apply_labels(baseline, soft_oracle),
                }
                areas = box_areas(gt) * (640 / max(record["size"])) ** 2
                for arm, values in predictions.items():
                    matched, _ = match_at_threshold(iou, cls, values, .25)
                    fp[arm] += int(at_point.sum() - (matched >= 0).sum())
                    changes[arm] += int((values[:, 5] != original).sum())
                    for group in GROUPS:
                        include = group_mask(areas, group)
                        totals[arm][group]["gt"] += int(include.sum())
                        totals[arm][group]["tp"] += int((include & (matched >= 0)).sum())
                target = output / "frames" / f"{stem}.npz"
                np.savez_compressed(target, baseline=baseline, hard_valid=hard_valid,
                                    soft_valid=soft_valid, hard_score=hard_score,
                                    soft_score=soft_score, owner=owner)
                index_manifest[stem] = {"file": target.relative_to(output).as_posix(),
                                        "sha256": file_hash(target)}
                if index == 1 or index % 32 == 0:
                    print(f"Native soft class audit: {index}/128; {perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    correct, gap = (np.concatenate([v[i] for v in audit_vectors]) for i in range(2))
    auc = weighted_auc(correct, gap, np.ones(len(gap)))
    metrics = {
        a: {**{g + "_recall_percent": v["tp"] / max(v["gt"], 1) * 100
               for g, v in counts.items()}, "all_fppi": fp[a] / 128,
            "changed_labels_all_scores": changes[a], "gt_tp": counts}
        for a, counts in totals.items()
    }
    verdict = decide(metrics, new_count, auc, registration["policy"])
    save_json(output / "frames.json", index_manifest)
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "raw_view_and_native_kept_anchor_replay_exact": True,
        "detector_view_forwards_excluding_one_warmup": 640,
        "no_training": True, "no_threshold_search": True,
        "owned_original_correct_at025": int(correct.sum()),
        "owned_original_wrong_at025": int((~correct).sum()),
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
