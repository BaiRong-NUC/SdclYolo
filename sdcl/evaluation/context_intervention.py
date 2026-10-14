"""Bounded original-pixel context intervention with matched zoom and fixed anchors."""

import json
from time import perf_counter

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source, checked_data


REGISTRATION = "docs/experiments/candidate17_context_intervention_registration_20261014.json"
OUTPUT = "output/analysis/context-intervention-fixed-20261014"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
SOFT = "output/analysis/detection-soft-evidence-fixed-20261014"
ARMS = ("uniform", "raw_crop", "far_context_blur", "far_context_fill")
GROUPS = ("all", "tiny", "small_all")
SIDE = 192


def selected_rows(baseline, soft_score, size):
    wh = baseline[:, 2:4] - baseline[:, :2]
    eligible = np.flatnonzero(
        (baseline[:, 4] >= .25) & (np.sqrt(wh.prod(1)) * 640 / max(size) < 32)
        & (wh <= 96).all(1))
    score = soft_score.sum(2)
    ordered = np.sort(score, axis=1)[:, ::-1]
    gap = (ordered[:, 0] - ordered[:, 1]) / np.maximum(ordered[:, 0], 1e-6)
    uncertain = eligible[np.argsort(gap[eligible], kind="stable")[:4]]
    remaining = eligible[~np.isin(eligible, uncertain)]
    confident = remaining[np.argsort(-baseline[remaining, 4], kind="stable")[:4]]
    return np.r_[uncertain, confident]


def crop_variants(image, box):
    center = (box[:2] + box[2:4]) / 2
    low = np.floor(center - SIDE / 2).astype(int)
    high = low + SIDE
    patch = np.full((SIDE, SIDE, 3), 114, np.uint8)
    source_low = np.maximum(low, 0)
    source_high = np.minimum(high, [image.shape[1], image.shape[0]])
    if (source_high > source_low).all():
        a, b = source_low - low, source_high - low
        patch[a[1]:b[1], a[0]:b[0]] = image[
            source_low[1]:source_high[1], source_low[0]:source_high[0]]
    local_box = np.asarray(box[:4], float) - np.tile(low, 2)
    wh = box[2:4] - box[:2]
    keep_low = np.floor(center - wh - low).astype(int).clip(0, SIDE)
    keep_high = np.ceil(center + wh - low).astype(int).clip(0, SIDE)
    keep = np.zeros((SIDE, SIDE), bool)
    keep[keep_low[1]:keep_high[1], keep_low[0]:keep_high[0]] = True
    blur = cv2.GaussianBlur(patch, (31, 31), 0)
    fill = np.full_like(patch, 114)
    blur[keep], fill[keep] = patch[keep], patch[keep]
    if not np.array_equal(blur[keep], patch[keep]) or not np.array_equal(fill[keep], patch[keep]):
        raise AssertionError("Context intervention changed preserved target pixels.")
    return (patch, blur, fill), local_box, keep


def anchor_choice(decoded, box):
    center, wh = decoded[:, :2], decoded[:, 2:4]
    boxes = np.column_stack((center - wh / 2, center + wh / 2)) * (SIDE / 960)
    overlap = box_iou_matrix(np.asarray(box).reshape(1, 4), boxes)[0]
    score = decoded[:, 4:14].max(1)
    utility = np.where((overlap >= .5) & (score >= .001), score * overlap, 0)
    return int(utility.argmax()) if len(utility) and utility.max() > 0 else -1


def class_choice(probability, original):
    best = probability.argmax(1)
    rows = np.arange(len(best))
    changed = (probability[rows, best] >= .05) & (
        probability[rows, best] - probability[rows, original] > .1)
    return np.where(changed, best, original)


def decide(metrics, wrong, policy):
    blur, base, raw = metrics["far_context_blur"], metrics["uniform"], metrics["raw_crop"]
    values = {
        "owned_original_wrong": wrong,
        "small_all_vs_uniform_pp": blur["small_all_recall_percent"] - base["small_all_recall_percent"],
        "small_all_vs_raw_crop_pp": blur["small_all_recall_percent"] - raw["small_all_recall_percent"],
        "all_fppi_increase": blur["all_fppi"] - base["all_fppi"],
    }
    checks = {
        "wrong_sample_count": wrong >= policy["owned_original_wrong_count_min"],
        "baseline_gain": values["small_all_vs_uniform_pp"] >= policy["blur_small_all_recall_vs_uniform_min_pp"],
        "matched_context_gain": values["small_all_vs_raw_crop_pp"] >
        policy["blur_small_all_recall_vs_raw_crop_min_exclusive_pp"],
        "fp_budget": values["all_fppi_increase"] <= policy["blur_all_fppi_vs_uniform_max_increase"],
    }
    return {"decision": "CONTEXT_SIGNAL_WORTH_CONTROLLED_PILOT" if all(checks.values()) else
            "STOP_THIS_CONTEXT_INTERVENTION", "checks": checks, "values": values,
            "scope": "Frozen sampled context intervention, not AP/AR or novelty."}


def evaluate(device="0", output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior, soft = project_path(PRIOR), project_path(SOFT)
    _, plan = verify_record(prior, "docs/experiments/candidate12_class_choice_result_20261014.json")
    verify_record(soft, "docs/experiments/candidate15_detection_soft_evidence_result_20261014.json")
    _, checkpoint, all_records = checked_data()
    lookup = {r["stem"]: r for r in all_records}
    records = [lookup[s] for s in plan["image_selection"]["holdout"][:64]]
    frames = json.loads((prior / "frames.json").read_text())
    soft_frames = json.loads((soft / "frames.json").read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    paths = (REGISTRATION, "sdcl/evaluation/context_intervention.py",
             "scripts/evaluation/diagnose_context_intervention.py",
             "sdcl/training/class_margin_probe.py", "sdcl/training/marginal_class_intervention.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/rescue.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "candidate12_plan_sha256": file_hash(prior / "plan.json"),
        "candidate15_frames_sha256": file_hash(soft / "frames.json"),
        "detector_sha256": file_hash(checkpoint), "images": [r["stem"] for r in records],
    })
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    extractor = FrozenViews(device)
    kwargs = {**extractor.kwargs, "batch": 8}
    jobs, baseline_by_stem, selected_by_stem = [], {}, {}
    for record in records:
        stem = record["stem"]
        if (file_hash(record["image"]) != record["image_sha256"] or
                file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) != record["metadata_sha256"]):
            raise ValueError("Fixed context image/annotation changed.")
        with np.load(prior / frames[stem]["file"], allow_pickle=False) as c:
            baseline = c["baseline"].copy()
        with np.load(soft / soft_frames[stem]["file"], allow_pickle=False) as c:
            soft_score = c["soft_score"]
        image = cv2.imread(record["image"])
        if image is None or list(image.shape[1::-1]) != record["size"]:
            raise ValueError("Invalid context image.")
        baseline_by_stem[stem] = baseline
        selected = selected_rows(baseline, soft_score, record["size"])
        selected_by_stem[stem] = selected
        for row in selected:
            variants, box, _ = crop_variants(image, baseline[row])
            jobs.append({"stem": stem, "row": int(row), "images": variants, "box": box})
    if not jobs:
        extractor.close()
        raise ValueError("No prediction-only context samples.")
    forwards, missing, selected_total = 0, 0, len(jobs)
    probabilities = {s: np.zeros((len(rows), 3, 10), np.float32) for s, rows in selected_by_stem.items()}
    logits = {s: np.zeros((len(rows), 3, 10), np.float32) for s, rows in selected_by_stem.items()}
    positions = {s: {int(row): i for i, row in enumerate(rows)} for s, rows in selected_by_stem.items()}
    try:
        with torch.inference_mode():
            # Equal inputs must produce equal raw logits within one shared batch.
            sham = [jobs[0]["images"][0].copy(), jobs[0]["images"][0].copy()]
            extractor.model.predict(sham, **kwargs)
            if not torch.equal(extractor.raw["scores"][0], extractor.raw["scores"][1]):
                raise ValueError("Batch-position sham logits differ.")
            for start in range(0, len(jobs), 8):
                batch = jobs[start:start + 8]
                anchors, gathered, raw_gathered = None, [], []
                for arm in range(3):
                    extractor.model.predict([job["images"][arm] for job in batch], **kwargs)
                    decoded = extractor.decoded.permute(0, 2, 1).float().cpu().numpy()
                    raw_logits = extractor.raw["scores"].permute(0, 2, 1).float().cpu().numpy()
                    if anchors is None:
                        anchors = [anchor_choice(value, job["box"]) for value, job in zip(decoded, batch)]
                        missing += sum(i < 0 for i in anchors)
                    gathered.append(np.array([
                        value[index, 4:14] if index >= 0 else np.zeros(10, np.float32)
                        for value, index in zip(decoded, anchors)]))
                    raw_gathered.append(np.array([
                        value[index] if index >= 0 else np.zeros(10, np.float32)
                        for value, index in zip(raw_logits, anchors)]))
                    forwards += len(batch)
                for b, job in enumerate(batch):
                    probabilities[job["stem"]][positions[job["stem"]][job["row"]]] = np.stack(
                        [g[b] for g in gathered])
                    logits[job["stem"]][positions[job["stem"]][job["row"]]] = np.stack(
                        [g[b] for g in raw_gathered])
                if start == 0 or (start // 8 + 1) % 16 == 0:
                    print(f"Context crops: {min(start+8,len(jobs))}/{len(jobs)}; "
                          f"{perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    totals = {a: {g: {"gt": 0, "tp": 0} for g in GROUPS} for a in ARMS}
    fp, changes = dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0)
    owned_wrong, owned_correct = 0, 0
    fixes, losses = dict.fromkeys(ARMS[1:], 0), dict.fromkeys(ARMS[1:], 0)
    margin_samples = {a: [] for a in ARMS[1:]}
    frame_manifest = {}
    for record in records:
        stem = record["stem"]
        baseline, rows = baseline_by_stem[stem], selected_by_stem[stem]
        original = baseline[rows, 5].astype(int)
        predictions = {"uniform": baseline}
        for i, arm in enumerate(ARMS[1:]):
            values = baseline.copy()
            values[rows, 5] = class_choice(probabilities[stem][:, i], original)
            predictions[arm] = values
        gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
        classes = np.array([o["class_id"] for o in record["objects"]], int)
        iou = box_iou_matrix(gt, baseline[:, :4])
        geometry = baseline.copy()
        geometry[:, 5] = 0
        owned, _ = match_at_threshold(iou, np.zeros(len(classes), int), geometry, .001)
        owner = np.full(len(baseline), -1, int)
        owner[owned[owned >= 0]] = classes[owned >= 0]
        active = owner[rows] >= 0
        correct = owner[rows] == original
        owned_correct += int((active & correct).sum())
        owned_wrong += int((active & ~correct).sum())
        for arm in ARMS[1:]:
            new_correct = predictions[arm][rows, 5].astype(int) == owner[rows]
            fixes[arm] += int((active & ~correct & new_correct).sum())
            losses[arm] += int((active & correct & ~new_correct).sum())
        native_active = active & (probabilities[stem][:, 0].max(1) > 0)
        for i, arm in enumerate(ARMS[1:]):
            values = logits[stem][native_active, i].copy()
            labels = owner[rows][native_active]
            if len(labels):
                true = values[np.arange(len(labels)), labels].copy()
                values[np.arange(len(labels)), labels] = -np.inf
                margin_samples[arm].append((true - values.max(1), correct[native_active]))
        areas = box_areas(gt) * (640 / max(record["size"])) ** 2
        for arm, values in predictions.items():
            matched, _ = match_at_threshold(iou, classes, values, .25)
            fp[arm] += int((values[:, 4] >= .25).sum() - (matched >= 0).sum())
            changes[arm] += int((values[:, 5] != baseline[:, 5]).sum())
            for group in GROUPS:
                include = group_mask(areas, group)
                totals[arm][group]["gt"] += int(include.sum())
                totals[arm][group]["tp"] += int((include & (matched >= 0)).sum())
        target = output / "frames" / f"{stem}.npz"
        np.savez_compressed(target, baseline=baseline, selected_rows=rows, probability=probabilities[stem],
                            logits=logits[stem], owner=owner,
                            **{a: p[:, 5].astype(int) for a, p in predictions.items()})
        frame_manifest[stem] = {"file": target.relative_to(output).as_posix(), "sha256": file_hash(target)}
    metrics = {
        a: {**{g + "_recall_percent": v["tp"] / max(v["gt"], 1) * 100
               for g, v in groups.items()}, "all_fppi": fp[a] / len(records),
            "changed_labels": changes[a], "gt_tp": groups}
        for a, groups in totals.items()
    }
    verdict = decide(metrics, owned_wrong, registration["policy"])
    margins = {}
    for arm, chunks in margin_samples.items():
        if chunks:
            margin = np.concatenate([c[0] for c in chunks])
            correct = np.concatenate([c[1] for c in chunks])
            margins[arm] = {
                "owned_mean_true_class_logit_margin": float(margin.mean()),
                "owned_original_wrong_mean_margin": float(margin[~correct].mean()) if (~correct).any() else None,
                "owned_original_correct_mean_margin": float(margin[correct].mean()) if correct.any() else None,
            }
    save_json(output / "frames.json", frame_manifest)
    save_json(output / "metrics.json", metrics)
    save_json(output / "logit_margins.json", margins)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": len(records), "selected_prediction_rows": selected_total,
        "missing_original_crop_anchor": missing, "owned_original_wrong": owned_wrong,
        "owned_original_correct": owned_correct, "owned_fixes": fixes, "owned_losses": losses,
        "patch_forwards_excluding_warmup_sham": forwards,
        "sham_equal_input_raw_logits_exact": True, "preserved_region_pixels_exact": True,
        "no_training": True, "no_testdev": True, "no_threshold_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
