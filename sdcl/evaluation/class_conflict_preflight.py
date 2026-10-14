"""No-training cross-view class-choice headroom against an ordinary vote."""

import hashlib
import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.candidate_series import CACHE, checked_cache
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold, write_csv,
)
from sdcl.postprocess.class_conflict import apply_labels, class_evidence, vote_labels
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate11_class_conflict_registration_20261014.json"
OUTPUT = "output/analysis/class-conflict-headroom-fixed-20261014"
ARMS = ("uniform", "weighted_vote", "class_oracle")
GROUPS = ("all", "tiny", "small_all")


def owned_oracle(record, candidates, valid):
    gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
    cls = np.array([o["class_id"] for o in record["objects"]], int)
    geometric = candidates.copy()
    geometric[:, 5] = 0
    matched, _ = match_at_threshold(box_iou_matrix(gt, candidates[:, :4]),
                                   np.zeros(len(cls), int), geometric, .001)
    labels = candidates[:, 5].astype(int).copy()
    count = 0
    for index in np.flatnonzero(matched >= 0):
        row = matched[index]
        if valid[row, cls[index]] and labels[row] != cls[index]:
            labels[row] = cls[index]
            count += 1
    return labels, count


def decide(summary, corrections, policy):
    point = summary["0.25"]
    gt = point["uniform"]["small_all"]["gt"]
    method = point["class_oracle"]["small_all"]["tp"]
    original = point["uniform"]["small_all"]["tp"]
    vote = point["weighted_vote"]["small_all"]["tp"]
    gains = {
        "small_all_oracle_vs_uniform_recall_pp": (method - original) / max(gt, 1) * 100,
        "small_all_oracle_vs_weighted_vote_recall_pp": (method - vote) / max(gt, 1) * 100,
        "available_owned_wrong_class_actions": corrections,
    }
    checks = {
        "wrong_class_actions": corrections >= policy["available_owned_wrong_class_actions_min"],
        "oracle_vs_uniform": gains["small_all_oracle_vs_uniform_recall_pp"] >=
        policy["small_all_oracle_vs_uniform_recall_min_pp"],
        "oracle_vs_vote": gains["small_all_oracle_vs_weighted_vote_recall_pp"] >=
        policy["small_all_oracle_vs_weighted_vote_recall_min_pp"],
    }
    return {
        "decision": "SMALL_CLASS_CHOICE_PILOT_WORTH_TESTING" if all(checks.values()) else
        "STOP_THIS_CLASS_ACTION_MENU",
        "checks": checks, "values": gains, "policy": policy,
        "scope": "Sampled annotation-assisted headroom only; not official AP or novelty.",
    }


def evaluate(cache=CACHE, output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    cache, cache_plan, all_records = checked_cache(cache)
    records = sorted(all_records, key=lambda r: hashlib.sha256(
        ("class-conflict-preflight-v1:" + r["stem"]).encode()).digest())[:256]
    expected = {r["stem"]: r for r in records}
    torch.set_num_threads(1)
    models, _, _, heads = frozen_heads()
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/class_conflict.py",
             "sdcl/evaluation/class_conflict_preflight.py",
             "scripts/evaluation/diagnose_class_conflict.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "heads": heads, "images": list(expected),
        "historical_dependencies": cache_plan["source_hashes"],
    })
    totals = {str(p): {a: {g: {"gt": 0, "tp": 0} for g in GROUPS}
                      for a in ARMS} for p in registration["operating_points"]}
    fp = {str(p): {a: 0 for a in ARMS} for p in registration["operating_points"]}
    chunks = json.loads((cache / "chunks.json").read_text())
    corrections, eligible, conflicts, seen = 0, 0, 0, set()
    rows, transition = [], np.zeros((10, 10), int)
    with torch.inference_mode(), (cache / "raw_views.jsonl").open() as stream:
        for line in stream:
            raw = json.loads(line)
            stem = raw["image"]
            if stem not in expected:
                continue
            if stem in seen:
                raise ValueError("Duplicate sampled image.")
            seen.add(stem)
            record = expected[stem]
            with np.load(cache / chunks[stem]["file"], allow_pickle=False) as chunk:
                selected = select_actions(models["uniform"](torch.from_numpy(
                    chunk["normalized"])).numpy(), chunk["valid"], "absolute")
                baseline = apply_actions(chunk["calibration"], chunk["boxes"], selected)
            valid, score, _, _ = class_evidence(
                baseline, raw["views"], raw["windows"], record["size"])
            eligible += len(baseline)
            conflicts += int((valid.sum(1) > 1).sum())
            metadata = json.loads(project_path(f"data/visdrone/metadata/test/{stem}.json").read_text())
            ignores = np.array(metadata["ignore_boxes_xyxy"]).reshape(-1, 4)
            keep = ~prediction_ignore_mask(torch.from_numpy(baseline[:, :4]),
                torch.from_numpy(ignores), .5).numpy()
            # This annotation mask belongs only to evaluation; evidence above remains prediction-only.
            baseline, valid, score = baseline[keep], valid[keep], score[keep]
            labels, count = owned_oracle(record, baseline, valid)
            corrections += count
            changed = labels != baseline[:, 5]
            np.add.at(transition, (baseline[changed, 5].astype(int), labels[changed]), 1)
            predictions = {
                "uniform": baseline,
                "weighted_vote": apply_labels(baseline, vote_labels(baseline, valid, score)),
                "class_oracle": apply_labels(baseline, labels),
            }
            gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in record["objects"]], int)
            areas = box_areas(gt) * (640 / max(record["size"])) ** 2
            ious = box_iou_matrix(gt, baseline[:, :4])
            for p in registration["operating_points"]:
                for arm, values in predictions.items():
                    matches, _ = match_at_threshold(ious, classes, values, p)
                    false = int((values[:, 4] >= p).sum() - (matches >= 0).sum())
                    fp[str(p)][arm] += false
                    for group in GROUPS:
                        mask = group_mask(areas, group)
                        counts = {"gt": int(mask.sum()), "tp": int(((matches >= 0) & mask).sum())}
                        rows.append({"image": stem, "point": p, "arm": arm, "group": group,
                                     **counts, "image_fp": false})
                        for key, value in counts.items():
                            totals[str(p)][arm][group][key] += value
            if len(seen) == 1 or len(seen) % 64 == 0:
                print(f"Class disagreement: {len(seen)}/{len(records)}, "
                      f"{perf_counter() - started:.1f}s.", flush=True)
    if seen != set(expected):
        raise ValueError("Incomplete sampled cache.")
    verdict = decide(totals, corrections, registration["policy"])
    for points in totals.values():
        for groups in points.values():
            for values in groups.values():
                values["recall_percent"] = values["tp"] / max(values["gt"], 1) * 100
    save_json(output / "summary.json", totals)
    save_json(output / "false_positives.json", fp)
    save_json(output / "decision.json", verdict)
    save_json(output / "class_transitions.json", {
        "rows_original_class_columns_owned_gt_class": transition.tolist(),
        "available_owned_wrong_class_actions": corrections,
        "prediction_rows_before_diagnostic_ignore": eligible,
        "cross_class_action_rows_before_diagnostic_ignore": conflicts,
    })
    write_csv(output / "per_image.csv", rows)
    save_json(output / "verification.json", {
        "images": len(seen), "no_training": True, "no_detector_inference": True,
        "no_image_decoding": True, "no_official_evaluation": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "total_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "summary": totals,
            "false_positives": fp, "elapsed_seconds": perf_counter() - started}
