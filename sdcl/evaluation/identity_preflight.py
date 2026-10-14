"""Cached, annotation-assisted screen for harmful cross-view instance drift."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.candidate_series import CACHE, checked_cache
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold, write_csv,
)
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate8_identity_preflight_registration_20261014.json"
OUTPUT = "output/analysis/identity-preflight-fixed-20261014"
THRESHOLDS = np.linspace(.5, .95, 10)
GROUPS = ("all", "tiny", "small_all")
ARMS = ("uniform", "regret")


def classify_drift(old, current, owner, classes):
    """GT-by-owned-row IoUs; require a change of nearest same-class reference."""
    if old.shape != current.shape or old.shape[1] != len(owner):
        raise ValueError("Misaligned ownership/IoUs.")
    row = np.arange(len(owner))
    old_owned, new_owned = old[owner, row], current[owner, row]
    competitors = (classes[:, None] == classes[owner][None, :])
    competitors[owner, row] = False
    old_other = np.where(competitors, old, -1.).max(0, initial=-1.)
    new_other = np.where(competitors, current, -1.).max(0, initial=-1.)
    harmed = new_owned < old_owned - .01
    suspect = harmed & (new_other > new_owned + .01) & (new_other >= .5)
    strict = suspect & (old_other <= old_owned + 1e-6)
    return old_owned, new_owned, harmed, suspect, strict


def threshold_recovery(old, current, mask):
    recovered = ((old[:, None] >= THRESHOLDS) & (current[:, None] < THRESHOLDS)).sum(1)
    return float(recovered[mask].sum() / len(THRESHOLDS))


def decide(summary, policy):
    primary = summary["uniform"]
    space = max(primary[g]["strict_drift_fixed_owner_recovery_pp"]
                for g in ("all", "small_all"))
    checks = {
        "strict_drift_count": primary["all"]["strict_drift_count"] >=
        policy["primary_strict_drift_count_min"],
        "fixed_owner_threshold_recovery": space >=
        policy["primary_fixed_owner_threshold_recovery_all_or_small_all_min_pp"],
    }
    return {
        "decision": "PILOT_WORTH_TESTING" if all(checks.values()) else
        "STOP_IDENTITY_REPAIR_RECIPE",
        "checks": checks, "primary_arm": "uniform",
        "primary_space_pp": space, "policy": policy,
        "scope": "Diagnosis only; no official AP or deployable gain.",
    }


def evaluate(cache=CACHE, output=OUTPUT):
    started = perf_counter()
    registration_path = project_path(REGISTRATION)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration["primary_arm"] != "uniform" or registration["arms"] != list(ARMS):
        raise ValueError("Registered primary comparison changed.")
    torch.set_num_threads(1)
    cache, cache_plan, records = checked_cache(cache)
    models, _, _, heads = frozen_heads()
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_paths = ("sdcl/evaluation/identity_preflight.py",
                    "scripts/evaluation/diagnose_identity_preflight.py", REGISTRATION)
    hashes = {p: file_hash(project_path(p)) for p in source_paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "registration_sha256": file_hash(registration_path),
        "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "source_hashes": hashes, "source_archive_sha256": file_hash(output / "source.zip"),
        "historical_dependencies": cache_plan["source_hashes"], "heads": heads,
    })
    chunks = json.loads((cache / "chunks.json").read_text())
    totals = {
        arm: {g: dict.fromkeys(("gt", "owned", "changed_owned", "harmed_owned",
                               "suspect_count", "strict_drift_count",
                               "strict_recovered_threshold_hits",
                               "all_harm_recovered_threshold_hits"), 0)
              for g in GROUPS} for arm in ARMS
    }
    image_rows, examples = [], []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            with np.load(cache / chunks[record["stem"]]["file"], allow_pickle=False) as chunk:
                calibration, boxes = chunk["calibration"], chunk["boxes"]
                normalized, valid = chunk["normalized"], chunk["valid"]
            gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in record["objects"]])
            ious = box_iou_matrix(gt, calibration[:, :4])
            matched, _ = match_at_threshold(ious, classes, calibration, .001)
            owner = np.flatnonzero(matched >= 0)
            dt_rows = matched[owner]
            areas = box_areas(gt) * (640. / max(record["size"])) ** 2
            for arm in ARMS:
                selected = select_actions(models[arm](torch.from_numpy(normalized)).numpy(),
                                          valid, "absolute")
                prediction = apply_actions(calibration, boxes, selected)
                current = box_iou_matrix(gt, prediction[dt_rows, :4])
                old, new, harm, suspect, strict = classify_drift(
                    ious[:, dt_rows], current, owner, classes)
                changed = np.any(prediction[dt_rows, :4] != calibration[dt_rows, :4], axis=1)
                for group in GROUPS:
                    gt_mask = group_mask(areas, group)
                    mask = gt_mask[owner]
                    values = {
                        "gt": int(gt_mask.sum()), "owned": int(mask.sum()),
                        "changed_owned": int((changed & mask).sum()),
                        "harmed_owned": int((harm & mask).sum()),
                        "suspect_count": int((suspect & mask).sum()),
                        "strict_drift_count": int((strict & mask).sum()),
                        "strict_recovered_threshold_hits":
                            threshold_recovery(old, new, strict & mask),
                        "all_harm_recovered_threshold_hits":
                            threshold_recovery(old, new, harm & mask),
                    }
                    image_rows.append({"image": record["stem"], "arm": arm,
                                       "group": group, **values})
                    for key, value in values.items():
                        totals[arm][group][key] += value
                for position in np.flatnonzero(strict):
                    row, owned = int(dt_rows[position]), int(owner[position])
                    examples.append({
                        "image": record["stem"], "arm": arm, "row": row,
                        "owner_gt": owned, "class_id": int(classes[owned]),
                        "action": int(selected[row]), "old_owned_iou": float(old[position]),
                        "new_owned_iou": float(new[position]),
                        "original_box": calibration[row, :4].tolist(),
                        "new_box": prediction[row, :4].tolist(),
                    })
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"Identity diagnosis: {index}/{len(records)}, "
                      f"{perf_counter() - started:.1f}s.", flush=True)
    for groups in totals.values():
        for values in groups.values():
            denominator = max(values["gt"], 1)
            values["strict_drift_fixed_owner_recovery_pp"] = (
                values["strict_recovered_threshold_hits"] / denominator * 100)
            values["all_harm_fixed_owner_recovery_pp"] = (
                values["all_harm_recovered_threshold_hits"] / denominator * 100)
    verdict = decide(totals, registration["policy"])
    verification = {
        "images": len(records), "complete": len(records) == 1610,
        "registered_source_unchanged": all(file_hash(project_path(p)) == h
                                           for p, h in hashes.items()),
        "no_detector_inference": True, "no_training": True,
        "no_image_decoding": True, "no_official_evaluation": True,
    }
    if not verification["complete"] or not verification["registered_source_unchanged"]:
        raise ValueError("Incomplete or changed diagnostic.")
    save_json(output / "summary.json", totals)
    save_json(output / "decision.json", verdict)
    save_json(output / "examples.json", examples)
    save_json(output / "verification.json", verification)
    write_csv(output / "per_image.csv", image_rows)
    return {"output": str(output), "decision": verdict, "summary": totals,
            "elapsed_seconds": perf_counter() - started}
