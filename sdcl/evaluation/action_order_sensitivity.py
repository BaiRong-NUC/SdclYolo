"""Finite annotation-corner probe on fixed, cached coordinate action menus."""

from itertools import combinations, product
import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold,
)
from sdcl.postprocess.localization import (
    action_menu, apply_actions, calibrated_candidates, frozen_control, load_calibrator,
    normalized_actions, select_actions,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate18_action_order_sensitivity_registration_20261014.json"
OUTPUT = "output/analysis/action-order-sensitivity-fixed-20261014"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
GROUPS = ("all", "tiny", "small", "small_all", "medium", "large")
THRESHOLDS = np.linspace(.5, .95, 10)
GAP = .01


def perturbed_boxes(gt, size):
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    offsets = np.array([(0, 0, 0, 0), *product((-1., 1.), repeat=4)])
    boxes = gt[:, None, :] + offsets[None]
    boxes = np.clip(boxes, 0, np.array([*size, *size]))
    valid = (boxes[:, :, 2:] > boxes[:, :, :2]).all(axis=(1, 2))
    return boxes, valid


def aligned_ious(actions, variants):
    low = np.maximum(actions[:, None, :, :2], variants[:, :, None, :2])
    high = np.minimum(actions[:, None, :, 2:], variants[:, :, None, 2:])
    intersection = np.maximum(high - low, 0).prod(-1)
    action_area = np.maximum(actions[:, :, 2:] - actions[:, :, :2], 0).prod(-1)
    gt_area = np.maximum(variants[:, :, 2:] - variants[:, :, :2], 0).prod(-1)
    return intersection / np.maximum(
        action_area[:, None] + gt_area[:, :, None] - intersection, 1e-12)


def summarize(ious, valid, uniform, areas, eligible):
    summaries = {}
    row = np.arange(len(ious))
    reference = ious[row, :, uniform]
    delta = ious - reference[:, :, None]
    beneficial = valid & (delta[:, 0] > GAP)
    reversal = beneficial & (delta.min(1) < -GAP)
    stable = valid & (delta.min(1) > GAP)
    stable_choice = np.argmax(np.where(stable, ious[:, 0], -1), axis=1)
    stable_choice = np.where(stable.any(1), stable_choice, uniform)
    nominal_choice = np.argmax(np.where(valid, ious[:, 0], -1), axis=1)
    nominal = ious[:, 0]
    values = {
        "uniform": nominal[row, uniform],
        "nominal_oracle": nominal[row, nominal_choice],
        "stable_oracle": nominal[row, stable_choice],
    }
    pair_gap, pair_valid = [], []
    for a, b in combinations(range(ious.shape[2]), 2):
        difference = ious[:, :, a] - ious[:, :, b]
        pair_gap.append(difference * np.sign(difference[:, :1]))
        pair_valid.append(valid[:, a] & valid[:, b] & (np.abs(difference[:, 0]) > GAP))
    pair_gap = np.stack(pair_gap, axis=2)
    pair_valid = np.stack(pair_valid, axis=1)
    for group in GROUPS:
        mask = eligible & group_mask(areas, group)
        action_count = int(beneficial[mask].sum())
        hard_count = int(reversal[mask].sum())
        pairs = int(pair_valid[mask].sum())
        result = {
            "eligible_owned_anchors": int(mask.sum()),
            "nominal_beneficial_actions": action_count,
            "hard_reversed_beneficial_actions": hard_count,
            "hard_reversal_percent": hard_count / max(action_count, 1) * 100,
            "stable_improvement_anchors": int((mask & stable.any(1)).sum()),
            "nominal_ordered_pairs": pairs,
            "sign_reversed_pairs": int((pair_valid[mask] & (pair_gap[mask].min(1) < 0)).sum()),
            "hard_reversed_pairs": int((pair_valid[mask] & (pair_gap[mask].min(1) < -GAP)).sum()),
        }
        for arm, chosen in values.items():
            result[arm + "_owned_mean_iou_percent"] = float(
                chosen[mask].mean() * 100) if mask.any() else None
            result[arm + "_owned_threshold_percent"] = float(
                (chosen[mask, None] >= THRESHOLDS).mean() * 100) if mask.any() else None
        result["stable_oracle_threshold_gain_pp"] = (
            result["stable_oracle_owned_threshold_percent"] -
            result["uniform_owned_threshold_percent"]) if mask.any() else None
        summaries[group] = result
    return summaries


def decide(metrics, policy):
    tiny, small = metrics["tiny"], metrics["small_all"]
    checks = {
        "enough_beneficial_tiny_actions": tiny["nominal_beneficial_actions"] >=
        policy["holdout_tiny_nominal_beneficial_actions_min"],
        "meaningful_order_reversals": tiny["hard_reversal_percent"] >=
        policy["holdout_tiny_hard_reversal_percent_min"],
        "remaining_stable_headroom": small["stable_oracle_threshold_gain_pp"] is not None and
        small["stable_oracle_threshold_gain_pp"] >=
        policy["holdout_small_all_stable_oracle_threshold_gain_min_pp"],
    }
    return {
        "decision": "MATCHED_STABLE_ORDER_HEAD_PILOT_ALLOWED" if all(checks.values()) else
        "STOP_THIS_ANNOTATION_ORDER_RECIPE",
        "checks": checks,
        "values": {
            "tiny_nominal_beneficial_actions": tiny["nominal_beneficial_actions"],
            "tiny_hard_reversal_percent": tiny["hard_reversal_percent"],
            "small_all_stable_oracle_threshold_gain_pp": small["stable_oracle_threshold_gain_pp"],
        },
        "scope": "Annotation-assisted sensitivity/headroom only, not a deployable improvement or novelty.",
    }


def evaluate(output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior = project_path(PRIOR)
    _, prior_plan = verify_record(
        prior, "docs/experiments/candidate12_class_choice_result_20261014.json")
    selected = {g: prior_plan["image_selection"][g][:registration["sample"][g + "_images"]]
                for g in ("train", "holdout")}
    if {s.split("_")[0] for s in selected["train"]} & {
            s.split("_")[0] for s in selected["holdout"]}:
        raise ValueError("Head split scene overlap.")
    records = {r["stem"]: r for rows in prior_plan["records"].values() for r in rows}
    raw = {r["image"]: r for r in
           (json.loads(line) for line in (prior / "raw_views.jsonl").read_text().splitlines())}
    historical_frames = json.loads((prior / "frames.json").read_text())
    heads, mean, std, provenance = frozen_heads()
    if provenance != prior_plan["heads"]:
        raise ValueError("Frozen coordinate heads differ from candidate12.")
    calibrator = load_calibrator(frozen_control())
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    paths = (REGISTRATION, "sdcl/evaluation/action_order_sensitivity.py",
             "scripts/evaluation/diagnose_action_order_sensitivity.py",
             "sdcl/postprocess/localization.py", "sdcl/evaluation/size_analysis.py",
             "sdcl/data/geometry.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate12_class_choice_result_20261014.json")),
        "prior_plan_sha256": file_hash(prior / "plan.json"),
        "raw_views_sha256": file_hash(prior / "raw_views.jsonl"),
        "image_selection": selected, "heads": provenance,
    })
    frames, arrays, excluded = {}, {}, {}
    with torch.inference_mode():
        for group, stems in selected.items():
            collected, degenerate = [], 0
            for index, stem in enumerate(stems, 1):
                metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                if file_hash(metadata_path) != records[stem]["metadata_sha256"]:
                    raise ValueError("Frozen metadata differs.")
                meta = json.loads(metadata_path.read_text())
                size, item = meta["original_size"], raw[stem]
                views = [np.asarray(v, dtype=float).reshape(-1, 6) for v in item["views"]]
                c, origins, scores, base = calibrated_candidates(
                    views, item["windows"], size, [], calibrator)
                boxes, valid, features = action_menu(
                    c, origins, scores, base, views, size, [])
                features = normalized_actions(features, mean, std)
                uniform = select_actions(
                    heads["uniform"](torch.from_numpy(features)).numpy(), valid, "absolute")
                baseline = apply_actions(c, boxes, uniform)
                ignored = prediction_ignore_mask(torch.from_numpy(baseline[:, :4]),
                    torch.tensor(meta["ignore_boxes_xyxy"], dtype=torch.float64).reshape(-1, 4))
                keep = ~ignored.numpy()
                baseline, c, boxes, valid, features, uniform = (
                    a[keep] for a in (baseline, c, boxes, valid, features, uniform))
                cached = prior / historical_frames[stem]["file"]
                if file_hash(cached) != historical_frames[stem]["sha256"]:
                    raise ValueError("Prior frozen frame changed.")
                with np.load(cached, allow_pickle=False) as saved:
                    if not np.array_equal(baseline, saved["baseline"]):
                        raise ValueError("Uniform raw replay differs; stop without changing the control.")
                objects = meta["valid_objects"]
                gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
                classes = np.asarray([o["class_id"] for o in objects], int)
                matched, _ = match_at_threshold(box_iou_matrix(gt, c[:, :4]), classes, c, .001)
                owner = np.full(len(c), -1, int)
                owner[matched[matched >= 0]] = np.flatnonzero(matched >= 0)
                rows = np.flatnonzero(owner >= 0)
                variants, complete = perturbed_boxes(gt[owner[rows]], size)
                ious = np.zeros((len(c), 17, 6), dtype=np.float32)
                ious[rows] = aligned_ious(boxes[rows], variants)
                areas = np.zeros(len(c))
                areas[rows] = box_areas(gt[owner[rows]]) * (640 / max(size)) ** 2
                eligible = np.zeros(len(c), bool)
                eligible[rows] = complete & (c[rows, 4] >= .05) & (valid[rows].sum(1) >= 2)
                degenerate += int((~complete).sum())
                target = output / "frames" / f"{stem}.npz"
                np.savez_compressed(target, features=features, candidates=c, boxes=boxes,
                                    valid=valid, uniform=uniform, ious=ious, areas=areas,
                                    eligible=eligible, owner=owner, baseline=baseline)
                frames[stem] = {"file": target.relative_to(output).as_posix(),
                                "sha256": file_hash(target), "group": group,
                                "metadata_sha256": records[stem]["metadata_sha256"]}
                collected.append((ious, valid, uniform, areas, eligible))
                if index == 1 or index % 32 == 0 or index == len(stems):
                    print(f"Coordinate-order probe {group}: {index}/{len(stems)}, "
                          f"{perf_counter()-started:.1f}s", flush=True)
            arrays[group] = tuple(np.concatenate([r[k] for r in collected]) for k in range(5))
            excluded[group] = degenerate
    metrics = {g: summarize(*a) for g, a in arrays.items()}
    verdict = decide(metrics["holdout"], registration["policy"])
    save_json(output / "frames.json", frames)
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": {g: len(s) for g, s in selected.items()},
        "all_uniform_predictions_exact_to_candidate12": True,
        "invalid_perturbation_anchor_count": excluded, "detector_forwards": 0,
        "training_steps": 0, "no_testdev": True, "no_radius_or_threshold_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "holdout": metrics["holdout"]}
