"""Read-only frozen class-information audit on historical TAL feature samples."""

import hashlib
import json
from time import perf_counter
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.class_choice_signal import weighted_auc
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate14_soft_class_information_registration_20261014.json"
OUTPUT = "output/analysis/soft-class-information-fixed-20261014"


def class_statistics(logits, targets, tiny):
    logits, targets = np.asarray(logits), np.asarray(targets)
    if (logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 10
            or np.asarray(tiny).shape != (len(logits),)
            or not np.isfinite(logits).all() or not np.isfinite(targets).all()
            or (targets < 0).any()):
        raise ValueError("Invalid cached class-information arrays.")
    quality = targets.sum(1)
    positive = quality > 0
    if (np.count_nonzero(targets[positive], axis=1) != 1).any():
        raise ValueError("Expected one TAL target class per positive row.")
    # Positive is equivalent to maximum sigmoid score >= .25, without exp overflow.
    active = positive & np.asarray(tiny, bool) & (logits.max(1) >= np.log(.25 / .75))
    value, label, quality = logits[active], targets[active].argmax(1), quality[active]
    order = np.argsort(-value, axis=1, kind="stable")
    rank = (order == label[:, None]).argmax(1) + 1
    correct, wrong = rank == 1, rank > 1
    gap = value[np.arange(len(value)), order[:, 0]] - value[np.arange(len(value)), order[:, 1]]
    return {
        "tiny_positive_rows_at_score025": int(len(value)),
        "correct_rows": int(correct.sum()), "wrong_rows": int(wrong.sum()),
        "wrong_target_rank_histogram": [int((rank[wrong] == k).sum()) for k in range(2, 11)],
        "wrong_top3_coverage": float((rank[wrong] <= 3).mean()) if wrong.any() else None,
        "wrong_top3_quality_weighted_coverage": float(
            np.average(rank[wrong] <= 3, weights=quality[wrong])) if wrong.any() else None,
        "top2_gap_correctness_auc": weighted_auc(correct, gap, np.ones(len(gap))),
        "top2_gap_quality_weighted_correctness_auc": weighted_auc(correct, gap, quality),
    }


def decide(summary, policy):
    held = summary["holdout"]
    coverage, auc = held["wrong_top3_coverage"], held["top2_gap_correctness_auc"]
    checks = {
        "wrong_sample_count": held["wrong_rows"] >= policy["holdout_tiny_wrong_count_min"],
        "top3_information": coverage is not None and coverage >= policy["holdout_tiny_wrong_top3_coverage_min"],
        "gap_discrimination": auc is not None and auc >= policy["holdout_tiny_gap_auc_min"],
    }
    return {
        "decision": "RAW_POSTERIOR_AUDIT_ONLY" if all(checks.values()) else "STOP_THIS_SOFT_CLASS_SIGNAL",
        "checks": checks,
        "scope": "Biased sampled TAL-anchor information, not detection correction, AP, or originality.",
        "prior_class_margin_stop_unchanged": True,
    }


def evaluate(output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    cache = project_path(registration["cache"])
    if file_hash(cache / "manifest.json") != registration["manifest_sha256"]:
        raise ValueError("Historical feature-cache manifest changed.")
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["complete"] or len(manifest["chunks"]) != 640:
        raise ValueError("Incomplete fixed feature cache.")
    if file_hash(cache / "source.zip") != manifest["source_archive_sha256"]:
        raise ValueError("Historical source archive changed.")
    archived = manifest["settings"]["source_hashes"]
    with zipfile.ZipFile(cache / "source.zip") as source:
        if set(source.namelist()) != set(archived) or any(
                hashlib.sha256(source.read(name)).hexdigest() != digest
                for name, digest in archived.items()):
            raise ValueError("Archived cache sources do not match their recorded hashes.")
    if file_hash(cache / "initial_projection.npz") != manifest["initial_projection_sha256"]:
        raise ValueError("Initial frozen projection changed.")
    with np.load(cache / "initial_projection.npz", allow_pickle=False) as projection:
        weight, bias = projection["weight"], projection["bias"]
    if weight.shape != (3, 10, 128) or bias.shape != (3, 10):
        raise ValueError("Unexpected frozen class projection.")
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/evaluation/soft_class_information.py",
             "scripts/evaluation/diagnose_soft_class_information.py",
             "sdcl/evaluation/class_choice_signal.py", "sdcl/config.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/rescue.py",
             "sdcl/training/rescue.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "initial_projection_sha256": manifest["initial_projection_sha256"],
        "historical_source_archive_verified": True,
        "live_historical_evaluator_executed": False,
    })
    torch.set_num_threads(1)
    groups = {False: [], True: []}
    scenes = {False: set(), True: set()}
    images = {False: 0, True: 0}
    with torch.inference_mode():
        for stem, item in manifest["chunks"].items():
            path = cache / item["file"]
            if file_hash(path) != item["sha256"]:
                raise ValueError("Historical cache chunk changed.")
            with np.load(path, allow_pickle=False) as chunk:
                features, levels = chunk["features"].astype(np.float32), chunk["levels"]
                targets, tiny = chunk["targets"], chunk["tiny"]
            if features.shape != (item["rows"], 128) or not np.isin(levels, [0, 1, 2]).all():
                raise ValueError("Unexpected cache features or pyramid level.")
            logits = np.empty((len(features), 10), np.float32)
            for level in range(3):
                mask = levels == level
                logits[mask] = F.linear(torch.from_numpy(features[mask]),
                                       torch.from_numpy(weight[level]),
                                       torch.from_numpy(bias[level])).numpy()
            group = item["monitor"]
            groups[group].append((logits, targets, tiny))
            images[group] += 1
            scenes[group].add(stem.split("_")[0])
    if images != {False: 512, True: 128} or scenes[False] & scenes[True]:
        raise ValueError("Fixed scene-separated cache split changed.")
    summary = {
        "holdout" if group else "train": class_statistics(
            *(np.concatenate([row[i] for row in rows]) for i in range(3)))
        for group, rows in groups.items()
    }
    verdict = decide(summary, registration["policy"])
    save_json(output / "summary.json", summary)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "all_640_cache_chunks_verified": True,
        "fitting_images": 512, "head_holdout_images": 128, "scene_overlap": 0,
        "no_training": True, "no_detector_forward": True, "no_policy_execution": True,
        "approximate_fp32_reconstruction_only": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "summary": summary}
