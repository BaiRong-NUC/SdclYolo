"""Threshold-free audit of cached class-head discrimination, never efficacy."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.class_choice import CONTEXT_DIM, ClassChoiceScorer
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate13_class_signal_registration_20261014.json"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
OUTPUT = "output/analysis/class-choice-signal-fixed-20261014"
ARMS = ("evidence_only", "observation_context")


def weighted_auc(labels, score, weight):
    labels = np.asarray(labels, bool)
    score, weight = np.asarray(score, float), np.asarray(weight, float)
    if not len(labels):
        return None
    if (score.shape != labels.shape or weight.shape != labels.shape or
            not np.isfinite(score).all() or not np.isfinite(weight).all() or np.any(weight <= 0)):
        raise ValueError("Invalid discrimination vectors.")
    positive, negative = weight[labels].sum(), weight[~labels].sum()
    if positive <= 0 or negative <= 0:
        return None
    order = np.argsort(score, kind="stable")
    labels, score, weight = labels[order], score[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    pos = np.add.reduceat(weight * labels, starts)
    neg = np.add.reduceat(weight * ~labels, starts)
    preceding_neg = np.cumsum(neg) - neg
    return float((pos * (preceding_neg + .5 * neg)).sum() / (positive * negative))


def decide(summary, policy):
    held = summary["holdout"]
    ctx, plain = held["observation_context"], held["evidence_only"]
    auc, matched = ctx["weighted_auc"], plain["weighted_auc"]
    checks = {
        "positive_count": ctx["positive_count"] >= policy["holdout_positive_count_min"],
        "context_discrimination": auc is not None and auc >= policy["context_holdout_weighted_auc_min"],
        "extra_context_signal": auc is not None and matched is not None and
        auc - matched >= policy["context_holdout_weighted_auc_vs_evidence_only_min"],
    }
    return {
        "decision": "CONTEXT_DISCRIMINATION_HYPOTHESIS_ONLY" if all(checks.values()) else
        "NO_EXTRA_CONTEXT_DISCRIMINATION_SIGNAL",
        "checks": checks, "context_minus_evidence_weighted_auc":
            None if auc is None or matched is None else auc - matched,
        "candidate12_decision": "STOP_THIS_CONTEXT_CLASS_CHOICE_PILOT; unchanged.",
        "scope": "Diagnostic proxy ranking, not a deployment or official AP result.",
    }


def evaluate(output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior = project_path(PRIOR)
    result = json.loads(project_path(
        "docs/experiments/candidate12_class_choice_result_20261014.json").read_text())
    for name, digest in result["artifacts_sha256"].items():
        if file_hash(prior / name) != digest:
            raise ValueError("Failed pilot inputs/results changed.")
    plan = json.loads((prior / "plan.json").read_text())
    verify_dependencies(plan["source_hashes"], prior / "source.zip")
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/evaluation/class_choice_signal.py",
             "scripts/evaluation/diagnose_class_choice_signal.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_plan_sha256": file_hash(prior / "plan.json"),
        "checkpoint_sha256": file_hash(prior / "seed0/last.pt"),
    })
    torch.set_num_threads(1)
    saved = torch.load(prior / "seed0/last.pt", map_location="cpu", weights_only=False)
    models = {a: ClassChoiceScorer().eval() for a in ARMS}
    for arm in ARMS:
        models[arm].load_state_dict(saved["models"][arm])
    mean, std = saved["mean"].numpy(), saved["std"].numpy()
    frames = json.loads((prior / "frames.json").read_text())
    groups = {g: {a: [] for a in ARMS} for g in ("train", "holdout")}
    with torch.inference_mode():
        for frame in frames.values():
            with np.load(prior / frame["file"], allow_pickle=False) as c:
                mask = (c["valid"].sum(1) > 1) & (c["baseline"][:, 4] >= .25)
                if not mask.any():
                    continue
                raw, valid = c["features"][mask], c["valid"][mask]
                baseline, target = c["baseline"][mask], c["targets"][mask]
            original = baseline[:, 5].astype(int)
            positive = target != original
            inputs = torch.from_numpy(np.clip((raw - mean) / std, -6, 6).astype(np.float32))
            rows = np.arange(len(original))
            for arm, model in models.items():
                arm_x = inputs.clone()
                if arm == "evidence_only":
                    arm_x[:, :, -CONTEXT_DIM:] = 0
                value = np.where(valid, model(arm_x).numpy(), -np.inf)
                probability = np.exp(value - value.max(1, keepdims=True))
                probability /= probability.sum(1, keepdims=True)
                alternate = probability.copy()
                alternate[rows, original] = -np.inf
                best = alternate.argmax(1)
                gap = probability[rows, best] - probability[rows, original]
                groups[frame["group"]][arm].append((
                    positive, gap, baseline[:, 4], best == target,
                ))
    summary = {}
    for group, arms in groups.items():
        summary[group] = {}
        for arm, chunks in arms.items():
            if chunks:
                positive, gap, weight, correct = (
                    np.concatenate([c[i] for c in chunks]) for i in range(4))
            else:
                positive, gap, weight, correct = np.array([], bool), np.array([]), np.array([]), np.array([], bool)
            summary[group][arm] = {
                "positive_count": int(positive.sum()), "negative_count": int((~positive).sum()),
                "weighted_auc": weighted_auc(positive, gap, weight),
                "unweighted_auc": weighted_auc(positive, gap, np.ones(len(weight))),
                "alternative_target_rank_accuracy_on_positive": float(correct[positive].mean())
                    if positive.any() else None,
                "positive_mean_probability_gap": float(gap[positive].mean()) if positive.any() else None,
                "negative_mean_probability_gap": float(gap[~positive].mean()) if (~positive).any() else None,
            }
    verdict = decide(summary, registration["policy"])
    save_json(output / "summary.json", summary)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "no_training": True, "no_detector_inference": True, "no_image_decoding": True,
        "no_new_action_execution_or_threshold_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "summary": summary}
