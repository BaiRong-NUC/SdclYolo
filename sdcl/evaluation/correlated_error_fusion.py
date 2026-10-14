"""Cached train-fitted error-dependence fusion versus matched independent voting."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.censored_retrieval import score_image
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import box_areas, file_hash
from sdcl.postprocess.correlated_error_fusion import fuse, source_order, weight_table
from sdcl.postprocess.localization import action_feature_names, apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate31_correlated_error_fusion_registration_20261014.json"
OUTPUT = "output/analysis/correlated-error-fusion-fixed-20261014"
ARMS = ("full_second_moment", "diagonal_second_moment", "equal_native_vote",
        "confidence_native_vote", "uniform", "regret")
GROUPS = ("all", "tiny", "small_all")


def prepare(frames):
    _, mean, std, _ = frozen_heads()
    names = action_feature_names()
    columns = [names.index("view_" + str(v)) for v in range(5)]
    for records in frames.values():
        for f in records:
            native, supported = source_order(f["features"], mean, std, columns, f["boxes"], f["valid"])
            f["native"], f["supported"] = native, supported
            area = box_areas(f["candidates"][:, :4]) * (640/max(f["metadata"]["original_size"]))**2
            f["group"] = np.where(area < 16**2, 0, np.where(area < 32**2, 1, 2))


def fit(frames, seed):
    selected = np.arange(len(frames)) if seed == 0 else np.random.default_rng(seed).integers(
        0, len(frames), len(frames))
    sums = np.zeros((4, 5, 5))
    counts = np.zeros((4, 5, 5), int)
    for index in selected:
        f = frames[index]
        active = (f["owner"] >= 0) & (f["candidates"][:, 4] >= .05) & (f["supported"].sum(1) >= 2)
        rows = np.flatnonzero(active)
        gt = np.asarray([o["box_xyxy"] for o in f["metadata"]["valid_objects"]], float).reshape(-1, 4)
        target = gt[f["owner"][rows]]
        scale = np.tile(f["candidates"][rows, 2:4]-f["candidates"][rows, :2], (1, 2))
        errors = (f["native"][rows]-target[:, None])/np.maximum(scale[:, None], 1e-6)
        support = f["supported"][rows]
        for group in range(4):
            included = np.ones(len(rows), bool) if group == 3 else f["group"][rows] == group
            for a in range(5):
                for b in range(5):
                    keep = included & support[:, a] & support[:, b]
                    counts[group, a, b] += int(keep.sum())
                    sums[group, a, b] += (errors[keep, a]*errors[keep, b]).sum()/4
    raw = sums/np.maximum(counts, 1)
    matrices, alignment = [], []
    for group in range(3):
        matrix = raw[group].copy()
        for view in range(5):
            if counts[group, view, view] < 32:
                if counts[3, view, view] < 32:
                    raise ValueError("Insufficient pooled diagonal support.")
                matrix[view, view] = raw[3, view, view]
        for a in range(5):
            for b in range(a+1, 5):
                if counts[group, a, b] >= 32:
                    alignment.append(abs(float(matrix[a, b] /
                        max(np.sqrt(matrix[a, a]*matrix[b, b]), 1e-12))))
                    matrix[a, b] *= .9
                    matrix[b, a] *= .9
                else:
                    matrix[a, b] = matrix[b, a] = 0.
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        floor = max(np.trace(matrix)/5*.01, 1e-9)
        matrix = (eigenvectors*np.maximum(eigenvalues, floor)) @ eigenvectors.T
        matrices.append(matrix)
    return np.stack(matrices), {
        "fit_image_indices": selected.tolist(), "paired_counts": counts.tolist(),
        "raw_second_moments": raw.tolist(), "psd_second_moments": np.stack(matrices).tolist(),
        "mean_abs_supported_offdiagonal_alignment": float(np.mean(alignment)) if alignment else 0.,
        "scope": "Uncentered normalized coordinate-error alignment, not probabilistic correlation."}


def evaluate_frames(frames, matrices, output, seed):
    full, diagonal = weight_table(matrices), weight_table(matrices, diagonal=True)
    equal = np.zeros_like(full)
    for mask in range(1, 32):
        supported = (mask & (1 << np.arange(5))) != 0
        equal[:, mask, supported] = 1/supported.sum()
    save_json(output / f"weights-fit{seed}.json", {"full": full.tolist(), "diagonal": diagonal.tolist()})
    totals = {a: {"groups": {g: dict.fromkeys(("gt", "tp50", "tp_ten"), 0) for g in GROUPS},
                  "fp50": 0, "moves": 0} for a in ARMS}
    predictions = output / f"predictions-fit{seed}"
    predictions.mkdir()
    manifest = {}
    for f in frames:
        arms = {
            "uniform": f["baseline"],
            "regret": apply_actions(f["candidates"], f["boxes"], f["regret"]),
        }
        for arm, table in (("full_second_moment", full), ("diagonal_second_moment", diagonal),
                           ("equal_native_vote", equal)):
            arms[arm] = fuse(f["candidates"], f["native"], f["supported"], f["group"], table, f["baseline"])
        vote = f["baseline"].copy()
        active = (f["candidates"][:, 4] >= .05) & (f["supported"].sum(1) >= 2)
        vote[active, :4] = f["boxes"][active, 5]
        if not f["valid"][active, 5].all():
            raise ValueError("Missing frozen confidence vote.")
        arms["confidence_native_vote"] = vote
        objects = f["metadata"]["valid_objects"]
        gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
        classes = np.asarray([o["class_id"] for o in objects], int)
        areas = box_areas(gt) * (640/max(f["metadata"]["original_size"]))**2
        for arm, prediction in arms.items():
            values = score_image(prediction, np.zeros(len(prediction), bool), gt, classes, areas)
            for g, row in values["groups"].items():
                for k, v in row.items():
                    totals[arm]["groups"][g][k] += v
            totals[arm]["fp50"] += values["fp50"]
            totals[arm]["moves"] += int((prediction[:, :4] != f["baseline"][:, :4]).any(1).sum())
        path = predictions / (f["stem"] + ".npz")
        np.savez_compressed(path, **arms)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
    save_json(output / f"frames-fit{seed}.json", manifest)
    return {a: {**{g+"_ten_threshold_recall_percent": c["tp_ten"]/max(c["gt"]*10, 1)*100
                    for g, c in values["groups"].items()},
                "all_fppi50": values["fp50"]/len(frames), "counts": values}
            for a, values in totals.items()}


def decide(metrics, mechanism, policy):
    local = metrics["full_second_moment"]
    key = "small_all_ten_threshold_recall_percent"
    comparisons = {a: local[key]-metrics[a][key] for a in ARMS if a != "full_second_moment"}
    tiny = local["tiny_ten_threshold_recall_percent"]-metrics["uniform"]["tiny_ten_threshold_recall_percent"]
    fp = local["all_fppi50"]-metrics["uniform"]["all_fppi50"]
    checks = {
        "tiny_gain": tiny >= policy["tiny_ten_threshold_vs_uniform_min_pp"],
        "small_gain": comparisons["uniform"] >= policy["small_all_ten_threshold_vs_uniform_min_pp"],
        **{a: comparisons[a] > policy["small_all_ten_threshold_vs_"+name+"_min_exclusive_pp"]
           for a, name in (("regret", "regret"), ("diagonal_second_moment", "diagonal"),
                           ("equal_native_vote", "equal"), ("confidence_native_vote", "confidence"))},
        "fp_budget": fp <= policy["all_fppi50_vs_uniform_max_increase"],
        "error_alignment": mechanism["mean_abs_supported_offdiagonal_alignment"] >=
        policy["mean_abs_supported_offdiagonal_alignment_min"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"small_all_gains_pp": comparisons, "tiny_vs_uniform_pp": tiny,
                       "fppi50_increase": fp, "mean_abs_supported_offdiagonal_alignment":
                       mechanism["mean_abs_supported_offdiagonal_alignment"]}}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/postprocess/correlated_error_fusion.py",
             "sdcl/evaluation/correlated_error_fusion.py", "scripts/evaluation/diagnose_correlated_error_fusion.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/postprocess/localization.py",
             "sdcl/evaluation/censored_retrieval.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads = load_frames()
    prepare(frames)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": prior["image_selection"],
        "source_archive_sha256": file_hash(output / "source.zip"), "heads": heads,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json"))})
    metrics, decisions, fitting = {}, {}, {}
    for seed in (0, 1, 2):
        matrices, mechanism = fit(frames["train"], seed)
        save_json(output / f"second-moments-fit{seed}.json", mechanism)
        fitting[str(seed)] = mechanism
        metrics[str(seed)] = evaluate_frames(frames["holdout"], matrices, output, seed)
        decisions[str(seed)] = decide(metrics[str(seed)], mechanism, registration["policy"])
        print(f"Error dependence fit{seed}: {json.dumps(decisions[str(seed)])}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    positive = sum(v["passed"] for v in decisions.values())
    gain = float(np.mean([v["values"]["small_all_gains_pp"]["uniform"] for v in decisions.values()]))
    passed = (len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_fits"] and
              gain >= registration["policy"]["small_all_ten_threshold_vs_uniform_min_pp"] and
              all(v["checks"]["fp_budget"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_ERROR_DEPENDENCE_OFFICIAL_AUDIT_ALLOWED" if passed else
               "STOP_THIS_POOLED_SECOND_MOMENT_FUSION",
               "per_fit": decisions, "fits_executed": len(decisions), "positive_fits": positive,
               "scope": "Reused train-head-holdout detection proxy, not AP or novelty."}
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "fit_images": 128, "head_holdout_images": 64, "detector_forwards": 0,
        "image_decodes": 0, "gradient_training_steps": 0, "source_view_identity_replay_exact": True,
        "no_GT_in_holdout_features_selection_or_grouping": True, "same_convex_weights_for_all_four_sides": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
