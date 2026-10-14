"""Matched ordinary absolute-quality score control versus binary recalibration."""

from copy import deepcopy
import json
import os
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.postprocess.absolute_quality_control import QualityControl, quality_scores
from sdcl.postprocess.coordinate_score_debt import threshold_quality
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate35_absolute_quality_control_registration_20261014.json"
OUTPUT = "output/analysis/absolute-quality-control-fixed-20261014"
ARMS = ("ordinary_quality", "ordinary_binary")


def prepare():
    evidence, _ = verify_record(project_path("output/analysis/coordinate-score-debt-fixed-20261014"),
                               "docs/experiments/candidate34_coordinate_score_debt_result_20261014.json")
    if (evidence["metrics"]["oracle_absolute"]["small_all"]["ap50_95"] <=
            evidence["metrics"]["regret"]["small_all"]["ap50_95"]):
        raise ValueError("No ordinary absolute-quality diagnostic headroom.")
    frames, prior, heads = load_frames()
    for records in frames.values():
        for f in records:
            row = np.arange(len(f["candidates"]))
            selected = f["regret"]
            f["score_features"] = f["features"][row, selected]
            f["active"] = f["candidates"][:, 4] >= .05
            f["reference"] = apply_actions(f["candidates"], f["boxes"], selected)
            f["selected_iou"] = f["ious"][row, 0, selected]
            f["target_quality"] = threshold_quality(f["selected_iou"][:, None], f["owner"] >= 0)[:, 0]
            f["target_binary"] = ((f["owner"] >= 0) & (f["selected_iou"] >= .5)).astype(np.float32)
    return frames, prior, heads


def fit(seed, frames, output, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    features = torch.from_numpy(np.concatenate([f["score_features"][f["active"]] for f in frames])).to(gpu)
    score = torch.from_numpy(np.concatenate([f["candidates"][f["active"], 4].astype(np.float32) for f in frames])).to(gpu)
    offset = torch.logit(score.clamp(1e-6, 1-1e-6))
    targets = {a: torch.from_numpy(np.concatenate([f[name][f["active"]] for f in frames])).to(gpu)
               for a, name in (("ordinary_quality", "target_quality"), ("ordinary_binary", "target_binary"))}
    initial = QualityControl().to(gpu)
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.0001) for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(features), generator=torch.Generator().manual_seed(seed*100+epoch)).to(gpu)
        losses = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 512):
            rows = order[start:start+512]
            for arm, model in models.items():
                loss = F.binary_cross_entropy_with_logits(offset[rows]+model(features[rows]), targets[arm][rows])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite ordinary score-control loss.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach())*len(rows)/len(order)
        history.append({"epoch": epoch+1, **losses})
    path = output/f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
                "seed": seed, "epochs": 8}, path)
    save_json(output/f"losses-seed{seed}.json", history)
    return models, {"checkpoint": path.name, "sha256": file_hash(path), "active_training_rows": len(features),
                    "parameters_each": sum(p.numel() for p in initial.parameters())}


def evaluate(models, frames, output, seed, gpu):
    predictions = {a: {} for a in ("regret", *ARMS)}
    output_path = output/f"predictions-seed{seed}"
    output_path.mkdir()
    records, manifest = [], {}
    with torch.inference_mode():
        for index, f in enumerate(frames, 1):
            reference = f["reference"]
            arms = {"regret": reference}
            corrections = {}
            for arm, model in models.items():
                residual = np.zeros(len(reference))
                residual[f["active"]] = model(torch.from_numpy(f["score_features"][f["active"]]).to(gpu)).cpu().numpy()
                arms[arm], corrections[arm] = quality_scores(reference, residual, f["active"])
            for arm, prediction in arms.items():
                predictions[arm][f["stem"]] = prediction
            path = output_path/(f["stem"]+".npz")
            np.savez_compressed(path, **arms, **{a+"_score_correction": c for a, c in corrections.items()})
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
            records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                            "objects": f["metadata"]["valid_objects"]})
    save_json(output/f"frames-seed{seed}.json", manifest)
    metrics, budgets = {}, {}
    for arm in predictions:
        metrics[arm] = coco_size_metrics(records, predictions[arm], dict(enumerate(NAMES)), 640)
        print(f"Ordinary quality seed{seed}/{arm}: small_allAP{metrics[arm]['small_all']['ap50_95']*100:.6f}", flush=True)
        if arm != "regret":
            budgets[arm], _ = recall_at_fp_budgets(records, {"baseline": predictions["regret"],
                "sdcl": predictions[arm]}, reference_size=640, budgets=(5, 10), bootstrap_samples=0)
    save_json(output/f"fixed-fp-budgets-seed{seed}.json", budgets)
    return metrics, budgets


def decide(metrics, budgets, policy):
    local, base = metrics["ordinary_quality"], metrics["regret"]
    gains = {g: (local[g]["ap50_95"]-base[g]["ap50_95"])*100 for g in ("all", "tiny", "small_all")}
    binary = (local["small_all"]["ap50_95"]-metrics["ordinary_binary"]["small_all"]["ap50_95"])*100
    fp_gains = {str(int(r["maximum_fppi"])): r["delta_recall_pp"]["small_all"] for r in budgets["ordinary_quality"]}
    checks = {
        **{g+"_AP": gains[g] >= policy[g+"_ap_vs_regret_min_pp"] for g in gains},
        "quality_vs_binary": binary > policy["small_all_ap_vs_binary_min_exclusive_pp"],
        **{"fppi"+b: fp_gains[b] >= policy["small_all_recall_fppi"+b+"_vs_regret_min_pp"] for b in ("5", "10")},
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"AP_vs_regret_pp": gains, "small_all_AP_vs_binary_pp": binary,
                       "small_all_fixed_FP_recall_gains_pp": fp_gains}}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Available CUDA index required.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/postprocess/absolute_quality_control.py", "sdcl/training/absolute_quality_control.py",
             "scripts/training/run_absolute_quality_control.py", "sdcl/postprocess/coordinate_score_debt.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads = prepare()
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": prior["image_selection"],
        "source_archive_sha256": file_hash(output/"source.zip"), "heads": heads,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate34_coordinate_score_debt_result_20261014.json"))})
    initial = QualityControl().eval()
    with torch.inference_mode():
        for records in frames.values():
            for f in records:
                residual = initial(torch.from_numpy(f["score_features"])).numpy()
                replay, correction = quality_scores(f["reference"], residual, f["active"])
                if not np.array_equal(replay, f["reference"]) or np.any(correction != 0):
                    raise ValueError("Initial score model does not exactly replay frozen regret.")
    gpu = torch.device("cuda:"+str(device))
    metrics, decisions, training = {}, {}, {}
    for seed in (0, 1, 2):
        models, saved = fit(seed, frames["train"], output, gpu)
        values, budgets = evaluate(models, frames["holdout"], output, seed, gpu)
        metrics[str(seed)], training[str(seed)] = values, saved
        decisions[str(seed)] = decide(values, budgets, registration["policy"])
        print(f"Ordinary quality seed{seed}: {json.dumps(decisions[str(seed)])}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    positive = sum(v["passed"] for v in decisions.values())
    gains = {g: float(np.mean([v["values"]["AP_vs_regret_pp"][g] for v in decisions.values()]))
             for g in ("tiny", "small_all")}
    passed = (len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_seeds"] and
              all(gains[g] >= registration["policy"][g+"_ap_vs_regret_min_pp"] for g in gains) and
              all(v["checks"]["fppi5"] and v["checks"]["fppi10"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_ORDINARY_QUALITY_CONTROL_OFFICIAL_AUDIT_ALLOWED" if passed else
               "STOP_THIS_ORDINARY_ABSOLUTE_QUALITY_CONTROL_RECIPE", "per_seed": decisions,
               "seeds_executed": len(decisions), "positive_seeds": positive,
               "scope": "Conventional stronger-control feasibility only, not innovation or official efficacy."}
    save_json(output/"metrics.json", metrics)
    save_json(output/"decision.json", verdict)
    save_json(output/"training.json", training)
    save_json(output/"verification.json", {
        "fit_images": 128, "head_holdout_images": 64, "detector_forwards": 0, "image_decodes": 0,
        "detector_training_steps": 0, "initial_all_arms_exact_regret_replay": True,
        "same_training_rows_batches_and_initial_weights": True,
        "GT_excluded_from_features_and_inference": True, "same_original_GT_owner_all_targets": True,
        "all_coordinates_classes_counts_and_array_order_preserved": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
