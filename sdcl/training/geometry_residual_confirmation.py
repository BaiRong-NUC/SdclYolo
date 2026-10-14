"""Confirm a conventional continuous-geometry control without detector inference."""

from copy import deepcopy
import json
import os
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.native_regression_residual import NativeResidual, correct_coordinates
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate39_geometry_residual_confirmation_registration_20261014.json"
OUTPUT = "output/analysis/geometry-residual-confirmation-fixed-20261014"
PRIOR = "output/analysis/native-regression-residual-fixed-20261014"


class ConstantResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(4))

    def forward(self, features):
        return (.1*torch.tanh(self.logit))[None].expand(len(features), -1)


def prepare():
    base = project_path(PRIOR)
    prior_result, _ = verify_record(base, "docs/experiments/candidate38_native_regression_features_result_20261014.json")
    frames, prior, heads = load_frames()
    manifest = json.loads((base/"features.json").read_text())
    norm = json.loads((base/"normalization.json").read_text())["geometry"]
    for split, values in frames.items():
        for f in values:
            entry = manifest[f["stem"]]
            path = base/entry["file"]
            if file_hash(path) != entry["sha256"] or entry["split"] != split:
                raise ValueError("Frozen conventional-control features changed.")
            with np.load(path, allow_pickle=False) as c:
                geometry, extra, active = c["geometry"].copy(), c["extra_geometry"].copy(), c["active"].copy()
            f["reference"] = apply_actions(f["candidates"], f["boxes"], f["regret"])
            f["active"] = active
            # Match candidate38 float32 arithmetic before concatenating to avoid
            # a change of precision when loading JSON normalization statistics.
            f["input"] = np.concatenate((geometry, (extra-np.asarray(norm["mean"], np.float32)) /
                                        np.asarray(norm["std"], np.float32)), -1).astype(np.float32)
            f["scale"] = np.tile(f["candidates"][:, 2:4]-f["candidates"][:, :2], (1, 2))
            gt = np.array([o["box_xyxy"] for o in f["metadata"]["valid_objects"]], float).reshape(-1, 4)
            owned = f["owner"] >= 0
            target = np.zeros((len(owned), 4), np.float32)
            target[owned] = (gt[f["owner"][owned]]-f["reference"][owned, :4])/np.maximum(f["scale"][owned], 1e-6)
            f["target"] = np.clip(target, -.1, .1)
    checkpoint = torch.load(base/"heads-seed0.pt", map_location="cpu", weights_only=True)
    initial = NativeResidual().eval()
    initial.load_state_dict(checkpoint["models"]["geometry"])
    prior_predictions = json.loads((base/"frames-seed0.json").read_text())
    with torch.inference_mode():
        for f in frames["holdout"]:
            path = base/prior_predictions[f["stem"]]["file"]
            if file_hash(path) != prior_predictions[f["stem"]]["sha256"]:
                raise ValueError("Historical geometry predictions changed.")
            with np.load(path, allow_pickle=False) as c:
                f["seed0_geometry"] = c["geometry"].copy()
    return frames, prior, heads, prior_result, initial


def fit(seed, frames, output, gpu, geometry):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.from_numpy(np.concatenate([f["input"][f["active"]] for f in frames])).to(gpu)
    y = torch.from_numpy(np.concatenate([f["target"][f["active"]] for f in frames])).to(gpu)
    owned = torch.from_numpy(np.concatenate([f["owner"][f["active"]] >= 0 for f in frames])).to(gpu)
    initial = NativeResidual().to(gpu)
    models = {"geometry": deepcopy(initial)} if geometry else {}
    models["constant"] = ConstantResidual().to(gpu)
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.0001) for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed*100+epoch)).to(gpu)
        losses = dict.fromkeys(models, 0.)
        for start in range(0, len(order), 512):
            row = order[start:start+512]
            for a, model in models.items():
                perrow = F.smooth_l1_loss(model(x[row]), y[row], beta=.02, reduction="none").mean(1)
                groups = [perrow[owned[row] == state].mean() for state in (False, True) if (owned[row] == state).any()]
                loss = torch.stack(groups).mean()
                optimizers[a].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[a].step()
                losses[a] += float(loss.detach())*len(row)/len(x)
        history.append({"epoch": epoch+1, **losses})
    path = output/f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
                "seed": seed, "epochs": 8}, path)
    save_json(output/f"losses-seed{seed}.json", history)
    return models, {"checkpoint": path.name, "sha256": file_hash(path), "training_rows": len(x),
                    "parameters": {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}}


def evaluate(models, frames, output, seed, gpu, prior_metrics):
    predictions = {a: {} for a in ("regret", "geometry", "constant")}
    records, manifest = [], {}
    target = output/f"predictions-seed{seed}"
    target.mkdir()
    with torch.inference_mode():
        for index, f in enumerate(frames, 1):
            arms = {"regret": f["reference"]}
            for a, model in models.items():
                delta = np.zeros((len(f["reference"]), 4))
                delta[f["active"]] = model(torch.from_numpy(f["input"][f["active"]]).to(gpu)).cpu().numpy()
                arms[a] = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
            if seed == 0 and not np.array_equal(arms["geometry"], f["seed0_geometry"]):
                raise ValueError("Conventional geometry seed0 does not exactly replay candidate38.")
            for a, values in arms.items():
                predictions[a][f["stem"]] = values
            path = target/(f["stem"]+".npz")
            np.savez_compressed(path, **arms)
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
            records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                            "objects": f["metadata"]["valid_objects"]})
    metrics = {"regret": prior_metrics["regret"]}
    if seed == 0:
        metrics["geometry"] = prior_metrics["geometry"]
    for a in ("geometry", "constant"):
        if a not in metrics:
            metrics[a] = coco_size_metrics(records, predictions[a], dict(enumerate(NAMES)), 640)
        print(f"Geometry confirmation seed{seed}/{a}: small_allAP{metrics[a]['small_all']['ap50_95']*100:.6f}", flush=True)
    budgets, _ = recall_at_fp_budgets(records, {"baseline": predictions["regret"], "sdcl": predictions["geometry"]},
                                     reference_size=640, budgets=(10,), bootstrap_samples=0)
    save_json(output/f"frames-seed{seed}.json", manifest)
    save_json(output/f"fixed-fp-budgets-seed{seed}.json", budgets)
    return metrics, budgets


def decide(metrics, budgets, policy):
    gains = {g: (metrics["geometry"][g]["ap50_95"]-metrics["regret"][g]["ap50_95"])*100
             for g in ("tiny", "small_all", "all")}
    bias = {g: (metrics["geometry"][g]["ap50_95"]-metrics["constant"][g]["ap50_95"])*100
            for g in ("tiny", "small_all")}
    fp = budgets[0]["delta_recall_pp"]["small_all"]
    checks = {**{g+"_AP": v >= policy[g+"_ap_vs_regret_min_pp"] for g, v in gains.items()},
              **{g+"_vs_constant": v >= policy[g+"_ap_vs_constant_min_pp"] for g, v in bias.items()},
              "fppi10": fp >= policy["small_all_recall_fppi10_vs_regret_min_pp"]}
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"AP_vs_regret_pp": gains, "AP_vs_constant_pp": bias,
                       "small_all_recall_fppi10_gain_pp": fp}}


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
    paths = (REGISTRATION, "sdcl/training/geometry_residual_confirmation.py",
             "scripts/training/run_geometry_residual_confirmation.py",
             "sdcl/postprocess/native_regression_residual.py", "sdcl/training/native_regression_residual.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads, prior_result, seed0 = prepare()
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "heads": heads,
        "source_archive_sha256": file_hash(output/"source.zip"), "image_selection": prior["image_selection"],
        "prior_result_sha256": file_hash(project_path("docs/experiments/candidate38_native_regression_features_result_20261014.json"))})
    gpu = torch.device("cuda:"+str(device))
    prior_metrics = json.loads((project_path(PRIOR)/"metrics.json").read_text())["0"]
    metrics, decisions, training = {}, {}, {}
    for seed in (0, 1, 2):
        models, record = fit(seed, frames["train"], output, gpu, seed != 0)
        if seed == 0:
            models["geometry"] = seed0.to(gpu)
        values, budgets = evaluate(models, frames["holdout"], output, seed, gpu, prior_metrics)
        metrics[str(seed)], training[str(seed)] = values, record
        decisions[str(seed)] = decide(values, budgets, registration["policy"])
        print(f"Geometry confirmation seed{seed}: {json.dumps(decisions[str(seed)])}, {perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    mean_gains = {comparison: {g: float(np.mean([v["values"][comparison][g] for v in decisions.values()]))
                              for g in ("tiny", "small_all")}
                  for comparison in ("AP_vs_regret_pp", "AP_vs_constant_pp")}
    passed = (len(decisions) == 3 and sum(v["passed"] for v in decisions.values()) >= 2 and
              all(mean_gains[c][g] >= registration["policy"][g+"_ap_vs_"+a+"_min_pp"]
                  for c, a in (("AP_vs_regret_pp", "regret"), ("AP_vs_constant_pp", "constant"))
                  for g in ("tiny", "small_all")) and
              all(v["checks"]["all_AP"] and v["checks"]["fppi10"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_CONVENTIONAL_GEOMETRY_OFFICIAL_AUDIT_ALLOWED" if passed else
               "STOP_THIS_ORDINARY_GEOMETRY_RESIDUAL_CONFIRMATION", "per_seed": decisions, "mean_gains": mean_gains,
               "scope": "Repeated-train-scene conventional control only, no novelty or official confirmation."}
    for name, values in (("metrics", metrics), ("training", training), ("decision", verdict)):
        save_json(output/(name+".json"), values)
    save_json(output/"verification.json", {
        "detector_forwards": 0, "image_decodes": 0, "detector_training_steps": 0,
        "seed0_geometry_predictions_exact_replay": True, "train_only_normalization_and_same_original_owner": True,
        "unchanged_candidate38_geometry_recipe": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
