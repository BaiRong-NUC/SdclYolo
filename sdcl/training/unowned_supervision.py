"""Conventional geometry controls for a pre-registered zero-target coverage gap."""

from copy import deepcopy
import hashlib
import json
import os
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import box_iou_matrix, coco_size_metrics, file_hash, match_at_threshold, recall_at_fp_budgets
from sdcl.evaluation.unowned_reachability import coordinate_bounds, projected_iou
from sdcl.postprocess.native_regression_residual import NativeResidual, correct_coordinates
from sdcl.postprocess.rescue import save_json
from sdcl.training.geometry_residual_confirmation import prepare
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate44_unowned_supervision_registration_20261014.json"
OUTPUT = "output/analysis/unowned-supervision-fixed-20261014"
GEOMETRY = "output/analysis/geometry-residual-confirmation-fixed-20261014"
ARMS = ("original", "naive", "unique", "random_count")
GROUPS = ("tiny", "small_all", "all")


def supervision_targets(gt, classes, regret, scale, active, owner, original_target, geometry, size, stem):
    lower, upper = coordinate_bounds(regret, scale, size)
    possible = ((projected_iou(gt, lower, upper) >= .5) &
                (classes[:, None] == regret[None, :, 5]))
    matches, used = match_at_threshold(box_iou_matrix(gt, geometry[:, :4]), classes, geometry, .25)
    eligible = possible & (matches < 0)[:, None]
    naive_rows = active & (owner < 0) & (regret[:, 4] >= .25) & ~used & eligible.any(0)
    unique_rows = naive_rows & (possible.sum(0) == 1)
    assigned = np.full(len(regret), -1, int)
    rows = np.flatnonzero(naive_rows)
    if len(rows):
        quality = box_iou_matrix(gt, regret[rows, :4])
        assigned[rows] = np.where(eligible[:, rows], quality, -1).argmax(0)
    seed = int.from_bytes(hashlib.sha256(f"{stem}:coverage-count".encode()).digest()[:8], "little")
    selected = np.random.default_rng(seed).choice(rows, int(unique_rows.sum()), replace=False)
    random_rows = np.zeros(len(regret), bool)
    random_rows[selected] = True
    additions = {"original": np.zeros(len(regret), bool), "naive": naive_rows,
                 "unique": unique_rows, "random_count": random_rows}
    targets, owners = {}, {}
    for arm, mask in additions.items():
        targets[arm], owners[arm] = original_target.copy(), owner.copy()
        owners[arm][mask] = assigned[mask]
        targets[arm][mask] = np.clip((gt[assigned[mask]]-regret[mask, :4])/np.maximum(scale[mask], 1e-6), -.1, .1)
        if np.any(mask & (owner >= 0)):
            raise ValueError("Added supervision changed original owned rows.")
    return targets, owners, additions


def build_labels(frames, initial, gpu, output):
    manifest, counts = {}, dict.fromkeys(ARMS, 0)
    (output/"labels").mkdir()
    initial = initial.to(gpu).eval()
    with torch.inference_mode():
        for f in frames:
            delta = np.zeros((len(f["reference"]), 4))
            delta[f["active"]] = initial(torch.from_numpy(f["input"][f["active"]]).to(gpu)).cpu().numpy()
            geometry = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
            objects = f["metadata"]["valid_objects"]
            gt = np.array([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in objects], int)
            targets, owners, additions = supervision_targets(gt, classes, f["reference"], f["scale"], f["active"],
                                                            f["owner"], f["target"], geometry,
                                                            f["metadata"]["original_size"], f["stem"])
            f.update(coverage_targets=targets, coverage_owners=owners)
            if additions["unique"].sum() != additions["random_count"].sum():
                raise ValueError("Per-image positive-count control differs.")
            path = output/"labels"/(f["stem"]+".npz")
            np.savez_compressed(path, **{a+"_targets": v for a, v in targets.items()},
                                **{a+"_owners": v for a, v in owners.items()},
                                **{a+"_added": v for a, v in additions.items()})
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
            for arm in ARMS:
                counts[arm] += int(additions[arm].sum())
    save_json(output/"labels.json", manifest)
    return counts


def fit(seed, frames, output, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.from_numpy(np.concatenate([f["input"][f["active"]] for f in frames])).to(gpu)
    y = {a: torch.from_numpy(np.concatenate([f["coverage_targets"][a][f["active"]] for f in frames])).to(gpu) for a in ARMS}
    owned = {a: torch.from_numpy(np.concatenate([f["coverage_owners"][a][f["active"]] >= 0 for f in frames])).to(gpu) for a in ARMS}
    initial = NativeResidual().to(gpu)
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.0001) for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed*100+epoch)).to(gpu)
        losses = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 512):
            rows = order[start:start+512]
            for arm, model in models.items():
                perrow = F.smooth_l1_loss(model(x[rows]), y[arm][rows], beta=.02, reduction="none").mean(1)
                groups = [perrow[owned[arm][rows] == state].mean() for state in (False, True) if (owned[arm][rows] == state).any()]
                loss = torch.stack(groups).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite conventional supervision loss.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach())*len(rows)/len(x)
        history.append({"epoch": epoch+1, **losses})
    path = output/f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
                "seed": seed, "epochs": 8}, path)
    save_json(output/f"losses-seed{seed}.json", history)
    return models, {"checkpoint": path.name, "sha256": file_hash(path), "training_rows": len(x),
                    "parameters_each": sum(p.numel() for p in initial.parameters())}


def evaluate(models, frames, output, seed, gpu, controls):
    base = project_path(GEOMETRY)
    prior = json.loads((base/f"frames-seed{seed}.json").read_text())
    predictions = {a: {} for a in ("regret", *ARMS)}
    records, manifest = [], {}
    folder = output/f"predictions-seed{seed}"
    folder.mkdir()
    with torch.inference_mode():
        for index, f in enumerate(frames, 1):
            arms = {"regret": f["reference"]}
            for arm, model in models.items():
                delta = np.zeros((len(f["reference"]), 4))
                delta[f["active"]] = model(torch.from_numpy(f["input"][f["active"]]).to(gpu)).cpu().numpy()
                arms[arm] = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
                if not np.array_equal(arms[arm][:, 4:], f["reference"][:, 4:]) or not np.array_equal(
                        arms[arm][~f["active"]], f["reference"][~f["active"]]):
                    raise ValueError("Conventional supervision changed score/class/inactive rows.")
            path = base/prior[f["stem"]]["file"]
            if file_hash(path) != prior[f["stem"]]["sha256"]:
                raise ValueError("Frozen geometry confirmation prediction changed.")
            with np.load(path, allow_pickle=False) as previous:
                if not np.array_equal(arms["original"], previous["geometry"]):
                    raise ValueError("Original geometry refit is not an exact heldout replay.")
            for arm, values in arms.items():
                predictions[arm][f["stem"]] = values
            path = folder/(f["stem"]+".npz")
            np.savez_compressed(path, **arms)
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
            records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                            "objects": f["metadata"]["valid_objects"]})
    metrics = {"regret": controls[str(seed)]["regret"], "original": controls[str(seed)]["geometry"]}
    for arm in ARMS[1:]:
        metrics[arm] = coco_size_metrics(records, predictions[arm], dict(enumerate(NAMES)), 640)
        print(f"Unowned supervision seed{seed}/{arm}: small_allAP{metrics[arm]['small_all']['ap50_95']*100:.6f}", flush=True)
    budgets = {}
    for baseline in ("regret", "original"):
        budgets[baseline], _ = recall_at_fp_budgets(records, {"baseline": predictions[baseline], "sdcl": predictions["unique"]},
                                                  reference_size=640, budgets=(10,), bootstrap_samples=0)
    save_json(output/f"frames-seed{seed}.json", manifest)
    save_json(output/f"fixed-fp-budgets-seed{seed}.json", budgets)
    return metrics, budgets


def decide(metrics, budgets, policy):
    gains = {a: {g: (metrics["unique"][g]["ap50_95"]-metrics[a][g]["ap50_95"])*100 for g in GROUPS}
             for a in ("regret", "original", "naive", "random_count")}
    checks = {g+"_vs_"+a: gains[a][g] >= policy[g+"_ap_vs_"+a+"_min_pp"]
              for a in ("regret", "original") for g in GROUPS}
    checks.update({"small_all_vs_"+a: gains[a]["small_all"] >= policy["small_all_ap_vs_"+a+"_min_pp"]
                   for a in ("naive", "random_count")})
    fp = {a: budgets[a][0]["delta_recall_pp"]["small_all"] for a in ("regret", "original")}
    checks.update({"fppi10_vs_"+a: v >= policy["small_all_recall_fppi10_vs_"+a+"_min_pp"] for a, v in fp.items()})
    return {"passed": all(checks.values()), "checks": checks, "values": {"AP_gains_pp": gains, "FPPI10_recall_gains_pp": fp}}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("CUDA required for exact geometry training replay.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/training/unowned_supervision.py", "scripts/training/run_unowned_supervision.py",
             "sdcl/evaluation/unowned_reachability.py", "sdcl/training/geometry_residual_confirmation.py",
             "sdcl/postprocess/native_regression_residual.py", "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    gate, _ = verify_record(project_path("output/analysis/unowned-reachability-fixed-20261014"),
                            "docs/experiments/candidate43_unowned_reachability_result_20261014.json")
    if gate["decision"]["decision"] != "BOUNDED_UNOWNED_SUPERVISION_CONTROL_ALLOWED":
        raise ValueError("Candidate43 did not authorize conventional supervision controls.")
    verify_record(project_path(GEOMETRY), "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json")
    frames, prior, heads, _, initial = prepare()
    gpu = torch.device("cuda:"+str(device))
    save_json(output/"plan.json", {"registration": registration, "source_hashes": hashes, "heads": heads,
                                 "source_archive_sha256": file_hash(output/"source.zip"), "image_selection": prior["image_selection"],
                                 "prior_result_sha256": file_hash(project_path(
                                     "docs/experiments/candidate43_unowned_reachability_result_20261014.json"))})
    counts = build_labels(frames["train"], initial, gpu, output)
    print(f"Added fit-only labels: {json.dumps(counts)}", flush=True)
    controls = json.loads((project_path(GEOMETRY)/"metrics.json").read_text())
    metrics, training, decisions = {}, {}, {}
    for seed in (0, 1, 2):
        models, history = fit(seed, frames["train"], output, gpu)
        values, budgets = evaluate(models, frames["holdout"], output, seed, gpu, controls)
        metrics[str(seed)], training[str(seed)] = values, history
        decisions[str(seed)] = decide(values, budgets, registration["policy"])
        print(f"Unowned supervision seed{seed}: {json.dumps(decisions[str(seed)])}, {perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    mean_metrics = {a: {g: {"ap50_95": float(np.mean([v[a][g]["ap50_95"] for v in metrics.values()]))} for g in GROUPS}
                    for a in ("regret", *ARMS)}
    worst_fp = {a: [{"delta_recall_pp": {"small_all": min(v["values"]["FPPI10_recall_gains_pp"][a] for v in decisions.values())}}]
                for a in ("regret", "original")}
    mean_decision = decide(mean_metrics, worst_fp, registration["policy"])
    passed = (len(decisions) == 3 and sum(v["passed"] for v in decisions.values()) >=
              registration["policy"]["repeat_min_fully_passing_seeds"] and mean_decision["passed"] and
              all(v["checks"][g+"_vs_"+a] for v in decisions.values()
                  for g in ("all", "fppi10") for a in ("regret", "original")))
    verdict = {"decision": "BOUNDED_CONVENTIONAL_COVERAGE_CONFIRMATION_ALLOWED" if passed else
               "STOP_THIS_UNOWNED_SUPERVISION_CONTROL_RECIPE", "per_seed": decisions,
               "mean_AP_gains_pp": mean_decision["values"]["AP_gains_pp"],
               "scope": "Conventional coverage control only; not official AP, independent evidence or innovation."}
    for name, value in (("metrics", metrics), ("training", training), ("decision", verdict)):
        save_json(output/(name+".json"), value)
    save_json(output/"verification.json", {"detector_forwards": 0, "image_decodes": 0, "detector_updates": 0,
                                          "training_label_images": 128, "head_holdout_images": 64,
                                          "added_fit_labels": counts, "per_image_unique_random_positive_count_matched": True,
                                          "all_original_geometry_predictions_exact_replay": True,
                                          "holdout_GT_not_used_for_targets_or_inference_routing": True,
                                          "scores_classes_count_order_inactive_preserved": True,
                                          "unchanged_original132inputs_normalization_active_mask": True,
                                          "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                                          "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
