"""Small matched marginal-loss versus standalone-coverage view selectors."""

from copy import deepcopy
import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import max_pool_matches
from sdcl.evaluation.native_censoring import pool_edges
from sdcl.evaluation.size_analysis import box_areas, file_hash, group_mask
from sdcl.evaluation.view_budget_headroom import control_masks
from sdcl.postprocess.rescue import clean_views, save_json
from sdcl.postprocess.view_value import global_window_features, ViewValueRegressor
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate26_marginal_view_policy_registration_20261014.json"
OUTPUT = "output/analysis/marginal-view-policy-fixed-20261014"
BASE = "output/analysis/class-choice-pilot-fixed-20261014"
NATIVE = "output/analysis/native-censoring-fixed-20261014"
BUDGET = "output/analysis/view-budget-headroom-fixed-20261014"
ARMS = ("ordinary_coverage", "marginal_loss")
GROUPS = ("all", "tiny", "small_all")
THRESHOLDS = (.5, .75)


def labels_and_losses(gt, classes, areas, pool, view):
    edges = {t: pool_edges(gt, classes, pool, .05, t) for t in THRESHOLDS}
    full_support = {t: e.any(1) for t, e in edges.items()}
    full_capacity = {t: max_pool_matches(e) for t, e in edges.items()}
    targets, losses = [], []
    for omitted in range(1, 5):
        target = dict.fromkeys(ARMS, 0)
        loss = {}
        for threshold, edge in edges.items():
            remaining = edge[:, view != omitted]
            lost = full_support[threshold] & ~remaining.any(1)
            capacity_loss = full_capacity[threshold] - max_pool_matches(remaining)
            target["ordinary_coverage"] += max_pool_matches(edge[:, view == omitted])
            target["marginal_loss"] += max(int(lost.sum()), capacity_loss)
            loss[str(threshold)] = {
                "lost_supported_gt": {g: int((lost & group_mask(areas, g)).sum()) for g in GROUPS},
                "capacity_loss": capacity_loss,
            }
        targets.append([target[arm] for arm in ARMS])
        losses.append(loss)
    return np.asarray(targets, np.float32), losses


def prepare():
    base, native, budget = (project_path(p) for p in (BASE, NATIVE, BUDGET))
    _, plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    _, native_plan = verify_record(native, "docs/experiments/candidate21_native_censoring_result_20261014.json")
    gate, _ = verify_record(budget, "docs/experiments/candidate25_view_budget_result_20261014.json")
    if gate["decision"]["decision"] != "BOUNDED_VIEW_POLICY_SCREEN_ALLOWED":
        raise ValueError("View-budget screen did not pass.")
    selected = {"train": plan["image_selection"]["train"][:128],
                "holdout": plan["image_selection"]["holdout"][:64]}
    if ({s.split("_")[0] for s in selected["train"]} &
            {s.split("_")[0] for s in selected["holdout"]}):
        raise ValueError("Train and head-held-out scenes overlap.")
    raw_path = base / "raw_views.jsonl"
    if file_hash(raw_path) != native_plan["prior_raw_views_sha256"]:
        raise ValueError("Pinned raw predictions changed.")
    raw = {r["image"]: r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
    frames = json.loads((native / "frames.json").read_text())
    records = {r["stem"]: r for rows in plan["records"].values() for r in rows}
    result = {"train": [], "holdout": []}
    for split, stems in selected.items():
        for stem in stems:
            item = raw[stem]
            size = item["windows"][0][2:]
            # Unfiltered fullimage predictions are the ONLY model evidence.
            global_predictions = np.asarray(item["views"][0], float).reshape(-1, 6)
            x = global_window_features(global_predictions, item["windows"], size)
            controls = control_masks(global_predictions, item["windows"])
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            if file_hash(meta_path) != records[stem]["metadata_sha256"]:
                raise ValueError("Fixed diagnostic metadata changed.")
            meta = json.loads(meta_path.read_text())
            views = [np.asarray(v, np.float32).reshape(-1, 6) for v in item["views"]]
            views = [v[v[:, 4] >= .05] for v in views]
            pool, origin = clean_views(views, size, meta["ignore_boxes_xyxy"])
            if split == "holdout":
                path = native / frames[stem]["file"]
                if file_hash(path) != frames[stem]["sha256"]:
                    raise ValueError("Frozen native heldout frame changed.")
                with np.load(path, allow_pickle=False) as c:
                    if not np.array_equal(pool, c["kept"]) or not np.array_equal(origin, c["kept_view"]):
                        raise ValueError("Diagnostic pool reconstruction differs.")
            gt = np.asarray([o["box_xyxy"] for o in meta["valid_objects"]], float).reshape(-1, 4)
            classes = np.asarray([o["class_id"] for o in meta["valid_objects"]], int)
            areas = box_areas(gt) * (640 / max(size)) ** 2
            targets, losses = labels_and_losses(gt, classes, areas, pool, origin)
            result[split].append({
                "stem": stem, "features": x, "targets": targets, "losses": losses,
                "density_omitted_index": int(np.log2(15 ^ controls["global_density_one"])),
                "gt": {g: int(group_mask(areas, g).sum()) for g in GROUPS},
            })
    return selected, result, gate


def fit_and_choose(seed, frames, output):
    train = frames["train"]
    features = np.concatenate([r["features"] for r in train])
    targets = np.log1p(np.concatenate([r["targets"] for r in train]))
    mean = features.mean(0)
    std = features.std(0).clip(1e-3)
    target_mean, target_std = targets.mean(0), targets.std(0).clip(1e-3)
    x = torch.from_numpy(np.clip((features - mean) / std, -6, 6))
    y = torch.from_numpy((targets - target_mean) / target_std)
    torch.manual_seed(seed)
    initial = ViewValueRegressor(features.shape[1])
    models = {arm: deepcopy(initial) for arm in ARMS}
    optimizers = {arm: torch.optim.Adam(model.parameters(), lr=.003) for arm, model in models.items()}
    rng = np.random.default_rng(seed)
    losses = {}
    for epoch in range(80):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            rows = order[start:start+128]
            for column, arm in enumerate(ARMS):
                optimizers[arm].zero_grad(set_to_none=True)
                loss = (models[arm](x[rows]) - y[rows, column]).square().mean()
                loss.backward()
                optimizers[arm].step()
                losses[arm] = float(loss.detach())
    checkpoint = output / f"heads-seed{seed}.pt"
    torch.save({
        "seed": seed, "epochs": 80, "mean": mean, "std": std,
        "target_mean": target_mean, "target_std": target_std,
        "models": {a: m.state_dict() for a, m in models.items()},
        "dimensions": features.shape[1], "registration_sha256": file_hash(project_path(REGISTRATION)),
    }, checkpoint)
    choices = {}
    with torch.inference_mode():
        for frame in frames["holdout"]:
            xh = torch.from_numpy(np.clip((frame["features"] - mean) / std, -6, 6))
            choices[frame["stem"]] = {
                **{arm: int(model(xh).argmin()) for arm, model in models.items()},
                "global_density_one": frame["density_omitted_index"], "fixed_one": 3,
            }
    return choices, {
        "parameters_per_head": sum(p.numel() for p in initial.parameters()),
        "input_dimensions": features.shape[1], "training_rows": len(x),
        "final_batch_loss": losses, "checkpoint": checkpoint.name, "sha256": file_hash(checkpoint),
    }


def score_choices(choices, frames):
    counts = {g: sum(f["gt"][g] for f in frames) for g in GROUPS}
    arms = (*ARMS, "global_density_one", "fixed_one")
    totals = {a: {str(t): {"lost_supported_gt": dict.fromkeys(GROUPS, 0), "capacity_loss": 0}
                  for t in THRESHOLDS} for a in arms}
    for frame in frames:
        for arm, index in choices[frame["stem"]].items():
            for threshold, loss in frame["losses"][index].items():
                totals[arm][threshold]["capacity_loss"] += loss["capacity_loss"]
                for group, n in loss["lost_supported_gt"].items():
                    totals[arm][threshold]["lost_supported_gt"][group] += n
    return {a: {
        "forward_count_saving_percent": 20.,
        "thresholds": data,
        "lost_supported_gt_pp": {t: {g: n / max(counts[g], 1) * 100 for g, n in
                                     row["lost_supported_gt"].items()} for t, row in data.items()},
    } for a, data in totals.items()}


def decide(metrics, policy):
    def lost(arm, threshold, group):
        return metrics[arm]["thresholds"][threshold]["lost_supported_gt"][group]
    values = {
        "tiny075_reduction_vs_ordinary": lost("ordinary_coverage", "0.75", "tiny") -
        lost("marginal_loss", "0.75", "tiny"),
        "tiny075_reduction_vs_density": lost("global_density_one", "0.75", "tiny") -
        lost("marginal_loss", "0.75", "tiny"),
        "tiny050_extra_vs_density": lost("marginal_loss", "0.5", "tiny") -
        lost("global_density_one", "0.5", "tiny"),
        "all075_extra_vs_ordinary": lost("marginal_loss", "0.75", "all") -
        lost("ordinary_coverage", "0.75", "all"),
        "tiny075_loss_pp": metrics["marginal_loss"]["lost_supported_gt_pp"]["0.75"]["tiny"],
    }
    checks = {
        "ordinary_objective_gain": values["tiny075_reduction_vs_ordinary"] >=
        policy["tiny_lost_support075_vs_ordinary_min_reduction_count"],
        "density_gain": values["tiny075_reduction_vs_density"] >=
        policy["tiny_lost_support075_vs_density_min_reduction_count"],
        "low_iou_budget": values["tiny050_extra_vs_density"] <=
        policy["tiny_lost_support050_vs_density_max_extra_count"],
        "all_size_budget": values["all075_extra_vs_ordinary"] <=
        policy["all_lost_support075_vs_ordinary_max_extra_count"],
        "absolute_loss_budget": values["tiny075_loss_pp"] <= policy["tiny_lost_support075_max_pp"],
        "same_cost": all(abs(m["forward_count_saving_percent"] -
                             policy["forward_saving_percent"]) < 1e-9 for m in metrics.values()),
    }
    return {"passed": all(checks.values()), "checks": checks, "values": values}


def evaluate(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/view_value.py", "sdcl/training/marginal_view_policy.py",
             "scripts/training/run_marginal_view_policy.py", "sdcl/evaluation/view_budget_headroom.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/rescue.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    selected, frames, _ = prepare()
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": selected,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate25_view_budget_result_20261014.json")),
    })
    for split, records in frames.items():
        np.savez_compressed(output / f"{split}-features-targets.npz",
                            features=np.stack([f["features"] for f in records]),
                            targets=np.stack([f["targets"] for f in records]))
    metrics, decisions, checkpoints = {}, {}, {}
    for seed in (0, 1, 2):
        choices, training = fit_and_choose(seed, frames, output)
        # Prediction choices have been materialized before heldout labels are scored.
        save_json(output / f"choices-seed{seed}.json", choices)
        result = score_choices(choices, frames["holdout"])
        verdict = decide(result, registration["policy"])
        metrics[str(seed)], decisions[str(seed)], checkpoints[str(seed)] = result, verdict, training
        print(f"Marginal view policy seed{seed}: {json.dumps(verdict)}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not verdict["passed"]:
            break
    positive = sum(d["passed"] for d in decisions.values())
    passed = len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_seeds"]
    # A replicated policy cannot exceed the absolute loss budget on any seed.
    passed &= all(d["checks"]["absolute_loss_budget"] for d in decisions.values())
    verdict = {
        "decision": "BOUNDED_VIEW_POLICY_OFFICIAL_REPLAY_ALLOWED" if passed else
        "STOP_THIS_GLOBAL_FEATURE_MARGINAL_VIEW_REGRESSOR",
        "per_seed": decisions, "positive_seeds": positive, "seeds_executed": len(decisions),
        "scope": "Coverage proxy on reused train head-holdout, not AP/latency or novelty proof.",
    }
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "training.json", checkpoints)
    save_json(output / "verification.json", {
        "fitting_images": len(frames["train"]), "head_heldout_images": len(frames["holdout"]),
        "scene_prefix_sets_disjoint": True, "detector_forwards": 0, "no_val_no_testdev": True,
        "heldout_pools_exactly_match_candidate21": True,
        "model_features_only_unfiltered_global_predictions": True,
        "heads_same_initialization_capacity_batches_epochs_per_seed": True,
        "no_hyperparameter_or_threshold_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
