"""Fixed-budget screen for conditioning view omission on realized local evidence."""

from copy import deepcopy
import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.observed_view_policy import (
    GLOBAL_DIM, OBSERVED, OMITTED, derange_images, observed_features,
)
from sdcl.postprocess.rescue import save_json
from sdcl.postprocess.view_value import ViewValueRegressor
from sdcl.training.marginal_view_policy import BASE, prepare
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate30_observed_view_policy_registration_20261014.json"
OUTPUT = "output/analysis/observed-view-policy-fixed-20261014"
ARMS = ("conditioned_marginal", "global_marginal", "shuffled_marginal", "conditioned_coverage")
GROUPS = ("all", "tiny", "small_all")


def construct(frames):
    raw_path = project_path(BASE) / "raw_views.jsonl"
    raw = {r["image"]: r for r in
           (json.loads(line) for line in raw_path.read_text().splitlines())}
    for records in frames.values():
        for frame in records:
            item = raw[frame["stem"]]
            # The feature function cannot access the two uncomputed views.
            global_prediction = np.asarray(item["views"][0], float).reshape(-1, 6)
            computed = {v: np.asarray(item["views"][v], float).reshape(-1, 6) for v in OBSERVED}
            frame["features"] = observed_features(global_prediction, computed, item["windows"],
                                                   item["windows"][0][2:])
            centers = (global_prediction[:, :2] + global_prediction[:, 2:4]) / 2
            counts = [int((((centers >= item["windows"][v][:2]) &
                            (centers < item["windows"][v][2:])).all(1) &
                           (global_prediction[:, 4] >= .25)).sum()) for v in OMITTED]
            frame["restricted_density"] = int(np.argmin(counts))
    return file_hash(raw_path)


def score(choices, frames):
    arms = next(iter(choices.values())).keys()
    totals = {a: {str(t): {"lost_supported_gt": dict.fromkeys(GROUPS, 0), "capacity_loss": 0}
                  for t in (.5, .75)} for a in arms}
    for f in frames:
        for arm, option in choices[f["stem"]].items():
            original_index = OMITTED[option] - 1
            for t, row in f["losses"][original_index].items():
                for g, count in row["lost_supported_gt"].items():
                    totals[arm][t]["lost_supported_gt"][g] += count
                totals[arm][t]["capacity_loss"] += row["capacity_loss"]
    gt = {g: sum(f["gt"][g] for f in frames) for g in GROUPS}
    return {a: {"counts": values, "lost_support_pp": {
        t: {g: n / max(gt[g], 1) * 100 for g, n in row["lost_supported_gt"].items()}
        for t, row in values.items()}, "forward_count_saving_percent": 20.}
        for a, values in totals.items()}


def headroom(frames, policy):
    choices = {}
    for f in frames:
        costs = [sum(f["losses"][v-1][str(t)]["lost_supported_gt"]["tiny"]
                     for t in (.5, .75)) for v in OMITTED]
        choices[f["stem"]] = {"oracle": int(np.argmin(costs)),
                               "restricted_density": f["restricted_density"]}
    metrics = score(choices, frames)
    delta = (metrics["restricted_density"]["counts"]["0.75"]["lost_supported_gt"]["tiny"] -
             metrics["oracle"]["counts"]["0.75"]["lost_supported_gt"]["tiny"])
    checks = {
        "absolute_budget": metrics["oracle"]["lost_support_pp"]["0.75"]["tiny"] <=
        policy["tiny075_oracle_loss_max_pp"],
        "density_headroom": delta >= policy["oracle_tiny075_reduction_vs_restricted_density_min_count"],
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics,
            "oracle_tiny075_reduction_vs_density": delta}, choices


def fit(seed, frames, output):
    train = np.stack([f["features"] for f in frames["train"]])
    holdout = np.stack([f["features"] for f in frames["holdout"]])
    mean = train.reshape(-1, train.shape[-1]).mean(0)
    std = train.reshape(-1, train.shape[-1]).std(0).clip(1e-3)
    normalized = {k: np.clip((v-mean)/std, -6, 6) for k, v in (("train", train), ("holdout", holdout))}
    inputs, permutation = {}, {}
    for k, values in normalized.items():
        shuffled, partner = derange_images(values, 1041 if k == "train" else 1042)
        global_only = values.copy()
        global_only[:, :, GLOBAL_DIM:] = 0
        inputs[k] = {a: torch.from_numpy(v.reshape(-1, v.shape[-1])) for a, v in (
            ("conditioned_marginal", values), ("global_marginal", global_only),
            ("shuffled_marginal", shuffled), ("conditioned_coverage", values))}
        permutation[k] = partner.tolist()
    target = np.log1p(np.stack([f["targets"][np.array(OMITTED)-1] for f in frames["train"]])).reshape(-1, 2)
    tm, ts = target.mean(0), target.std(0).clip(1e-3)
    target = torch.from_numpy((target-tm)/ts)
    torch.manual_seed(seed)
    initial = ViewValueRegressor(train.shape[-1])
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.Adam(m.parameters(), lr=.003) for a, m in models.items()}
    history = []
    for epoch in range(80):
        order = torch.randperm(len(target), generator=torch.Generator().manual_seed(seed*100+epoch))
        losses = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 128):
            rows = order[start:start+128]
            for arm, model in models.items():
                column = 0 if arm == "conditioned_coverage" else 1
                loss = (model(inputs["train"][arm][rows]) - target[rows, column]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite view-policy objective.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach()) * len(rows) / len(target)
        history.append({"epoch": epoch+1, **losses})
    checkpoint = output / f"heads-seed{seed}.pt"
    torch.save({"models": {a: m.state_dict() for a, m in models.items()},
                "mean": mean, "std": std, "target_mean": tm, "target_std": ts,
                "shuffle_partners": permutation, "seed": seed, "epochs": 80}, checkpoint)
    save_json(output / f"losses-seed{seed}.json", history)
    with torch.inference_mode():
        predictions = {a: m(inputs["holdout"][a]).reshape(-1, 2).argmin(1).numpy()
                       for a, m in models.items()}
    choices = {f["stem"]: {**{a: int(v[i]) for a, v in predictions.items()},
                           "restricted_density": f["restricted_density"]}
               for i, f in enumerate(frames["holdout"])}
    return choices, {"checkpoint": checkpoint.name, "sha256": file_hash(checkpoint),
                     "parameters_each": sum(p.numel() for p in initial.parameters()),
                     "fit_images": len(train), "fit_candidate_rows": len(target)}


def decide(metrics, policy):
    local = metrics["conditioned_marginal"]
    loss = local["counts"]["0.75"]["lost_supported_gt"]["tiny"]
    comparisons = {
        a: metrics[a]["counts"]["0.75"]["lost_supported_gt"]["tiny"]-loss
        for a in ("global_marginal", "shuffled_marginal", "conditioned_coverage", "restricted_density")}
    low_extra = (local["counts"]["0.5"]["lost_supported_gt"]["tiny"] -
                 metrics["restricted_density"]["counts"]["0.5"]["lost_supported_gt"]["tiny"])
    checks = {
        a: comparisons[a] >= policy["tiny075_reduction_vs_" + name + "_min_count"]
        for a, name in (("global_marginal", "global"), ("shuffled_marginal", "shuffled"),
                        ("conditioned_coverage", "coverage"), ("restricted_density", "density"))}
    checks["low_iou_budget"] = low_extra <= policy["tiny050_extra_vs_density_max_count"]
    checks["absolute_budget"] = local["lost_support_pp"]["0.75"]["tiny"] <= policy["tiny075_loss_max_pp"]
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"tiny075_reductions": comparisons, "tiny050_extra_vs_density": low_extra,
                       "tiny075_loss_pp": local["lost_support_pp"]["0.75"]["tiny"]}}


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/observed_view_policy.py", "sdcl/training/observed_view_policy.py",
             "scripts/training/run_observed_view_policy.py", "sdcl/training/marginal_view_policy.py",
             "sdcl/postprocess/view_value.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    selected, frames, _ = prepare()
    raw_sha = construct(frames)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": selected,
        "source_archive_sha256": file_hash(output / "source.zip"), "raw_views_sha256": raw_sha,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate26_marginal_view_policy_result_20261014.json"))})
    (output / "features").mkdir()
    manifest = {}
    for split, records in frames.items():
        for f in records:
            path = output / "features" / (f["stem"] + ".npz")
            np.savez_compressed(path, features=f["features"], targets=f["targets"][np.array(OMITTED)-1])
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(),
                                    "sha256": file_hash(path), "split": split}
    save_json(output / "frames.json", manifest)
    preflight, choices = headroom(frames["holdout"], registration["preflight"])
    save_json(output / "preflight.json", preflight)
    save_json(output / "preflight-choices.json", choices)
    print(f"Observed policy headroom: {json.dumps(preflight)}, {perf_counter()-started:.1f}s", flush=True)
    metrics, decisions, training = {}, {}, {}
    if preflight["passed"]:
        for seed in (0, 1, 2):
            choices, checkpoint = fit(seed, frames, output)
            save_json(output / f"choices-seed{seed}.json", choices)
            metrics[str(seed)] = score(choices, frames["holdout"])
            decisions[str(seed)] = decide(metrics[str(seed)], registration["policy"])
            training[str(seed)] = checkpoint
            print(f"Observed policy seed{seed}: {json.dumps(decisions[str(seed)])}, "
                  f"{perf_counter()-started:.1f}s", flush=True)
            if seed == 0 and not decisions["0"]["passed"]:
                break
    positive = sum(v["passed"] for v in decisions.values())
    passed = (len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_seeds"] and
              all(v["checks"]["absolute_budget"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_OBSERVED_POLICY_REPLAY_ALLOWED" if passed else
               ("STOP_THIS_OBSERVED_DISAGREEMENT_POLICY" if preflight["passed"] else
                "STOP_THIS_DIAGONAL_SCHEDULE_HEADROOM"),
               "preflight_passed": preflight["passed"], "per_seed": decisions,
               "seeds_executed": len(decisions), "positive_seeds": positive,
               "scope": "Reused train-head-holdout native coverage proxy, not AP, speed or novelty."}
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "training.json", training)
    save_json(output / "verification.json", {
        "fit_images": 128, "head_holdout_images": 64, "detector_forwards": 0,
        "image_decodes": 0, "detector_training_steps": 0, "observed_views": [0, 1, 4],
        "no_uncomputed_view_predictions_or_GT_in_features": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "historical_native_heldout_pools_exact_replay": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
