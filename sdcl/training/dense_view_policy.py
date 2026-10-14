"""Matched sparse/dense view-policy screen with a frozen native detector."""

from copy import deepcopy
import json
from time import perf_counter

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.dense_view_evidence import dense_features, map_anchor_centers
from sdcl.postprocess.rescue import save_json
from sdcl.postprocess.view_value import ViewValueRegressor
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.marginal_view_policy import BASE, prepare
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate27_dense_view_policy_registration_20261014.json"
OUTPUT = "output/analysis/dense-view-policy-fixed-20261014"
ARMS = ("dense_marginal", "dense_coverage", "sparse_marginal")
GROUPS = ("all", "tiny", "small_all")


def extend_inputs(frames, device, output):
    base = project_path(BASE)
    _, plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    raw = {r["image"]: r for r in
           (json.loads(line) for line in (base / "raw_views.jsonl").read_text().splitlines())}
    snapshots = {r["stem"]: r for rows in plan["records"].values() for r in rows}
    (output / "features").mkdir()
    manifest = {}
    extractor = FrozenViews(device)
    try:
        with torch.inference_mode():
            for split, records in frames.items():
                for index, frame in enumerate(records, 1):
                    stem = frame["stem"]
                    path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                    if file_hash(path) != snapshots[stem]["image_sha256"]:
                        raise ValueError("Frozen image changed.")
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("Frozen image unreadable.")
                    prediction, _ = extractor.predict(image)
                    expected = np.asarray(raw[stem]["views"][0], np.float32).reshape(-1, 6)
                    if not np.array_equal(prediction["control"], expected):
                        raise ValueError("Frozen global raw replay differs.")
                    head = extractor.head
                    centers = (head.anchors.T * head.strides.T).float().cpu().numpy()
                    centers = map_anchor_centers(centers, raw[stem]["windows"][0][2:])
                    levels = np.concatenate([
                        np.full(f.shape[2] * f.shape[3], i) for i, f in enumerate(extractor.raw["feats"])])
                    probability = extractor.decoded[0, 4:14].T.float().cpu().numpy()
                    extra = dense_features(probability, centers, levels, raw[stem]["windows"],
                                           raw[stem]["windows"][0][2:])
                    frame["features"] = np.concatenate((frame["features"], extra), axis=1)
                    target = output / "features" / f"{stem}.npz"
                    np.savez_compressed(target, features=frame["features"])
                    manifest[stem] = {"file": target.relative_to(output).as_posix(),
                                      "sha256": file_hash(target), "split": split}
                    if index == 1 or index % 32 == 0 or index == len(records):
                        print(f"Dense policy input {split}: {index}/{len(records)}", flush=True)
    finally:
        extractor.close()
    return manifest


def fit(seed, frames, output):
    features = np.concatenate([f["features"] for f in frames["train"]])
    target = np.log1p(np.concatenate([f["targets"] for f in frames["train"]]))
    mean, std = features.mean(0), features.std(0).clip(1e-3)
    target_mean, target_std = target.mean(0), target.std(0).clip(1e-3)
    x = torch.from_numpy(np.clip((features - mean) / std, -6, 6))
    y = torch.from_numpy((target - target_mean) / target_std)
    torch.manual_seed(seed)
    initial = ViewValueRegressor(x.shape[1])
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.Adam(m.parameters(), lr=.003) for a, m in models.items()}
    rng = np.random.default_rng(seed)
    sparse = x.clone()
    sparse[:, 61:] = 0
    for epoch in range(80):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            rows = order[start:start+128]
            for arm, model in models.items():
                column = 0 if arm == "dense_coverage" else 1
                inputs = sparse if arm == "sparse_marginal" else x
                optimizers[arm].zero_grad(set_to_none=True)
                loss = (model(inputs[rows]) - y[rows, column]).square().mean()
                loss.backward()
                optimizers[arm].step()
    checkpoint = output / f"heads-seed{seed}.pt"
    torch.save({
        "models": {a: m.state_dict() for a, m in models.items()},
        "mean": mean, "std": std, "target_mean": target_mean, "target_std": target_std,
        "seed": seed, "epochs": 80,
    }, checkpoint)
    choices = {}
    with torch.inference_mode():
        for frame in frames["holdout"]:
            xh = torch.from_numpy(np.clip((frame["features"] - mean) / std, -6, 6))
            sh = xh.clone()
            sh[:, 61:] = 0
            choices[frame["stem"]] = {
                **{arm: int(model(sh if arm == "sparse_marginal" else xh).argmin())
                   for arm, model in models.items()},
                "global_density_one": frame["density_omitted_index"],
            }
    return choices, {"checkpoint": checkpoint.name, "sha256": file_hash(checkpoint),
                     "parameters_per_head": sum(p.numel() for p in initial.parameters())}


def score(choices, frames):
    totals = {arm: {t: dict.fromkeys(GROUPS, 0) for t in ("0.5", "0.75")}
              for arm in (*ARMS, "global_density_one")}
    for frame in frames:
        for arm, index in choices[frame["stem"]].items():
            for threshold, row in frame["losses"][index].items():
                for group, n in row["lost_supported_gt"].items():
                    totals[arm][threshold][group] += n
    counts = {g: sum(f["gt"][g] for f in frames) for g in GROUPS}
    return {a: {
        "lost_supported_gt": values,
        "lost_support_pp": {t: {g: n / max(counts[g], 1) * 100 for g, n in row.items()}
                            for t, row in values.items()},
        "forward_count_saving_percent": 20.,
    } for a, values in totals.items()}


def decide(metrics, policy):
    local = metrics["dense_marginal"]["lost_supported_gt"]
    reductions = {a: metrics[a]["lost_supported_gt"]["0.75"]["tiny"] - local["0.75"]["tiny"]
                  for a in ("dense_coverage", "sparse_marginal", "global_density_one")}
    values = {
        "tiny075_reductions": reductions,
        "tiny050_extra_vs_density": local["0.5"]["tiny"] -
        metrics["global_density_one"]["lost_supported_gt"]["0.5"]["tiny"],
        "tiny075_loss_pp": metrics["dense_marginal"]["lost_support_pp"]["0.75"]["tiny"],
    }
    checks = {
        **{a + "_gain": reductions[a] >= policy[key] for a, key in (
            ("dense_coverage", "tiny075_reduction_vs_dense_coverage_min_count"),
            ("sparse_marginal", "tiny075_reduction_vs_sparse_marginal_min_count"),
            ("global_density_one", "tiny075_reduction_vs_density_min_count"))},
        "low_iou_budget": values["tiny050_extra_vs_density"] <= policy["tiny050_extra_vs_density_max_count"],
        "absolute_loss_budget": values["tiny075_loss_pp"] <= policy["tiny075_loss_max_pp"],
    }
    return {"passed": all(checks.values()), "checks": checks, "values": values}


def evaluate(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("An available CUDA index is required.")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/dense_view_evidence.py", "sdcl/training/dense_view_policy.py",
             "scripts/training/run_dense_view_policy.py", "sdcl/training/marginal_view_policy.py",
             "sdcl/postprocess/view_value.py", "sdcl/training/class_margin_probe.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    selected, frames, _ = prepare()
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": selected,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "candidate26_result_sha256": file_hash(project_path(
            "docs/experiments/candidate26_marginal_view_policy_result_20261014.json")),
    })
    manifest = extend_inputs(frames, device, output)
    save_json(output / "frames.json", manifest)
    metrics, decisions, checkpoints = {}, {}, {}
    for seed in (0, 1, 2):
        choices, checkpoint = fit(seed, frames, output)
        save_json(output / f"choices-seed{seed}.json", choices)
        metrics[str(seed)] = score(choices, frames["holdout"])
        decisions[str(seed)] = decide(metrics[str(seed)], registration["policy"])
        checkpoints[str(seed)] = checkpoint
        print(f"Dense view policy seed{seed}: {json.dumps(decisions[str(seed)])}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions[str(seed)]["passed"]:
            break
    positive = sum(v["passed"] for v in decisions.values())
    passed = len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_seeds"]
    passed &= all(v["checks"]["absolute_loss_budget"] for v in decisions.values())
    verdict = {"decision": "BOUNDED_DENSE_VIEW_POLICY_REPLAY_ALLOWED" if passed else
               "STOP_THIS_DENSE_SUMMARY_VIEW_POLICY",
               "per_seed": decisions, "positive_seeds": positive, "seeds_executed": len(decisions),
               "scope": "Reused train head-holdout coverage proxy; not AP, latency or novelty proof."}
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "training.json", checkpoints)
    save_json(output / "verification.json", {
        "fit_images": 128, "head_heldout_images": 64,
        "fullimage_forwards_excluding_warmup": len(manifest), "local_view_forwards": 0,
        "detector_training_steps": 0, "raw_global_replay_exact": True,
        "no_val_no_testdev_no_hyperparameter_or_threshold_search": True,
        "dense_inputs_include_prethreshold_nonargmax_probabilities": True,
        "no_local_predictions_or_annotations_in_features": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
