"""Cross-class image-shared residual information with fixed oracle controls."""

import json
import os
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash, group_mask
from sdcl.postprocess.native_regression_residual import correct_coordinates
from sdcl.postprocess.rescue import save_json
from sdcl.training.geometry_residual_confirmation import prepare
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate41_shared_residual_bias_registration_20261014.json"
OUTPUT = "output/analysis/shared-residual-bias-fixed-20261014"
ARMS = ("class_mean", "same_image_oracle", "wrong_image_oracle")


def huber(error):
    absolute = np.abs(error)
    return np.where(absolute < .02, .5*absolute**2/.02, absolute-.01).mean(-1)


def class_means(frames):
    sums, counts = np.zeros((10, 4)), np.zeros(10, int)
    for f in frames:
        for c in range(10):
            rows = f["classes"] == c
            sums[c] += f["error"][rows].sum(0)
            counts[c] += int(rows.sum())
    return sums/np.maximum(counts[:, None], 1), counts


def donor_prediction(query_classes, donor_classes, donor_error, means):
    prediction = means[query_classes].copy()
    available = np.zeros(len(query_classes), bool)
    count = np.zeros(len(query_classes), int)
    centered = donor_error-means[donor_classes]
    for c in np.unique(query_classes):
        donors = donor_classes != c
        rows = query_classes == c
        count[rows] = int(donors.sum())
        if donors.sum() >= 5:
            prediction[rows] += centered[donors].mean(0)
            available[rows] = True
    return np.clip(prediction, -.1, .1), available, count


def summarize(totals):
    result = {}
    draws = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    for g, rows in totals.items():
        count = np.array([r["count"] for r in rows])
        loss = {a: np.array([r[a] for r in rows]) for a in ARMS}
        denominator = max(int(count.sum()), 1)
        means = {a: float(x.sum()/denominator) for a, x in loss.items()}
        ratio = lambda baseline: (means[baseline]-means["same_image_oracle"])/max(means[baseline], 1e-12)
        base = draws@loss["class_mean"]
        delta = draws@(loss["class_mean"]-loss["same_image_oracle"])
        ci = np.quantile(delta[base > 0]/base[base > 0], [.025, .975]).tolist() if np.any(base > 0) else [None, None]
        result[g] = {"queries": int(count.sum()), "represented_images": int((count > 0).sum()),
                     "mean_smooth_l1": means, "relative_loss_reduction_vs_class": ratio("class_mean"),
                     "relative_loss_reduction_vs_wrong_image": ratio("wrong_image_oracle"),
                     "paired_image_bootstrap_relative_reduction_vs_class_95ci": ci}
    return result


def decide(metrics, policy):
    row = metrics["tiny"]
    lower = row["paired_image_bootstrap_relative_reduction_vs_class_95ci"][0]
    checks = {
        "tiny_query_count": row["queries"] >= policy["tiny_queries_min"],
        "represented_images": row["represented_images"] >= policy["tiny_represented_images_min"],
        "vs_class_mean": row["relative_loss_reduction_vs_class"] >= policy["tiny_relative_loss_reduction_vs_class_min"],
        "vs_wrong_image": row["relative_loss_reduction_vs_wrong_image"] >= policy["tiny_relative_loss_reduction_vs_wrong_image_min"],
        "paired_image_interval": lower is not None and
        lower > policy["tiny_paired_image_bootstrap_vs_class_lower95_min_exclusive"],
    }
    return {"decision": "MATCHED_IMAGE_BIAS_LEARNABILITY_PROBE_ALLOWED" if all(checks.values()) else
            "STOP_THIS_CROSS_CLASS_SHARED_RESIDUAL_BIAS_RECIPE", "checks": checks,
            "scope": "Optimistic GT-assisted residual information only; not deployment, AP or innovation."}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("CUDA replay of the frozen geometry control required.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"frames").mkdir()
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/evaluation/shared_residual_bias.py",
             "scripts/evaluation/diagnose_shared_residual_bias.py",
             "sdcl/training/geometry_residual_confirmation.py",
             "sdcl/postprocess/native_regression_residual.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads, _, model = prepare()
    model = model.to("cuda:"+str(device)).eval()
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "heads": heads,
        "source_archive_sha256": file_hash(output/"source.zip"), "image_selection": prior["image_selection"],
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json"))})
    packed = {"train": [], "holdout": []}
    with torch.inference_mode():
        for split, values in frames.items():
            for f in values:
                delta = np.zeros((len(f["reference"]), 4))
                delta[f["active"]] = model(torch.from_numpy(f["input"][f["active"]]).to(
                    "cuda:"+str(device))).cpu().numpy()
                prediction = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
                if split == "holdout" and not np.array_equal(prediction, f["seed0_geometry"]):
                    raise ValueError("Frozen geometry replay differs.")
                rows = np.flatnonzero(f["active"] & (f["owner"] >= 0))
                if len(np.unique(f["owner"][rows])) != len(rows):
                    raise ValueError("A GT owner cannot donate through duplicate predictions.")
                gt = np.array([o["box_xyxy"] for o in f["metadata"]["valid_objects"]], float).reshape(-1, 4)
                error = np.clip((gt[f["owner"][rows]]-prediction[rows, :4]) /
                                np.maximum(f["scale"][rows], 1e-6), -.1, .1)
                packed[split].append({"stem": f["stem"], "classes": f["candidates"][rows, 5].astype(int),
                                      "error": error, "areas": f["areas"][rows], "owners": f["owner"][rows]})
    means, counts = class_means(packed["train"])
    save_json(output/"class_means.json", {"means": means.tolist(), "train_class_counts": counts.tolist()})
    totals = {g: [] for g in ("all", "tiny", "small_all")}
    manifest = {}
    for index, f in enumerate(packed["holdout"]):
        wrong = packed["holdout"][(index+1) % 64]
        same, valid_same, count_same = donor_prediction(f["classes"], f["classes"], f["error"], means)
        other, valid_wrong, count_wrong = donor_prediction(f["classes"], wrong["classes"], wrong["error"], means)
        eligible = valid_same & valid_wrong
        predictions = {"class_mean": means[f["classes"]], "same_image_oracle": same, "wrong_image_oracle": other}
        losses = {a: huber(p-f["error"]) for a, p in predictions.items()}
        for g, values in totals.items():
            include = eligible & group_mask(f["areas"], g)
            values.append({"count": int(include.sum()), **{a: float(x[include].sum()) for a, x in losses.items()}})
        path = output/"frames"/(f["stem"]+".npz")
        np.savez_compressed(path, classes=f["classes"], owners=f["owners"], areas=f["areas"], error=f["error"],
                            eligible=eligible, same_donor_counts=count_same, wrong_donor_counts=count_wrong, **predictions)
        manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path),
                              "wrong_image_stem": wrong["stem"]}
    metrics = summarize(totals)
    verdict = decide(metrics, registration["policy"])
    for name, values in (("frames", manifest), ("per_image", totals), ("metrics", metrics), ("decision", verdict)):
        save_json(output/(name+".json"), values)
    save_json(output/"verification.json", {
        "fit_images_for_class_means": 128, "head_holdout_images": 64, "detector_forwards": 0,
        "image_decodes": 0, "fitting_steps": 0, "all64_geometry_predictions_exact_replay": True,
        "class_means_training_only": True, "same_image_donors_exclude_entire_query_class": True,
        "query_owner_not_a_donor": True, "wrong_image_pairing_fixed_not_searched": True,
        "GT_USED_IN_BOTH_CONTEXT_ORACLES_NOT_DEPLOYABLE": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "metrics": metrics}
