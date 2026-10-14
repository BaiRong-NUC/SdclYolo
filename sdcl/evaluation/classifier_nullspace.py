"""Matched linear foreground-information screen in a frozen classifier nullspace."""

import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate48_classifier_nullspace_registration_20261014.json"
OUTPUT = "output/analysis/classifier-nullspace-fixed-20261014"
NATIVE = "output/analysis/native-regression-residual-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("null", "row", "shuffled_null", "logits_only")
GROUPS = ("all", "tiny", "small_all")


def projectors(weight):
    weight = np.asarray(weight, np.float64)
    _, singular, vt = np.linalg.svd(weight, full_matrices=False)
    rank = int((singular > singular[0]*max(weight.shape)*np.finfo(float).eps).sum())
    row = vt[:rank].T@vt[:rank]
    null = np.eye(weight.shape[1])-row
    error = np.linalg.norm(weight@null)/max(np.linalg.norm(weight), 1e-12)
    if error > 1e-12:
        raise ValueError("Classifier null projection identity failed.")
    return row, null, rank, error


def levels(indices):
    indices = np.asarray(indices)
    return np.where(indices < 14400, 0, np.where(indices < 18000, 1, 2))


def shuffle_null(values, classes, chosen, active, stem):
    permutation = np.arange(len(values))
    level = levels(chosen[:, 1])
    groups = sorted({(int(classes[i]), int(chosen[i, 0]), int(level[i])) for i in np.flatnonzero(active)})
    for cls, view, lev in groups:
        rows = np.flatnonzero(active & (classes == cls) & (chosen[:, 0] == view) & (level == lev))
        seed = int.from_bytes(hashlib.sha256(f"{stem}:{cls}:{view}:{lev}:nullspace-v1".encode()).digest()[:8], "little")
        permutation[rows] = np.random.default_rng(seed).permutation(rows)
    return values[permutation], permutation


def logistic_value(theta, inputs, offset, labels):
    z = offset+inputs@theta[:-1]+theta[-1]
    value = np.mean(np.logaddexp(0, z)-labels*z)+.005*np.dot(theta[:-1], theta[:-1])
    residual = (expit(z)-labels)/len(z)
    gradient = np.r_[inputs.T@residual+.01*theta[:-1], residual.sum()]
    return float(value), gradient


def fit(inputs, offset, labels):
    initial = np.zeros(inputs.shape[1]+1)
    result = minimize(logistic_value, initial, args=(inputs, offset, labels), jac=True,
                      method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-6})
    value, gradient = logistic_value(result.x, inputs, offset, labels)
    valid = bool(result.success and np.max(np.abs(gradient)) <= 1e-5)
    return result.x, {"success": valid, "solver_success": bool(result.success), "iterations": int(result.nit),
                      "objective": value, "gradient_inf": float(np.max(np.abs(gradient))), "message": str(result.message)}


def labels_for(reference, metadata, confidence):
    gt = np.asarray([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
    classes = np.asarray([o["class_id"] for o in metadata["valid_objects"]], int)
    iou = box_iou_matrix(gt, reference[:, :4])
    _, used = match_at_threshold(iou, classes, reference, confidence)
    duplicate = ~used & ((iou >= .5) & (classes[:, None] == reference[None, :, 5])).any(0)
    return used, duplicate


def normalize(frames):
    fit_frames = frames["train"]
    native_logits = np.concatenate([f["logits"][f["active"]] for f in fit_frames])
    mean, std = native_logits.mean(0), np.maximum(native_logits.std(0), 1e-6)
    stats = {"logits_mean": mean.tolist(), "logits_std": std.tolist(), "extra": {}}
    for arm in ("null", "row"):
        stats["extra"][arm] = []
        for level in range(3):
            values = np.concatenate([f[arm][f["active"] & (f["level"] == level)] for f in fit_frames])
            if not len(values):
                raise ValueError("Insufficient training level for fixed normalization.")
            center = values.mean(0)
            scale = max(float(np.sqrt(np.mean((values-center)**2))), 1e-6)
            stats["extra"][arm].append({"mean": center.tolist(), "rms": scale, "rows": len(values)})
    for records in frames.values():
        for f in records:
            n = len(f["reference"])
            view_hot, level_hot = np.zeros((n, 5)), np.zeros((n, 3))
            rows = np.flatnonzero(f["active"])
            view_hot[rows, f["chosen"][rows, 0]] = 1
            level_hot[rows, f["level"][rows]] = 1
            base = np.concatenate([f["geometry"], (f["logits"]-mean)/std, view_hot, level_hot], 1)
            f["inputs"] = {}
            for arm in ARMS:
                extra = np.zeros((n, 128))
                if arm != "logits_only":
                    family = "row" if arm == "row" else "null"
                    raw = f[family] if arm != "shuffled_null" else f["null"][f["permutation"]]
                    for level in range(3):
                        item = stats["extra"][family][level]
                        keep = f["active"] & (f["level"] == level)
                        extra[keep] = (raw[keep]-item["mean"])/item["rms"]
                f["inputs"][arm] = np.concatenate([base, extra], 1)
                if f["inputs"][arm].shape != (n, 214):
                    raise ValueError("Unexpected information-head input dimensions.")
    return stats


def summarize(frames, coefficients):
    per_image, manifest_arrays = [], {}
    for f in frames:
        reference = f["reference"]
        original = np.log(np.clip(reference[:, 4], 1e-6, 1-1e-6)/(1-np.clip(reference[:, 4], 1e-6, 1-1e-6)))
        labels, duplicate = labels_for(reference, f["metadata"], .25)
        areas = box_areas(reference[:, :4])*(640/max(f["metadata"]["original_size"]))**2
        eligible = f["active"] & (reference[:, 4] >= .25)
        logits = {a: original+f["inputs"][a]@theta[:-1]+theta[-1] for a, theta in coefficients.items()}
        losses = {a: np.logaddexp(0, z)-labels*z for a, z in logits.items()}
        trace = {}
        arrays = {"reference": reference, "active": f["active"], "labels": labels,
                  "duplicate": duplicate, "eligible": eligible, "areas": areas,
                  **{a+"_logits": z for a, z in logits.items()}}
        for group in GROUPS:
            all_rows = eligible & group_mask(areas, group)
            selected = all_rows & ~duplicate
            arrays[group+"_selected"] = selected
            trace[group] = {
                "rows": int(selected.sum()), "tp": int((selected & labels).sum()),
                "nonduplicate_fp": int((selected & ~labels).sum()),
                "full_fp": int((all_rows & ~labels).sum()), "duplicate_fp": int((all_rows & duplicate).sum()),
                "loss_sums": {a: float(loss[selected].sum()) for a, loss in losses.items()},
            }
        per_image.append(trace)
        manifest_arrays[f["stem"]] = arrays
    metrics = {}
    draws = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    for group in GROUPS:
        traces = [t[group] for t in per_image]
        counts = np.array([t["rows"] for t in traces])
        sums = {a: np.array([t["loss_sums"][a] for t in traces]) for a in ARMS}
        labels = np.concatenate([v["labels"][v[group+"_selected"]] for v in manifest_arrays.values()])
        scores = {a: np.concatenate([v[a+"_logits"][v[group+"_selected"]] for v in manifest_arrays.values()]) for a in ARMS}
        denominator = draws@counts
        difference = draws@(sums["logits_only"]-sums["null"])
        bootstrap = difference[denominator > 0]/denominator[denominator > 0]
        metrics[group] = {
            "rows": int(counts.sum()), "tp": sum(t["tp"] for t in traces),
            "nonduplicate_fp": sum(t["nonduplicate_fp"] for t in traces),
            "full_fp": sum(t["full_fp"] for t in traces), "duplicate_fp": sum(t["duplicate_fp"] for t in traces),
            "images": int((counts > 0).sum()),
            "NLL": {a: float(v.sum()/counts.sum()) for a, v in sums.items()},
            "AUROC": {a: float(roc_auc_score(labels, score)) for a, score in scores.items()},
            "null_NLL_improvement_over_logits_image_bootstrap95ci": np.quantile(bootstrap, [.025, .975]).tolist(),
        }
    return metrics, per_image, manifest_arrays


def decide(metrics, policy, numerical=True):
    tiny = metrics["tiny"]
    gains = {a: tiny["NLL"][a]-tiny["NLL"]["null"] for a in ("logits_only", "row", "shuffled_null")}
    checks = {
        "numerical_convergence": numerical,
        "tp": tiny["tp"] >= policy["tiny_tp_min"],
        "nonduplicate_fp": tiny["nonduplicate_fp"] >= policy["tiny_nonduplicate_fp_min"],
        "images": tiny["images"] >= policy["tiny_images_min"],
        "NLL_vs_logits": gains["logits_only"] >= policy["tiny_NLL_improvement_vs_logits_min"],
        "NLL_vs_row": gains["row"] >= policy["tiny_NLL_improvement_vs_row_min"],
        "NLL_vs_shuffle": gains["shuffled_null"] >= policy["tiny_NLL_improvement_vs_shuffle_min"],
        "AUROC": tiny["AUROC"]["null"]-tiny["AUROC"]["logits_only"] >= policy["tiny_AUROC_improvement_vs_logits_min"],
        "interval": tiny["null_NLL_improvement_over_logits_image_bootstrap95ci"][0] > policy["tiny_NLL_improvement_bootstrap_lower95_exclusive"],
        "all_NLL": metrics["all"]["NLL"]["logits_only"]-metrics["all"]["NLL"]["null"] >= policy["all_NLL_improvement_vs_logits_min"],
    }
    return {"decision": "BOUNDED_NULLSPACE_SCORE_PILOT_ALLOWED" if all(checks.values()) else "STOP_THIS_CLASSIFIER_NULLSPACE_LINEAR_FOREGROUND_RECIPE",
            "checks": checks, "tiny_NLL_gains": gains,
            "scope": "Representation discrimination only, not AP, OOD classification, independent evidence or innovation."}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Available CUDA index required for fixed feature extraction.")
    output = project_path(output)
    if output.exists():
        raise FileExistsError("Refuse an existing nullspace experiment directory.")
    registration = json.loads(project_path(REGISTRATION).read_text())
    frames, prior, heads = load_frames()
    native = project_path(NATIVE)
    _, native_plan = verify_record(native, "docs/experiments/candidate38_native_regression_features_result_20261014.json")
    manifest38 = json.loads((native/"features.json").read_text())
    _, raw_plan = verify_record(project_path(RAW), "docs/experiments/candidate12_class_choice_result_20261014.json")
    raw_path = project_path(RAW)/"raw_views.jsonl"
    raw = {r["image"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    records = {r["stem"]: r for split in ("train", "holdout") for r in raw_plan["records"][split]}
    paths = (REGISTRATION, "sdcl/evaluation/classifier_nullspace.py",
             "scripts/evaluation/diagnose_classifier_nullspace.py", "tests/test_classifier_nullspace.py",
             "sdcl/training/class_margin_probe.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/training/marginal_class_intervention.py", "sdcl/evaluation/size_analysis.py",
             "sdcl/postprocess/localization.py", "sdcl/evaluation/slicing_screen.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    output.mkdir(parents=True)
    (output/"features").mkdir()
    archive_source(output, hashes)
    extractor = FrozenViews(device)
    weight, bias = extractor.weight.numpy().astype(float), extractor.bias.numpy().astype(float)
    decompositions = [projectors(w) for w in weight]
    np.savez_compressed(output/"projection.npz", weight=weight, bias=bias,
                        row=np.stack([r[0] for r in decompositions]), null=np.stack([r[1] for r in decompositions]))
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "source_archive_sha256": file_hash(output/"source.zip"),
        "image_selection": prior["image_selection"], "heads": heads, "raw_views_sha256": file_hash(raw_path),
        "prior_association_result_sha256": file_hash(project_path(
            "docs/experiments/candidate38_native_regression_features_result_20261014.json")),
        "projection_rank": [r[2] for r in decompositions], "relative_WN_errors": [r[3] for r in decompositions],
    })
    manifest, forwards, maximum_reconstruction = {}, 0, 0.
    try:
        with torch.inference_mode():
            for split, fs in frames.items():
                for f in fs:
                    path = native/manifest38[f["stem"]]["file"]
                    if file_hash(path) != manifest38[f["stem"]]["sha256"]:
                        raise ValueError("Prior native associations changed.")
                    with np.load(path, allow_pickle=False) as c:
                        chosen, active = c["chosen_native"].copy(), c["active"].copy()
                    f["chosen"], f["active"], f["level"] = chosen, active, levels(chosen[:, 1])
                    f["reference"] = apply_actions(f["candidates"], f["boxes"], f["regret"])
                    f["geometry"] = f["features"][np.arange(len(active)), f["regret"]].astype(float)
                    image_path = project_path(f"data/visdrone/images/train/{f['stem']}.jpg")
                    if file_hash(image_path) != records[f["stem"]]["image_sha256"]:
                        raise ValueError("Frozen feature image changed.")
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError("Unreadable fixed image.")
                    feature, logits = np.zeros((len(active), 128)), np.zeros((len(active), 10))
                    for view, window in enumerate(raw[f["stem"]]["windows"]):
                        x1, y1, x2, y2 = window
                        prediction, _ = extractor.predict(image[y1:y2, x1:x2].copy())
                        kept = map_tile_predictions(prediction["control"], window) if view else prediction["control"]
                        if not np.array_equal(kept, np.asarray(raw[f["stem"]]["views"][view], np.float32).reshape(-1, 6)):
                            raise ValueError("Frozen five-view raw FP16 replay differs.")
                        rows = np.flatnonzero(chosen[:, 0] == view)
                        if len(rows):
                            native_features = torch.cat([extractor.features[i].flatten(2) for i in range(3)], 2)[0].T
                            indices = torch.as_tensor(chosen[rows, 1], device=native_features.device)
                            feature[rows] = native_features[indices].float().cpu().numpy()
                            logits[rows] = extractor.raw["scores"][0, :, indices].T.float().cpu().numpy()
                        forwards += 1
                    row_component, null_component = np.zeros_like(feature), np.zeros_like(feature)
                    for lev, (row, null, _, _) in enumerate(decompositions):
                        keep = active & (f["level"] == lev)
                        row_component[keep] = feature[keep]@row
                        null_component[keep] = feature[keep]@null
                    reconstruction = np.linalg.norm(feature-row_component-null_component)/max(np.linalg.norm(feature), 1e-12)
                    maximum_reconstruction = max(maximum_reconstruction, reconstruction)
                    if reconstruction > 1e-12:
                        raise ValueError("Feature row/null reconstruction failed.")
                    f["row"], f["null"], f["logits"] = row_component, null_component, logits
                    _, permutation = shuffle_null(null_component, f["reference"][:, 5], chosen, active, f["stem"])
                    f["permutation"] = permutation
                    labels, _ = labels_for(f["reference"], f["metadata"], .05)
                    f["fit_labels"] = labels
                    dest = output/"features"/(f["stem"]+".npz")
                    np.savez_compressed(dest, features=feature, logits=logits, chosen=chosen, active=active,
                                        row=row_component, null=null_component, permutation=permutation, fit_labels=labels)
                    manifest[f["stem"]] = {"file": dest.relative_to(output).as_posix(), "sha256": file_hash(dest),
                                           "split": split, "image_sha256": records[f["stem"]]["image_sha256"],
                                           "association_source_sha256": manifest38[f["stem"]]["sha256"]}
                    if len(manifest) == 1 or len(manifest) % 32 == 0:
                        print(f"Full classifier nullspace features: {len(manifest)}/192", flush=True)
    finally:
        extractor.close()
    stats = normalize(frames)
    save_json(output/"features.json", manifest)
    save_json(output/"normalization.json", stats)
    score = np.concatenate([f["reference"][f["active"], 4] for f in frames["train"]])
    offset = np.log(np.clip(score, 1e-6, 1-1e-6)/(1-np.clip(score, 1e-6, 1-1e-6)))
    labels = np.concatenate([f["fit_labels"][f["active"]] for f in frames["train"]])
    coefficients, training = {}, {}
    for arm in ARMS:
        values = np.concatenate([f["inputs"][arm][f["active"]] for f in frames["train"]])
        coefficients[arm], training[arm] = fit(values, offset, labels)
        print(f"Nullspace linear {arm}: {training[arm]}", flush=True)
    np.savez_compressed(output/"coefficients.npz", **coefficients)
    metrics, per_image, prediction_arrays = summarize(frames["holdout"], coefficients)
    (output/"traces").mkdir()
    trace_manifest = {}
    for stem, values in prediction_arrays.items():
        path = output/"traces"/(stem+".npz")
        np.savez_compressed(path, **values)
        trace_manifest[stem] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
    decision = decide(metrics, registration["policy"], all(v["success"] for v in training.values()))
    verification = {
        "fit_images": 128, "head_holdout_images": 64, "detector_forwards_excluding_warmup": forwards,
        "image_decodes": 192, "detector_updates": 0, "linear_models_fit": 4,
        "full128_classification_features_and_all10_logits": True,
        "all_five_view_raw_replays_exact": True, "prior_native_associations_and_masks_unchanged": True,
        "GT_excluded_from_features_associations_and_inference_scores": True,
        "training_targets_consume_only128fit_images": True,
        "maximum_relative_row_null_reconstruction_error": maximum_reconstruction,
        "no_predictions_changed_or_AP_evaluator_run": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started,
    }
    for name, value in (("training.json", training), ("metrics.json", metrics), ("decision.json", decision),
                        ("traces.json", trace_manifest), ("per_image.json", per_image), ("verification.json", verification)):
        save_json(output/name, value)
    result = {"experiment": registration["experiment"], "date": "2026-10-14", "output": str(output),
              "decision": decision, "metrics": metrics, "training": training, "verification": verification,
              "artifacts_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    save_json(output/"result.json", result)
    return result
