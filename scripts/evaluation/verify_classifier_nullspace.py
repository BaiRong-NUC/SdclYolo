"""Independently audit frozen nullspace projections, consumed labels and linear fits."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.localization import apply_actions
from sdcl.training.side_composition_pilot import load_frames


ARMS = ("null", "row", "shuffled_null", "logits_only")
GROUPS = ("all", "tiny", "small_all")


def fixed_labels(reference, objects, confidence):
    gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
    classes = [o["class_id"] for o in objects]
    overlaps = np.zeros((len(gt), len(reference)))
    for g, a in enumerate(gt):
        b = reference[:, :4].astype(float)
        intersection = (np.minimum(a[2:], b[:, 2:])-np.maximum(a[:2], b[:, :2])).clip(0).prod(1)
        areas = (b[:, 2:]-b[:, :2]).clip(0).prod(1)
        overlaps[g] = intersection/np.maximum(np.prod(a[2:]-a[:2])+areas-intersection, 1e-12)
    labels, used = np.zeros(len(reference), bool), set()
    for i in sorted(range(len(reference)), key=lambda i: (-reference[i, 4], i)):
        if reference[i, 4] < confidence:
            continue
        possible = [g for g in range(len(gt)) if g not in used and classes[g] == reference[i, 5] and overlaps[g, i] >= .5]
        if possible:
            chosen = min(possible, key=lambda g: (-overlaps[g, i], g))
            used.add(chosen)
            labels[i] = True
    duplicate = np.array([not labels[i] and any(classes[g] == reference[i, 5] and overlaps[g, i] >= .5
                         for g in range(len(gt))) for i in range(len(reference))])
    return labels, duplicate


def rank_auc(labels, scores):
    # Pair-concordance form with rank ties, independent of sklearn's ROC curve.
    positive, negative = np.sort(scores[labels]), np.sort(scores[~labels])
    low = np.searchsorted(negative, positive, side="left")
    high = np.searchsorted(negative, positive, side="right")
    return float((low+.5*(high-low)).sum()/(len(positive)*len(negative)))


def verify():
    torch.set_num_threads(1)
    output = project_path("output/analysis/classifier-nullspace-fixed-20261014")
    read = lambda name: json.loads((output/name).read_text())
    result, plan = read("result.json"), read("plan.json")
    manifest, traces_manifest, stats, training = (read(n) for n in
        ("features.json", "traces.json", "normalization.json", "training.json"))
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest, name
    with zipfile.ZipFile(output/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert file_hash(project_path(name)) == digest
    with np.load(output/"projection.npz") as z:
        projection = {k: z[k].copy() for k in z.files}
    checkpoint = project_path("output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt")
    assert file_hash(checkpoint) == "eac88f7e348134d3bdaf8b8d3c93845002b634b66a4fa28bd6246abbbeba1a25"
    head = YOLO(str(checkpoint)).model.half().model[-1]
    weight = np.stack([layer[-1].weight.detach().float().numpy()[:, :, 0, 0] for layer in head.cv3])
    bias = np.stack([layer[-1].bias.detach().float().numpy() for layer in head.cv3])
    np.testing.assert_array_equal(projection["weight"], weight)
    np.testing.assert_array_equal(projection["bias"], bias)
    for level in range(3):
        w = weight[level].astype(float)
        r = np.linalg.pinv(w)@w
        np.testing.assert_allclose(projection["row"][level], r, atol=1e-14)
        np.testing.assert_allclose(projection["null"][level], np.eye(128)-r, atol=1e-14)
        assert np.linalg.norm(w@projection["null"][level])/np.linalg.norm(w) < 1e-12
        assert np.linalg.matrix_rank(w) == plan["projection_rank"][level]
    with np.load(output/"coefficients.npz") as z:
        coefficients = {k: z[k].copy() for k in z.files}
    frames, prior, heads = load_frames()
    assert heads == plan["heads"] and prior["image_selection"] == plan["image_selection"]
    native = project_path("output/analysis/native-regression-residual-fixed-20261014")
    historical = json.loads((native/"features.json").read_text())
    max_full_logit_error, shuffled_changed, shuffled_total = 0., 0, 0
    for split, fs in frames.items():
        for f in fs:
            item = manifest[f["stem"]]
            assert item["split"] == split and file_hash(output/item["file"]) == item["sha256"]
            assert item["association_source_sha256"] == historical[f["stem"]]["sha256"]
            assert file_hash(native/historical[f["stem"]]["file"]) == historical[f["stem"]]["sha256"]
            with np.load(output/item["file"]) as z:
                saved = {k: z[k].copy() for k in z.files}
            with np.load(native/historical[f["stem"]]["file"]) as z:
                np.testing.assert_array_equal(saved["chosen"], z["chosen_native"])
                np.testing.assert_array_equal(saved["active"], z["active"])
                np.testing.assert_array_equal(saved["features"][:, :64], z["extra_native_cls"])
            f["reference"] = apply_actions(f["candidates"], f["boxes"], f["regret"])
            f["geometry"] = f["features"][np.arange(len(f["reference"])), f["regret"]].astype(float)
            f.update(saved)
            f["level"] = np.array([0 if i < 14400 else 1 if i < 18000 else 2 for i in saved["chosen"][:, 1]])
            for level in range(3):
                rows = saved["active"] & (f["level"] == level)
                for component in ("null", "row"):
                    np.testing.assert_allclose(saved[component][rows], saved["features"][rows]@projection[component][level], atol=1e-14)
                np.testing.assert_allclose(saved["null"][rows]+saved["row"][rows], saved["features"][rows], atol=1e-13)
                projected_logits = saved["features"][rows]@weight[level].T+bias[level]
                if len(projected_logits):
                    max_full_logit_error = max(max_full_logit_error, float(np.abs(projected_logits-saved["logits"][rows]).max()))
            # Rebuild class/view/level partitions and every seeded permutation.
            classes = f["reference"][:, 5]
            expected = np.arange(len(classes))
            strata = {}
            for row in np.flatnonzero(saved["active"]):
                key = (int(classes[row]), int(saved["chosen"][row, 0]), int(f["level"][row]))
                strata.setdefault(key, []).append(row)
            for (cls, view, level), rows in sorted(strata.items()):
                seed = int.from_bytes(hashlib.sha256(
                    f"{f['stem']}:{cls}:{view}:{level}:nullspace-v1".encode()).digest()[:8], "little")
                expected[rows] = np.random.default_rng(seed).permutation(rows)
            np.testing.assert_array_equal(expected, saved["permutation"])
            shuffled_total += int(saved["active"].sum())
            shuffled_changed += int((expected[saved["active"]] != np.flatnonzero(saved["active"])).sum())
            fit_labels, _ = fixed_labels(f["reference"], f["metadata"]["valid_objects"], .05)
            np.testing.assert_array_equal(fit_labels, saved["fit_labels"])
    train = frames["train"]
    logits = np.concatenate([f["logits"][f["active"]] for f in train])
    np.testing.assert_allclose(logits.mean(0), stats["logits_mean"], atol=1e-14)
    np.testing.assert_allclose(np.maximum(logits.std(0), 1e-6), stats["logits_std"], atol=1e-14)
    for family in ("null", "row"):
        for level in range(3):
            values = np.concatenate([f[family][f["active"] & (f["level"] == level)] for f in train])
            center = values.mean(0)
            np.testing.assert_allclose(center, stats["extra"][family][level]["mean"], atol=1e-14)
            assert max(float(np.sqrt(np.mean((values-center)**2))), 1e-6) == stats["extra"][family][level]["rms"]
    for fs in frames.values():
        for f in fs:
            n = len(f["reference"])
            view_hot, level_hot = np.zeros((n, 5)), np.zeros((n, 3))
            for row in np.flatnonzero(f["active"]):
                view_hot[row, f["chosen"][row, 0]] = 1
                level_hot[row, f["level"][row]] = 1
            base = np.c_[f["geometry"], (f["logits"]-stats["logits_mean"])/stats["logits_std"], view_hot, level_hot]
            f["inputs"] = {}
            for arm in ARMS:
                extra = np.zeros((n, 128))
                family = "row" if arm == "row" else "null"
                values = f[family] if arm != "shuffled_null" else f["null"][f["permutation"]]
                if arm != "logits_only":
                    for i in np.flatnonzero(f["active"]):
                        info = stats["extra"][family][f["level"][i]]
                        extra[i] = (values[i]-info["mean"])/info["rms"]
                f["inputs"][arm] = np.c_[base, extra]
    scores = np.concatenate([f["reference"][f["active"], 4] for f in train])
    offset = np.log(np.clip(scores, 1e-6, 1-1e-6)/(1-np.clip(scores, 1e-6, 1-1e-6)))
    labels = np.concatenate([f["fit_labels"][f["active"]] for f in train])
    for arm in ARMS:
        x = np.concatenate([f["inputs"][arm][f["active"]] for f in train])
        theta = coefficients[arm]
        z = offset+x@theta[:-1]+theta[-1]
        probability = np.exp(-np.logaddexp(0, -z))
        residual = (probability-labels)/len(labels)
        gradient = np.r_[x.T@residual+.01*theta[:-1], residual.sum()]
        objective = float(np.mean(np.logaddexp(0, z)-labels*z)+.005*np.dot(theta[:-1], theta[:-1]))
        np.testing.assert_allclose(objective, training[arm]["objective"], atol=1e-12)
        np.testing.assert_allclose(np.max(np.abs(gradient)), training[arm]["gradient_inf"], atol=1e-12)
        assert np.max(np.abs(gradient)) <= 1e-5 and training[arm]["success"]
    row_losses, group_labels, group_scores, group_counts = ({} for _ in range(4))
    for group in GROUPS:
        row_losses[group] = {a: [] for a in ARMS}
        group_labels[group] = []
        group_scores[group] = {a: [] for a in ARMS}
        group_counts[group] = []
    for f in frames["holdout"]:
        path = output/traces_manifest[f["stem"]]["file"]
        assert file_hash(path) == traces_manifest[f["stem"]]["sha256"]
        with np.load(path) as z:
            trace = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(trace["reference"], f["reference"])
        labels, duplicate = fixed_labels(f["reference"], f["metadata"]["valid_objects"], .25)
        np.testing.assert_array_equal(trace["labels"], labels)
        np.testing.assert_array_equal(trace["duplicate"], duplicate)
        score = np.clip(f["reference"][:, 4], 1e-6, 1-1e-6)
        offset = np.log(score/(1-score))
        values = {a: offset+f["inputs"][a]@coefficients[a][:-1]+coefficients[a][-1] for a in ARMS}
        for a in ARMS:
            np.testing.assert_allclose(values[a], trace[a+"_logits"], atol=1e-12)
        area = (f["reference"][:, 2:4]-f["reference"][:, :2]).prod(1)*(640/max(f["metadata"]["original_size"]))**2
        eligible = f["active"] & (f["reference"][:, 4] >= .25) & ~duplicate
        for group in GROUPS:
            selected = eligible.copy()
            if group != "all":
                selected &= area < (16 if group == "tiny" else 32)**2
            np.testing.assert_array_equal(selected, trace[group+"_selected"])
            group_counts[group].append(int(selected.sum()))
            group_labels[group].append(labels[selected])
            for a, z in values.items():
                loss = np.logaddexp(0, z)-labels*z
                row_losses[group][a].append(float(loss[selected].sum()))
                group_scores[group][a].append(z[selected])
    metrics = read("metrics.json")
    draws = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    for group in GROUPS:
        labels = np.concatenate(group_labels[group])
        for a in ARMS:
            nll = sum(row_losses[group][a])/sum(group_counts[group])
            np.testing.assert_allclose(nll, metrics[group]["NLL"][a], atol=1e-12)
            auc = rank_auc(labels, np.concatenate(group_scores[group][a]))
            np.testing.assert_allclose(auc, metrics[group]["AUROC"][a], atol=1e-14)
        assert int(labels.sum()) == metrics[group]["tp"]
        assert int((~labels).sum()) == metrics[group]["nonduplicate_fp"]
        numerator = draws@(np.array(row_losses[group]["logits_only"])-np.array(row_losses[group]["null"]))
        denominator = draws@np.array(group_counts[group])
        interval = np.quantile(numerator[denominator > 0]/denominator[denominator > 0], [.025, .975])
        np.testing.assert_allclose(interval, metrics[group]["null_NLL_improvement_over_logits_image_bootstrap95ci"], atol=1e-12)
    verification = {
        "all192_feature_associations_masks_first64historical_channels_replayed": True,
        "frozen_FP16_projection_weights_biases_verified_without_forward": True,
        "all_row_null_decompositions_independently_checked_with_pseudoinverse": True,
        "all_training_labels_recomputed_by_independent_greedy_matching": True,
        "all_class_view_level_shuffle_permutations_recomputed": True,
        "all_train_only_normalization_and_inputs_rebuilt": True,
        "all4_saved_coefficients_have_verified_objective_and_gradient_stationarity": True,
        "all64_prediction_labels_masks_linear_scores_NLL_AUROC_and_intervals_recomputed": True,
        "shuffled_rows_changed": shuffled_changed, "shuffled_active_rows": shuffled_total,
        "maximum_FP16_native_logit_vs_float64_projection_absolute_difference": max_full_logit_error,
        "verifier_sha256": file_hash(Path(__file__)),
        "limits": "Only first64 feature channels have an earlier independent cache. Other channels are new extraction; all10 actual logits are used, not asserted bitexact to a float64 matmul. No fresh-scene/AP proof.",
    }
    (output/"independent_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
