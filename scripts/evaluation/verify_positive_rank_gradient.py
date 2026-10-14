"""Independent pair/label and CPU FP64 full classification-gradient audit for65."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


ARMS = ("original", "transport", "ordinary")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_labels(frame, ignored):
    gt, classes = frame["gt"], frame["classes"].astype(int)
    n = len(gt)
    pairs = np.full((n, 2), -1, int)
    qualities, probabilities = np.zeros((n, 2)), np.zeros((n, 2))
    masses = {a: frame["target_mass"].astype(float).copy() for a in ARMS}
    for g, target in enumerate(gt.astype(float)):
        pool = [i for i in np.flatnonzero(frame["foreground"]) if frame["owner"][i] == g and frame["target_mass"][i] > 0]
        if not pool:
            continue
        boxes = frame["decoded_boxes"][pool].astype(float)
        intersection = np.maximum(0, np.minimum(target[2:], boxes[:, 2:])-np.maximum(target[:2], boxes[:, :2])).prod(1)
        union = np.maximum(0, target[2:]-target[:2]).prod()+np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(1)-intersection
        ious = intersection/np.maximum(union, 1e-12)
        p = frame["probabilities"][pool, classes[g]]
        best = min(range(len(pool)), key=lambda k: (-ious[k], -float(p[k]), pool[k]))
        peak = min(range(len(pool)), key=lambda k: (-float(p[k]), -ious[k], pool[k]))
        pairs[g], qualities[g], probabilities[g] = [pool[best], pool[peak]], ious[[best, peak]], p[[best, peak]]
        rank = sorted(range(len(pool)), key=lambda k: (ious[k], float(frame["target_mass"][pool[k]]), pool[k]))
        target_values = sorted(float(frame["target_mass"][i]) for i in pool)
        for k, value in zip(rank, target_values):
            masses["transport"][pool[k]] = value
        original_total = np.array(target_values).sum()
        lo, hi = 0., 1/min(ious)
        for _ in range(60):
            middle = (lo+hi)/2
            if np.minimum(1., middle*ious).sum() < original_total:
                lo = middle
            else:
                hi = middle
        control = np.minimum(1., (lo+hi)/2*ious)
        assert abs(control.sum()-original_total) <= 1e-10
        masses["ordinary"][pool] = control
    eligible = np.zeros(n, bool)
    for g, (best, peak) in enumerate(pairs):
        eligible[g] = (min(best, peak) >= 0 and qualities[g, 0] >= .75 and
            qualities[g, 0]-qualities[g, 1] >= .1 and probabilities[g, 1]-probabilities[g, 0] >= .05 and
            not ignored[best] and not ignored[peak])
    return dict(pairs=pairs, pair_iou=qualities, pair_probability=probabilities, eligible=eligible,
        areas=np.maximum(0, gt[:, 2:]-gt[:, :2]).prod(1)*(640/960)**2,
        **{"mass_"+a: m.astype(np.float32) for a, m in masses.items()})


def verify():
    torch.set_num_threads(1)
    root = project_path("output/analysis/positive-rank-gradient-fixed-20261015")
    destination = root/"independent_verification.json"
    assert not destination.exists()
    result, plan, manifest = read(root/"result.json"), read(root/"plan.json"), read(root/"frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(archive.read(name)).hexdigest() == digest
        for name, digest in plan["installed_source_hashes"].items():
            assert file_hash(name) == hashlib.sha256(archive.read("installed/"+Path(name).name)).hexdigest() == digest
    prior = {50: project_path("output/analysis/assignment-reachability-fixed-20261014"),
             63: project_path("output/analysis/assignment-quality-conflict-fixed-20261015")}
    closeouts = {50: project_path("docs/experiments/candidate50_assignment_reachability_result_20261014.json"),
                63: project_path("docs/experiments/candidate63_assignment_quality_conflict_result_20261015.json")}
    prior_frames = {}
    for number in (50, 63):
        assert file_hash(closeouts[number]) == plan["prior_closeout_sha256"][str(number)]
        assert file_hash(prior[number]/"plan.json") == plan["prior_plan_sha256"][str(number)]
        for name, digest in read(closeouts[number])["artifacts_sha256"].items():
            assert file_hash(prior[number]/name) == digest
        prior_frames[number] = read(prior[number]/"frames.json")
    assert plan["image_selection"] == read(prior[63]/"plan.json")["image_selection"]
    old = read(prior[50]/"plan.json")
    assert plan["checkpoint"] == old["checkpoint"]
    assert file_hash(project_path(plan["checkpoint"])) == old["checkpoint_sha256"] == plan["checkpoint_sha256"]
    assert file_hash(project_path("data/visdrone/labels/train.cache")) == plan["label_cache_sha256"]
    assert plan["model_state_before"] == result["model_state_before"] == result["model_state_after"]
    assert plan["parameter_count"] == 3870 and len(plan["parameter_names"]) == 6
    assert plan["actual_assigner"] == old["actual_assigner"]
    registration = read(project_path("docs/experiments/candidate65_positive_rank_gradient_registration_20261015.json"))
    assert registration == plan["registration"]
    audit = read(project_path("docs/literature/positive_rank_gradient_prior_audit_20261015.json"))
    for paper in audit["papers"]:
        for key in ("pdf", "text", "preview"):
            if key in paper:
                assert file_hash(project_path(paper[key])) == paper[key+"_sha256"]
    recorded = read(root/"per_image.json")
    rebuilt, maximum_error, checked_elements, quantization = {}, 0., 0, 0.
    largest_relative_L2 = 0.
    gradient_audits, logit_records, cpu_classification_differences = [], {}, 0
    for split, stems in plan["image_selection"].items():
        rebuilt[split] = []
        for stem in stems:
            entry = manifest[stem]
            assert entry["split"] == split and file_hash(root/entry["file"]) == entry["sha256"]
            with np.load(root/entry["file"], allow_pickle=False) as archive:
                f = {k: archive[k].copy() for k in archive.files}
            for number in (50, 63):
                pe = prior_frames[number][stem]
                assert file_hash(prior[number]/pe["file"]) == pe["sha256"] == entry[f"prior{number}_frame_sha256"]
            with np.load(prior[50]/prior_frames[50][stem]["file"], allow_pickle=False) as archive:
                original = {k: archive[k].copy() for k in archive.files}
            with np.load(prior[63]/prior_frames[63][stem]["file"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(f["ignored"], archive["ignored"])
            for key in ("gt", "classes", "foreground", "owner", "probabilities"):
                np.testing.assert_array_equal(f[key], original[key])
            for kind, folder, suffix in (("image", "images", ".jpg"), ("metadata", "metadata", ".json"), ("label", "labels", ".txt")):
                assert file_hash(project_path(f"data/visdrone/{folder}/train/{stem}{suffix}")) == prior_frames[50][stem][kind+"_sha256"]
            values = independent_labels(original, f["ignored"])
            for key, value in values.items():
                np.testing.assert_array_equal(value, f[key])
            features, levels = f["features"].astype(float), f["levels"]
            assert features.shape == (18900, 128)
            np.testing.assert_array_equal(levels, np.repeat(np.arange(3, dtype=np.int8), [14400, 3600, 900]))
            foreground, row_class = f["foreground"], f["classes"].astype(int)[f["owner"]]
            base_targets = np.zeros_like(f["probabilities"])
            positive = np.flatnonzero(foreground)
            base_targets[positive, row_class[positive]] = f["mass_original"][positive]
            assert float(torch.tensor(base_targets[None], device="cuda").sum().clamp_min(1)) == float(f["denominator"])
            cosines = {a: np.zeros(len(f["gt"])) for a in ARMS}
            independent_logit = {}
            for arm in ARMS:
                targets = base_targets.copy()
                targets[positive, row_class[positive]] = f["mass_"+arm][positive]
                # BCE derivative at saved FP32 logits; all native rows and classes contribute.
                derivative = (f["probabilities"].astype(float)-targets.astype(float))*plan["cls_gain"]/float(f["denominator"])
                derivative[f["ignored"] & ~foreground] = 0.
                weight = np.stack([derivative[levels == level].T@features[levels == level] for level in range(3)])
                bias = np.stack([derivative[levels == level].sum(0) for level in range(3)])
                combined = np.r_[weight.reshape(-1), bias.reshape(-1)]
                saved = np.r_[f["gradient_weight_"+arm].reshape(-1), f["gradient_bias_"+arm].reshape(-1)]
                relative = float(np.linalg.norm(saved-combined)/max(np.linalg.norm(combined), 1e-12))
                element_pass = bool(np.all(np.abs(saved-combined) <= 1e-6+1e-4*np.abs(combined)))
                gradient_audits.append(dict(stem=stem, arm=arm, elementwise_pass=element_pass,
                    relative_L2=relative, relative_L2_pass=relative <= 1e-4))
                maximum_error = max(maximum_error, float(np.max(np.abs(saved-combined))))
                largest_relative_L2 = max(largest_relative_L2, relative)
                checked_elements += combined.size
                exact_gradient = np.r_[f["gradient_weight_"+arm].astype(float).reshape(-1),
                                       f["gradient_bias_"+arm].astype(float).reshape(-1)]
                gradient_norm = np.linalg.norm(exact_gradient)
                independent_logit[arm] = np.zeros(len(f["gt"]))
                for g, (best, peak) in enumerate(values["pairs"]):
                    if min(best, peak) < 0:
                        continue
                    cls = int(f["classes"][g])
                    hweight, hbias = np.zeros((3, 10, 128)), np.zeros((3, 10))
                    hweight[levels[best], cls] += features[best]
                    hweight[levels[peak], cls] -= features[peak]
                    hbias[levels[best], cls] += 1.
                    hbias[levels[peak], cls] -= 1.
                    h = np.r_[hweight.reshape(-1), hbias.reshape(-1)]
                    norm = np.linalg.norm(h)
                    np.testing.assert_allclose(norm, f["hnorm"][g], atol=1e-12, rtol=1e-12)
                    cosines[arm][g] = -np.dot(exact_gradient, h)/(gradient_norm*norm) if gradient_norm > 0 and norm > 0 else 0.
                    cpu_norm = np.linalg.norm(combined)
                    cpu_cosine = -np.dot(combined, h)/(cpu_norm*norm) if cpu_norm > 0 and norm > 0 else 0.
                    if values["eligible"][g]:
                        cpu_classification_differences += int((cpu_cosine < -.01) != (cosines[arm][g] < -.01))
                        cpu_classification_differences += int((cpu_cosine > .01) != (cosines[arm][g] > .01))
                    independent_logit[arm][g] = (float(f["mass_"+arm][best])-float(f["probabilities"][best, cls])-
                                                float(f["mass_"+arm][peak])+float(f["probabilities"][peak, cls]))
                np.testing.assert_allclose(cosines[arm], f["cosine_"+arm], atol=1e-12, rtol=1e-12)
            logit_records[stem] = np.column_stack([independent_logit[a] for a in ARMS])
            for g in range(len(f["gt"])):
                pool = foreground & (f["owner"] == g) & (f["mass_original"] > 0)
                quantization = max(quantization, abs(f["mass_ordinary"][pool].astype(float).sum()-f["mass_original"][pool].astype(float).sum()))
            groups = {}
            for group, limit in (("all", np.inf), ("tiny", 256.), ("small_all", 1024.)):
                keep = values["areas"] < limit
                worsening = keep & values["eligible"] & (cosines["original"] < -.01)
                groups[group] = dict(gt=int(keep.sum()), eligible=int((keep & values["eligible"]).sum()),
                    worsening=int(worsening.sum()), worsening_images=int(worsening.any()),
                    repaired_transport=int((worsening & (cosines["transport"] > .01)).sum()),
                    repaired_ordinary=int((worsening & (cosines["ordinary"] > .01)).sum()))
            rebuilt[split].append(dict(stem=stem, groups=groups))
        assert rebuilt[split] == [dict(stem=r["stem"], groups=r["groups"]) for r in recorded[split]]
    assert quantization == result["ordinary_FP32_target_quantization_max_GTmass_drift"]
    metrics, checks = {}, {}
    policy = registration["policy_per_half"]
    for split, rows in rebuilt.items():
        totals = {g: {k: sum(row["groups"][g][k] for row in rows) for k in rows[0]["groups"][g]}
                  for g in ("all", "tiny", "small_all")}
        for m in totals.values():
            m["worsening_fraction_allGT"] = m["worsening"]/m["gt"] if m["gt"] else 0.
            m["repair_gain_fraction_allGT"] = (m["repaired_transport"]-m["repaired_ordinary"])/m["gt"] if m["gt"] else 0.
        sequences = sorted({row["stem"][:7] for row in rows})
        counts = np.array([[sum(row["groups"]["tiny"][key] for row in rows if row["stem"][:7] == s)
                           for key in ("gt", "repaired_transport", "repaired_ordinary")] for s in sequences])
        draws = np.random.default_rng(0).multinomial(len(sequences), [1/len(sequences)]*len(sequences), size=400)
        samples = draws@counts
        ci = np.percentile((samples[:, 1]-samples[:, 2])/samples[:, 0], [2.5, 97.5]).tolist() if (samples[:, 0] > 0).all() else None
        totals.update(sequence_units=len(sequences), repair_gain_sequence_bootstrap95ci=ci)
        m = totals["tiny"]
        checks[split] = dict(GT=m["gt"] >= policy["tiny_GT_min"], worsening=m["worsening"] >= policy["tiny_worsening_GT_min"],
            images=m["worsening_images"] >= policy["tiny_worsening_images_min"],
            fraction=m["worsening_fraction_allGT"] >= policy["tiny_worsening_fraction_allGT_min"],
            repair=m["repaired_transport"] >= policy["tiny_transport_repaired_GT_min"],
            vs_ordinary=m["repair_gain_fraction_allGT"] >= policy["tiny_repair_gain_vs_ordinary_fraction_allGT_min"],
            interval=ci is not None and ci[0] > policy["tiny_repair_gain_sequence_interval_lower_min_exclusive"])
        metrics[split] = totals
    assert metrics == result["metrics"] == read(root/"metrics.json")
    assert checks == result["decision"]["checks"]
    passed = all(all(c.values()) for c in checks.values())
    decision = ("REPLICATED_LAST_CLASS_PROJECTION_RANK_HARM_SUPPORTS_SEPARATE_BOUNDED_UPDATE_PILOT" if passed else
                "STOP_THIS_WITHINGT_QUALITY_HISTOGRAM_TRANSPORT_CLASS_PROJECTION_ROUTE")
    assert decision == result["decision"]["decision"] == read(root/"decision.json")["decision"]
    verification = dict(prior_sources_images_metadata_labels_and_paper_hashes_verified=True,
        all64pairs_histogramtargets_ordinarytargets_andscoregap_jacobians_rebuilt=True,
        full_CPU_FP64_gradient_elements_checked=checked_elements, maximum_absolute_gradient_error=maximum_error,
        maximum_relative_gradient_L2_error=largest_relative_L2, gradient_tolerances_unchanged=True,
        full_CPU_FP64_gradient_audit_passed=all(v["elementwise_pass"] and v["relative_L2_pass"] for v in gradient_audits),
        gradient_audits=gradient_audits,
        diagnostic_CPU_vs_saved_autograd_eligible_cosine_threshold_classification_differences=cpu_classification_differences,
        all_groups_sequence_bootstrap_and_fixed_gates_recomputed=True, decision=decision,
        added_detector_forwards=0, added_JPEG_samples=0, added_parameter_updates=0,
        verifier_sha256=file_hash(Path(__file__)),
        limits="Full cachedfeature/probability gradient reconstruction,not independent detector-feature extraction or model-state replay. ClosedformcpuFP64 compared with actualCUDAFP32autograd at preregistered tolerance. No optimizerupdate/AP/generalization/novelty."
    )
    np.savez_compressed(root/"independent_logit_directions.npz", **logit_records)
    destination.write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in verification.items() if k != "gradient_audits"}, indent=2), flush=True)
    if not verification["full_CPU_FP64_gradient_audit_passed"]:
        raise RuntimeError("Frozen CPU gradient tolerances failed; audit saved, no tolerance or primary result changes.")


if __name__ == "__main__":
    verify()
