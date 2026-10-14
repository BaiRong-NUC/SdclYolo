"""Audit candidate63 from original native caches and an independent pair calculation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import SDCLConfig, project_path
from sdcl.evaluation.assignment_quality_conflict import GeometryOnlyDataset
from sdcl.evaluation.size_analysis import file_hash
from sdcl.training.criterion import SDCLLoss


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    torch.set_num_threads(1)
    root = project_path("output/analysis/assignment-quality-conflict-fixed-20261015")
    prior = project_path("output/analysis/assignment-reachability-fixed-20261014")
    destination = root/"independent_verification.json"
    assert not destination.exists()
    result, plan = read(root/"result.json"), read(root/"plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == file_hash(project_path(name)) == digest
        for name, digest in plan["installed_source_hashes"].items():
            assert hashlib.sha256(archive.read("installed/"+Path(name).name)).hexdigest() == file_hash(name) == digest
    closeout_path = project_path("docs/experiments/candidate50_assignment_reachability_result_20261014.json")
    assert file_hash(closeout_path) == plan["prior50_closeout_sha256"]
    for name, digest in read(closeout_path)["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    for name in ("plan", "frames"):
        assert file_hash(prior/(name+".json")) == plan["prior50_"+name+"_sha256"]
    old, original_frames = read(prior/"plan.json"), read(prior/"frames.json")
    for name, digest in old["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    for item in old["installed_sources"].values():
        assert file_hash(item["path"]) == item["sha256"]
    audit = read(project_path("docs/literature/assignment_quality_conflict_prior_audit_20261015.json"))
    for paper in audit["papers"]:
        for key in ("pdf", "text", "preview"):
            if key in paper:
                assert file_hash(project_path(paper[key])) == paper[key+"_sha256"]
    registration = read(project_path("docs/experiments/candidate63_assignment_quality_conflict_registration_20261015.json"))
    assert registration == plan["registration"]
    sequences = sorted({stem[:7] for stem in old["images"]},
                       key=lambda s: hashlib.sha256(("candidate63-split-v1:"+s).encode()).hexdigest())
    assert len(sequences) == 34
    selection = {split: [stem for stem in old["images"] if stem[:7] in seqs]
                 for split, seqs in (("screen", sequences[:17]), ("confirmation", sequences[17:]))}
    assert plan["image_selection"] == selection
    cache_path = project_path("data/visdrone/labels/train.cache")
    assert file_hash(cache_path) == plan["label_cache_sha256"]
    args = get_cfg(overrides=dict(imgsz=960, batch=1, workers=0, rect=False, plots=False, device="0"))
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = GeometryOnlyDataset(img_path=str(data_path.parent/"images/train"), imgsz=960, batch_size=1,
        augment=False, hyp=args, rect=False, cache=False, single_cls=False, stride=32, pad=0.,
        prefix="verify-label-proxy: ", task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    checkpoint = project_path(old["checkpoint"])
    assert file_hash(checkpoint) == old["checkpoint_sha256"] == plan["checkpoint_sha256"]
    model = YOLO(str(checkpoint)).model.float().cuda().eval()
    model.args = args
    model.sdcl_settings = SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions = True
    model.coverage_settings = {"enabled": False}
    criterion = SDCLLoss(model)
    actual = criterion.assigner
    assert {k: getattr(actual, k) for k in plan["actual_assigner"]} == plan["actual_assigner"]
    control = deepcopy(actual)
    control.alpha = 0.
    assert {k: getattr(control, k) for k in plan["geometryonly_assigner"]} == plan["geometryonly_assigner"]
    frames, recorded_traces = read(root/"frames.json"), read(root/"per_image.json")
    rebuilt = {}
    native_gt_queries = 0
    for split, stems in selection.items():
        rebuilt[split] = []
        for stem in stems:
            entry, previous = frames[stem], original_frames[stem]
            assert entry["split"] == split
            assert file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(prior/previous["file"]) == previous["sha256"] == entry["prior_frame_sha256"]
            for kind, folder, suffix in (("image", "images", ".jpg"), ("metadata", "metadata", ".json"), ("label", "labels", ".txt")):
                assert file_hash(project_path(f"data/visdrone/{folder}/train/{stem}{suffix}")) == previous[kind+"_sha256"]
            with np.load(prior/previous["file"], allow_pickle=False) as archive:
                f = {k: archive[k].copy() for k in archive.files}
            with np.load(root/entry["file"], allow_pickle=False) as archive:
                saved = {k: archive[k].copy() for k in archive.files}
            batch = dataset.collate_fn([dataset[indices[stem]]])
            targets = torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1).cuda()
            labels, boxes = criterion.preprocess(targets, 1, torch.tensor([960]*4, device="cuda")).split((1, 4), 2)
            np.testing.assert_array_equal(boxes[0].cpu().numpy(), f["gt"])
            np.testing.assert_array_equal(labels[0, :, 0].cpu().numpy(), f["classes"])
            np.testing.assert_array_equal(saved["gt"], f["gt"])
            np.testing.assert_array_equal(saved["classes"], f["classes"])
            np.testing.assert_array_equal(batch["ignore_bboxes"].numpy(), saved["ignore_bboxes_normalized"])
            ignored = np.zeros(len(f["anchors"]), bool)
            for box in batch["ignore_bboxes"].numpy():
                center, half = box[:2]*960, box[2:]*960/2
                ignored |= ((f["anchors"] >= center-half).all(1) & (f["anchors"] <= center+half).all(1))
            np.testing.assert_array_equal(ignored, saved["ignored"])
            tensors = lambda a: torch.tensor(a, device="cuda")
            arguments = (tensors(f["probabilities"][None]), tensors(f["decoded_boxes"][None]),
                         tensors(f["anchors"]), labels, boxes, boxes.sum(2, keepdim=True) > 0)
            with torch.no_grad():
                assignments = (actual(*arguments), control(*arguments))
            arms = {}
            for name, assignment in zip(("actual", "geometryonly"), assignments):
                fg = assignment[3][0].cpu().numpy()
                owner = assignment[4][0].cpu().numpy()
                mass = assignment[2].sum(-1)[0].cpu().numpy()
                for key, value in (("foreground", fg), ("owner", owner), ("mass", mass)):
                    np.testing.assert_array_equal(value, saved[name+"_"+key])
                    if name == "actual":
                        np.testing.assert_array_equal(value, f["target_mass" if key == "mass" else key])
                if name == "actual":
                    np.testing.assert_array_equal(assignment[1][0].cpu().numpy(), f["assigned_boxes"])
                arms[name] = (fg, owner, mass)
            pairs = {k: [] for k in ("best", "peak", "best_iou", "peak_iou", "covered", "high", "peak_gap", "eligible")}
            directions = {arm: {k: [] for k in ("best_probability", "peak_probability", "best_target", "peak_target",
                         "best_descent", "peak_descent", "relative_descent", "best_background", "harmful")} for arm in arms}
            native_boxes, probability = f["decoded_boxes"].astype(float), f["probabilities"]
            classes = f["classes"].astype(int)
            for g, gt in enumerate(f["gt"].astype(float)):
                intersection = np.maximum(0, np.minimum(native_boxes[:, 2:], gt[2:])-np.maximum(native_boxes[:, :2], gt[:2])).prod(1)
                union = np.maximum(0, native_boxes[:, 2:]-native_boxes[:, :2]).prod(1)+np.maximum(0, gt[2:]-gt[:2]).prod()-intersection
                iou = intersection/np.maximum(union, 1e-12)
                pool = np.flatnonzero((probability.argmax(1) == classes[g]) & (probability[:, classes[g]] >= .05) & (iou >= .5)).tolist()
                best = min(pool, key=lambda i: (-iou[i], -float(probability[i, classes[g]]), i)) if pool else -1
                peak = min(pool, key=lambda i: (-float(probability[i, classes[g]]), -iou[i], i)) if pool else -1
                bi, pi = (float(iou[i]) if i >= 0 else 0. for i in (best, peak))
                covered = bool(np.any(f["foreground"] & (f["owner"] == g) & (f["target_mass"] > 0)))
                high, gap = bi >= .75, bi >= .75 and pi < .75
                inside = best >= 0 and bool((f["anchors"][best] > gt[:2]).all() and (f["anchors"][best] < gt[2:]).all())
                eligible = bool(covered and gap and inside and not ignored[max(best, 0)] and not ignored[max(peak, 0)])
                for key, value in zip(pairs, (best, peak, bi, pi, covered, high, gap, eligible)):
                    pairs[key].append(value)
                for name, (fg, owner, mass) in arms.items():
                    p, t, d = [], [], []
                    for index in (max(best, 0), max(peak, 0)):
                        score = float(probability[index, classes[g]])
                        target = float(mass[index]) if fg[index] and classes[owner[index]] == classes[g] else 0.
                        p.append(score)
                        t.append(target)
                        d.append(0. if ignored[index] and not fg[index] else target-score)
                    harmful = bool(eligible and not fg[max(best, 0)] and fg[max(peak, 0)] and
                                   owner[max(peak, 0)] == g and mass[max(peak, 0)] > 0 and d[1] > 1e-6)
                    values = (p[0], p[1], t[0], t[1], d[0], d[1], d[0]-d[1],
                              bool(eligible and not fg[max(best, 0)]), harmful)
                    for key, value in zip(directions[name], values):
                        directions[name][key].append(value)
                native_gt_queries += len(native_boxes)
            pairs = {k: np.array(v) for k, v in pairs.items()}
            for key, values in pairs.items():
                np.testing.assert_array_equal(values, saved[key])
            for arm, columns in directions.items():
                for key, values in columns.items():
                    np.testing.assert_array_equal(values, saved[arm+"_"+key])
            area = np.maximum(0, f["gt"][:, 2:]-f["gt"][:, :2]).prod(1)*(640/960)**2
            np.testing.assert_array_equal(area, saved["areas"])
            groups = {}
            for group, limit in (("all", np.inf), ("tiny", 256.), ("small_all", 1024.)):
                keep = area < limit
                count = lambda values: int(np.count_nonzero(keep & np.asarray(values, bool)))
                groups[group] = dict(gt=int(keep.sum()), covered=count(pairs["covered"]),
                    high_quality_supported=count(pairs["high"]), peak_gap=count(pairs["peak_gap"]), eligible_pairs=count(pairs["eligible"]),
                    best_background_actual=count(directions["actual"]["best_background"]),
                    best_background_geometryonly=count(directions["geometryonly"]["best_background"]),
                    harmful_actual=count(directions["actual"]["harmful"]),
                    harmful_geometryonly=count(directions["geometryonly"]["harmful"]),
                    harmful_actual_images=int(bool(count(directions["actual"]["harmful"]))))
            rebuilt[split].append(dict(stem=stem, groups=groups,
                actual_positive_positions=int(arms["actual"][0].sum()), geometryonly_positive_positions=int(arms["geometryonly"][0].sum()),
                actual_target_mass=float(arms["actual"][2].sum()), geometryonly_target_mass=float(arms["geometryonly"][2].sum())))
        assert rebuilt[split] == recorded_traces[split]
    metrics, checks = {}, {}
    policy = registration["policy_per_half"]
    for split, rows in rebuilt.items():
        totals = {group: {key: sum(row["groups"][group][key] for row in rows) for key in rows[0]["groups"][group]}
                  for group in ("all", "tiny", "small_all")}
        for m in totals.values():
            m["harmful_actual_fraction_allGT"] = m["harmful_actual"]/m["gt"] if m["gt"] else 0.
            m["harmful_reduction_fraction_allGT"] = (m["harmful_actual"]-m["harmful_geometryonly"])/m["gt"] if m["gt"] else 0.
        prefixes = sorted({r["stem"][:7] for r in rows})
        counts = np.array([[sum(r["groups"]["tiny"][key] for r in rows if r["stem"][:7] == seq)
                            for key in ("gt", "harmful_actual", "harmful_geometryonly")] for seq in prefixes])
        draws = np.random.default_rng(0).multinomial(len(prefixes), [1/len(prefixes)]*len(prefixes), size=400)
        samples = draws@counts
        ci = np.percentile((samples[:, 1]-samples[:, 2])/samples[:, 0], [2.5, 97.5]).tolist() if (samples[:, 0] > 0).all() else None
        totals.update(sequence_units=len(prefixes), tiny_sequence_bootstrap_reduction95ci=ci)
        m = totals["tiny"]
        checks[split] = dict(gt=m["gt"] >= policy["tiny_GT_min"],
            harmful=m["harmful_actual"] >= policy["tiny_harmful_actual_min"],
            images=m["harmful_actual_images"] >= policy["tiny_harmful_actual_images_min"],
            fraction=m["harmful_actual_fraction_allGT"] >= policy["tiny_harmful_actual_fraction_allGT_min"],
            reduction=m["harmful_reduction_fraction_allGT"] >= policy["tiny_harmful_reduction_vs_geometryonly_fraction_allGT_min"],
            interval=ci is not None and ci[0] > policy["sequence_bootstrap_reduction_lower_min_exclusive"])
        metrics[split] = totals
    assert metrics == result["metrics"] == read(root/"metrics.json")
    assert checks == result["decision"]["checks"]
    passed = all(all(c.values()) for c in checks.values())
    decision = ("REPLICATED_CACHED_LABEL_RANK_CONFLICT_SUPPORTS_SEPARATE_SHARED_PARAMETER_FEASIBILITY" if passed else
                "STOP_THIS_ACTIVE_BACKGROUND_BETTER_BOX_ASSIGNMENT_FEEDBACK_ROUTE_ON_FIXED_PROXY")
    assert decision == result["decision"]["decision"] == read(root/"decision.json")["decision"]
    assert file_hash(cache_path) == plan["label_cache_sha256"]
    verification = dict(original_cache_source_archive_installed_sources_images_metadata_labels_and_paper_hashes_verified=True,
        original_GT_classes_and_ignore_geometry_reconstructed_for_all64=True,
        original_and_geometryonly_TAL_replays=128, independently_recomputed_GT_native_IoU_queries=native_gt_queries,
        all_pairs_targets_independent_logit_directions_groups_sequence_bootstrap_and_gates_rebuilt=True,
        decision=decision, added_detector_forwards=0, added_JPEG_decodes=0, added_parameter_updates=0,
        verifier_sha256=file_hash(Path(__file__)),
        limits="Reuses blank-image dataset geometry, original SDCL criterion preprocess and TAL engines, not independent implementations of those. IoUs, pair selection, ignore mask, targets, descent, counts, bootstrap and gates independently calculated. Repeatedly inspected detector-training proxy; not optimizer harm, AP or novelty.")
    destination.write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    verify()
