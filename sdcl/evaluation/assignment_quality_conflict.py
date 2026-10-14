"""Cached supervised-score inversion screen with an ordinary TAL ablation."""

from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import base as base_module, dataset as dataset_module, augment as augment_module
from ultralytics.utils import loss as loss_module, tal as tal_module
import yaml

from sdcl.config import SDCLConfig, project_path
from sdcl.data.dataset import IgnoreDataset
from sdcl.data.geometry import anchor_ignore_mask
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash
from sdcl.training.criterion import SDCLLoss
from sdcl.training.trainer import check_version


REGISTRATION = "docs/experiments/candidate63_assignment_quality_conflict_registration_20261015.json"
PRIOR_AUDIT = "docs/literature/assignment_quality_conflict_prior_audit_20261015.json"
PRIOR = "output/analysis/assignment-reachability-fixed-20261014"
OUTPUT = "output/analysis/assignment-quality-conflict-fixed-20261015"
GROUPS = dict(all=float("inf"), tiny=256., small_all=1024.)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


class GeometryOnlyDataset(IgnoreDataset):
    """Reuse the label pipeline without reading or processing real image pixels."""
    def cache_labels(self, *args, **kwargs):
        raise RuntimeError("The existing label cache must be valid; no image-scanning fallback is allowed.")

    def load_image(self, i, rect_mode=True, resize_short=False):
        if not rect_mode or resize_short:
            raise ValueError("Unexpected original loader mode.")
        path = Path(self.im_files[i])
        metadata = read(path.parents[2]/"metadata"/path.parent.name/(path.stem+".json"))
        width, height = metadata["original_size"]
        ratio = self.imgsz/max(height, width)
        w = min(math.ceil(width*ratio), self.imgsz) if ratio != 1 else width
        h = min(math.ceil(height*ratio), self.imgsz) if ratio != 1 else height
        return np.zeros((h, w, 3), np.uint8), (height, width), (h, w)


def choose_pairs(frame, ignored):
    gt, classes = frame["gt"], frame["classes"].astype(int)
    probability, boxes = frame["probabilities"], frame["decoded_boxes"]
    overlap = box_iou_matrix(gt, boxes)
    predicted = probability.argmax(1)
    n = len(gt)
    best, peak = np.full(n, -1, int), np.full(n, -1, int)
    best_iou, peak_iou = np.zeros(n), np.zeros(n)
    covered = np.zeros(n, bool)
    for g in range(n):
        covered[g] = np.any(frame["foreground"] & (frame["owner"] == g) & (frame["target_mass"] > 0))
        pool = np.flatnonzero((predicted == classes[g]) & (probability[:, classes[g]] >= .05) &
                              (overlap[g] >= .5))
        if len(pool):
            best_order = np.lexsort((pool, -probability[pool, classes[g]], -overlap[g, pool]))
            peak_order = np.lexsort((pool, -overlap[g, pool], -probability[pool, classes[g]]))
            best[g], peak[g] = pool[best_order[0]], pool[peak_order[0]]
            best_iou[g], peak_iou[g] = overlap[g, best[g]], overlap[g, peak[g]]
    high = best_iou >= .75
    gap = high & (peak_iou < .75)
    safe = np.maximum(best, 0)
    inside = ((frame["anchors"][safe] > gt[:, :2]).all(1) &
              (frame["anchors"][safe] < gt[:, 2:]).all(1) & (best >= 0))
    eligible = covered & gap & inside & ~ignored[safe] & ~ignored[np.maximum(peak, 0)]
    return dict(best=best, peak=peak, best_iou=best_iou, peak_iou=peak_iou,
                covered=covered, high=high, peak_gap=gap, eligible=eligible)


def pair_descent(frame, pairs, ignored, foreground, owner, mass):
    gt_classes = frame["classes"].astype(int)
    best, peak = np.maximum(pairs["best"], 0), np.maximum(pairs["peak"], 0)
    actual_class = gt_classes[owner]
    def query(indices):
        probability = frame["probabilities"][indices, gt_classes].astype(float)
        targets = np.where(foreground[indices] & (actual_class[indices] == gt_classes), mass[indices], 0.).astype(float)
        direction = np.where(ignored[indices] & ~foreground[indices], 0., targets-probability)
        return probability, targets, direction
    bp, bt, bd = query(best)
    pp, pt, pd = query(peak)
    harmful = (pairs["eligible"] & ~foreground[best] & foreground[peak] &
               (owner[peak] == np.arange(len(gt_classes))) & (mass[peak] > 0) & (pd > 1e-6))
    return dict(best_probability=bp, peak_probability=pp, best_target=bt, peak_target=pt,
                best_descent=bd, peak_descent=pd, relative_descent=bd-pd,
                best_background=pairs["eligible"] & ~foreground[best], harmful=harmful)


def group_statistics(frame, pairs, actual, control):
    areas = np.maximum(frame["gt"][:, 2:]-frame["gt"][:, :2], 0).prod(1)*(640/960)**2
    result = {}
    for group, limit in GROUPS.items():
        keep = areas < limit
        result[group] = dict(gt=int(keep.sum()), covered=int((keep & pairs["covered"]).sum()),
            high_quality_supported=int((keep & pairs["high"]).sum()),
            peak_gap=int((keep & pairs["peak_gap"]).sum()), eligible_pairs=int((keep & pairs["eligible"]).sum()),
            best_background_actual=int((keep & actual["best_background"]).sum()),
            best_background_geometryonly=int((keep & control["best_background"]).sum()),
            harmful_actual=int((keep & actual["harmful"]).sum()),
            harmful_geometryonly=int((keep & control["harmful"]).sum()),
            harmful_actual_images=int((keep & actual["harmful"]).any()))
    return result, areas


def aggregate(traces, policy):
    metrics, checks = {}, {}
    for split, rows in traces.items():
        totals = {g: {k: sum(row["groups"][g][k] for row in rows)
                      for k in rows[0]["groups"][g]} for g in GROUPS}
        for m in totals.values():
            m["harmful_actual_fraction_allGT"] = m["harmful_actual"]/m["gt"] if m["gt"] else 0.
            m["harmful_reduction_fraction_allGT"] = (m["harmful_actual"]-m["harmful_geometryonly"])/m["gt"] if m["gt"] else 0.
        prefixes = sorted({row["stem"][:7] for row in rows})
        gt = np.array([sum(r["groups"]["tiny"]["gt"] for r in rows if r["stem"][:7] == s) for s in prefixes])
        delta = np.array([sum(r["groups"]["tiny"]["harmful_actual"]-r["groups"]["tiny"]["harmful_geometryonly"]
                             for r in rows if r["stem"][:7] == s) for s in prefixes])
        draws = np.random.default_rng(0).multinomial(len(prefixes), [1/len(prefixes)]*len(prefixes), size=400)
        denominator = draws@gt
        ci = np.percentile((draws@delta)/denominator, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
        totals.update(sequence_units=len(prefixes), tiny_sequence_bootstrap_reduction95ci=ci)
        m = totals["tiny"]
        checks[split] = dict(gt=m["gt"] >= policy["tiny_GT_min"],
            harmful=m["harmful_actual"] >= policy["tiny_harmful_actual_min"],
            images=m["harmful_actual_images"] >= policy["tiny_harmful_actual_images_min"],
            fraction=m["harmful_actual_fraction_allGT"] >= policy["tiny_harmful_actual_fraction_allGT_min"],
            reduction=m["harmful_reduction_fraction_allGT"] >= policy[
                "tiny_harmful_reduction_vs_geometryonly_fraction_allGT_min"],
            interval=ci is not None and ci[0] > policy["sequence_bootstrap_reduction_lower_min_exclusive"])
        metrics[split] = totals
    passed = all(all(c.values()) for c in checks.values())
    return metrics, dict(checks=checks,
        decision="REPLICATED_CACHED_LABEL_RANK_CONFLICT_SUPPORTS_SEPARATE_SHARED_PARAMETER_FEASIBILITY" if passed else
        "STOP_THIS_ACTIVE_BACKGROUND_BETTER_BOX_ASSIGNMENT_FEEDBACK_ROUTE_ON_FIXED_PROXY",
        scope="GTconditioned cached labeling and independent-logit BCEdescent; not actual shared-parameter harm,AP,deployability orinnovation.")


def run(device="0"):
    started = perf_counter()
    torch.set_num_threads(1)
    check_version()
    if device != "0" or not torch.cuda.is_available():
        raise ValueError("The fixed cached proxy requires CUDA device 0.")
    root, prior = project_path(OUTPUT), project_path(PRIOR)
    if root.exists():
        raise FileExistsError(root)
    previous_path = project_path("docs/experiments/candidate50_assignment_reachability_result_20261014.json")
    previous = read(previous_path)
    for name, value in previous["artifacts_sha256"].items():
        assert file_hash(prior/name) == value
    old, manifest = read(prior/"plan.json"), read(prior/"frames.json")
    for name, digest in old["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    for item in old["installed_sources"].values():
        assert file_hash(item["path"]) == item["sha256"]
    prefixes = sorted({s[:7] for s in old["images"]}, key=lambda s: hashlib.sha256(
        ("candidate63-split-v1:"+s).encode()).hexdigest())
    assert len(prefixes) == 34
    selection = {half: [s for s in old["images"] if s[:7] in set(group)]
                 for half, group in (("screen", prefixes[:17]), ("confirmation", prefixes[17:]))}
    registration = read(project_path(REGISTRATION))
    audit = read(project_path(PRIOR_AUDIT))
    for paper in audit["papers"]:
        for key in ("pdf", "text", "preview"):
            if key in paper:
                assert file_hash(project_path(paper[key])) == paper[key+"_sha256"]
    paths = (REGISTRATION, PRIOR_AUDIT, "sdcl/evaluation/assignment_quality_conflict.py",
        "scripts/evaluation/diagnose_assignment_quality_conflict.py", "tests/test_assignment_quality_conflict.py",
        "sdcl/data/dataset.py", "sdcl/data/geometry.py", "sdcl/training/criterion.py",
        "sdcl/training/trainer.py", "sdcl/config.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    installed = {Path(inspect.getfile(m)): file_hash(inspect.getfile(m))
                 for m in (base_module, dataset_module, augment_module, loss_module, tal_module)}
    label_cache = project_path("data/visdrone/labels/train.cache")
    label_cache_hash = file_hash(label_cache)
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        for p in installed:
            z.write(p, "installed/"+p.name)
    plan = dict(registration=registration, image_selection=selection, source_hashes=hashes,
        installed_source_hashes={str(p): d for p, d in installed.items()},
        source_archive_sha256=file_hash(root/"source.zip"), prior50_closeout_sha256=file_hash(previous_path),
        prior50_plan_sha256=file_hash(prior/"plan.json"), prior50_frames_sha256=file_hash(prior/"frames.json"),
        label_cache_sha256=label_cache_hash, checkpoint_sha256=old["checkpoint_sha256"])
    save(root/"plan.json", plan)
    args = get_cfg(overrides=dict(imgsz=960, batch=1, workers=0, rect=False, plots=False, device=device))
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = GeometryOnlyDataset(img_path=str(data_path.parent/"images/train"), imgsz=960, batch_size=1,
        augment=False, hyp=args, rect=False, cache=False, single_cls=False, stride=32, pad=0.,
        prefix="quality-label-proxy: ", task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    checkpoint = project_path(old["checkpoint"])
    assert file_hash(checkpoint) == old["checkpoint_sha256"]
    model = YOLO(str(checkpoint)).model.float().to("cuda:"+device).eval()
    model.args = args
    model.sdcl_settings = SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions = True
    model.coverage_settings = {"enabled": False}
    criterion = SDCLLoss(model)
    assigner = criterion.assigner
    policy = {k: getattr(assigner, k) for k in ("topk", "topk2", "alpha", "beta", "stride_val", "eps")}
    assert policy == old["actual_assigner"]
    control = deepcopy(assigner)
    control.alpha = 0.
    plan["actual_assigner"] = policy
    plan["geometryonly_assigner"] = {**policy, "alpha": 0.}
    save(root/"plan.json", plan)
    traces, frames, count = {s: [] for s in selection}, {}, 0
    for split, stems in selection.items():
        for stem in stems:
            entry = manifest[stem]
            assert file_hash(prior/entry["file"]) == entry["sha256"]
            for kind, folder, suffix in (("image", "images", ".jpg"), ("metadata", "metadata", ".json"), ("label", "labels", ".txt")):
                assert file_hash(project_path(f"data/visdrone/{folder}/train/{stem}{suffix}")) == entry[kind+"_sha256"]
            with np.load(prior/entry["file"], allow_pickle=False) as z:
                f = {k: z[k].copy() for k in z.files}
            batch = dataset.collate_fn([dataset[indices[stem]]])
            raw_targets = torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1).to(criterion.device)
            targets = criterion.preprocess(raw_targets, 1, torch.tensor([960]*4, device=criterion.device))
            labels, gt = targets.split((1, 4), 2)
            np.testing.assert_array_equal(gt[0].cpu().numpy(), f["gt"])
            np.testing.assert_array_equal(labels[0, :, 0].cpu().numpy(), f["classes"])
            tensor = lambda x: torch.tensor(x, device=criterion.device)
            ignored = anchor_ignore_mask(tensor(f["anchors"]), batch["ignore_bboxes"].to(criterion.device),
                                         batch["ignore_batch_idx"].to(criterion.device), 1, (960, 960))[0].cpu().numpy()
            arguments = (tensor(f["probabilities"][None]), tensor(f["decoded_boxes"][None]),
                         tensor(f["anchors"]), labels, gt, gt.sum(2, keepdim=True) > 0)
            with torch.no_grad():
                actual = assigner(*arguments)
                alternative = control(*arguments)
            as_numpy = lambda t: t.detach().cpu().numpy()[0].copy()
            for tensor_value, name in ((actual[1], "assigned_boxes"), (actual[3], "foreground"), (actual[4], "owner")):
                np.testing.assert_array_equal(as_numpy(tensor_value), f[name])
            np.testing.assert_array_equal(as_numpy(actual[2].sum(-1)), f["target_mass"])
            pairs = choose_pairs(f, ignored)
            original = pair_descent(f, pairs, ignored, f["foreground"], f["owner"], f["target_mass"])
            fg, owner, mass = as_numpy(alternative[3]), as_numpy(alternative[4]), as_numpy(alternative[2].sum(-1))
            geometry = pair_descent(f, pairs, ignored, fg, owner, mass)
            groups, areas = group_statistics(f, pairs, original, geometry)
            trace = dict(stem=stem, groups=groups, actual_positive_positions=int(f["foreground"].sum()),
                geometryonly_positive_positions=int(fg.sum()), actual_target_mass=float(f["target_mass"].sum()),
                geometryonly_target_mass=float(mass.sum()))
            traces[split].append(trace)
            path = root/"frames"/(stem+".npz")
            np.savez_compressed(path, gt=f["gt"], classes=f["classes"], areas=areas, ignored=ignored,
                ignore_bboxes_normalized=batch["ignore_bboxes"].numpy(), **pairs,
                actual_foreground=f["foreground"], actual_owner=f["owner"], actual_mass=f["target_mass"],
                geometryonly_foreground=fg, geometryonly_owner=owner, geometryonly_mass=mass,
                **{"actual_"+k: v for k, v in original.items()}, **{"geometryonly_"+k: v for k, v in geometry.items()})
            frames[stem] = dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path),
                prior_frame_sha256=entry["sha256"], split=split)
            count += 1
            if count % 16 == 0:
                print(f"Assignment quality feedback: {count}/64", flush=True)
    metrics, decision = aggregate(traces, registration["policy_per_half"])
    for name, value in (("frames.json", frames), ("per_image.json", traces), ("metrics.json", metrics), ("decision.json", decision)):
        save(root/name, value)
    assert all(file_hash(project_path(p)) == d for p, d in hashes.items())
    assert all(file_hash(p) == d for p, d in installed.items())
    assert file_hash(label_cache) == label_cache_hash
    result = dict(experiment=registration["experiment"], metrics=metrics, decision=decision,
        costs=dict(cached_frames=64, TAL_calls=128, blank_geometry_samples=64, JPEG_decodes=0,
                   detector_forwards=0, fits=0, gradients=0, parameter_updates=0, AP_evaluations=0,
                   elapsed_seconds=perf_counter()-started),
        artifacts_sha256={p.name: file_hash(p) for p in root.iterdir() if p.is_file()})
    save(root/"result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
