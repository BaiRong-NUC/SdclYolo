"""Cached absolute-pixel versus relative-box CIoU gradient prerequisite."""

import inspect
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from ultralytics.utils import metrics as metrics_module
from ultralytics.utils.metrics import bbox_iou

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json


REGISTRATION = "docs/experiments/candidate53_label_gradient_registration_20261014.json"
OUTPUT = "output/analysis/label-gradient-fixed-20261014"
PRIOR = "output/analysis/assignment-reachability-fixed-20261014"
SIGNS = np.array(list(product((-1., 1.), repeat=4)))
GROUPS = {"all": np.inf, "tiny": 256., "small_all": 1024.}


def amplitudes_and_guard(gt, original_size):
    width, height = original_size
    ratio = 960/max(width, height)
    rw, rh = min(math.ceil(width*ratio), 960), min(math.ceil(height*ratio), 960)
    left, top = round((960-rw)/2-.1), round((960-rh)/2-.1)
    absolute = np.broadcast_to(.5*np.array([rw/width, rh/height, rw/width, rh/height]), gt.shape).copy()
    wh = gt[:, 2:]-gt[:, :2]
    relative = np.tile(wh/32, (1, 2))
    delta = np.maximum(absolute, relative)
    guard = ((gt[:, 2:]-delta[:, 2:] > gt[:, :2]+delta[:, :2]).all(1) &
             (gt[:, :2]-delta[:, :2] >= [left, top]).all(1) &
             (gt[:, 2:]+delta[:, 2:] <= [left+rw, top+rh]).all(1))
    return absolute, relative, guard


def coordinate_gradient(prediction, target, stride):
    coordinates = torch.tensor(prediction, dtype=torch.float64, requires_grad=True)
    gt = torch.tensor(target, dtype=torch.float64)
    step = torch.tensor(stride, dtype=torch.float64)
    loss = 1-bbox_iou(coordinates/step, gt/step, xywh=False, CIoU=True, eps=1e-7)
    gradient = torch.autograd.grad(loss.sum(), coordinates)[0].detach().numpy()
    if not np.isfinite(gradient).all():
        raise ValueError("Nonfinite CIoU coordinate gradient.")
    return gradient


def direction_flip(nominal, perturbed):
    return (np.asarray(perturbed, float)*np.asarray(nominal, float)[None]).sum(-1) < 0


def summarize(traces):
    metrics = {}
    for group in GROUPS:
        values = {}
        for key in traces[0][group]:
            values[key] = sum(t[group][key] for t in traces)
        mass = values["mass"]
        values["absolute_flip_rate"] = values["absolute_flip_mass"]/mass if mass else 0.
        values["relative_flip_rate"] = values["relative_flip_mass"]/mass if mass else 0.
        values["absolute_minus_relative_flip_rate"] = values["absolute_flip_rate"]-values["relative_flip_rate"]
        metrics[group] = values
    weights = np.random.default_rng(0).multinomial(len(traces), np.full(len(traces), 1/len(traces)), size=400)
    mass = np.array([t["tiny"]["mass"] for t in traces])
    delta = np.array([t["tiny"]["absolute_flip_mass"]-t["tiny"]["relative_flip_mass"] for t in traces])
    denominators = weights@mass
    interval = (np.percentile(weights@delta/denominators, [2.5, 97.5]).tolist()
                if np.all(denominators > 0) else None)
    metrics["tiny"]["absolute_minus_relative_image_bootstrap95ci"] = interval
    return metrics


def decide(metrics, policy):
    tiny = metrics["tiny"]
    interval = tiny["absolute_minus_relative_image_bootstrap95ci"]
    checks = dict(covered_gt=tiny["productive_covered_gt"] >= policy["tiny_productive_covered_gt_min"],
                  images=tiny["productive_images"] >= policy["tiny_productive_images_min"],
                  absolute_flip=tiny["absolute_flip_rate"] >= policy["tiny_absolute_flip_rate_min"],
                  excess=tiny["absolute_minus_relative_flip_rate"] >= policy["tiny_absolute_minus_relative_flip_rate_min"],
                  interval=interval is not None and interval[0] > policy["tiny_difference_bootstrap_lower_min"])
    return dict(decision="PIXEL_SPECIFIC_GRADIENT_SIGNAL_SUPPORTS_SEPARATE_LOSS_PILOT" if all(checks.values()) else
                "STOP_THIS_HALF_PIXEL_SPECIFIC_CIOU_ROUTE", checks=checks,
                scope="Synthetic noise, isolated coordinate-loss derivative only; no actual annotation-error, model-update, detection efficacy or novelty evidence.")


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    output, prior = project_path(output), project_path(PRIOR)
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(project_path(REGISTRATION).read_text())
    previous = json.loads((prior/"plan.json").read_text())
    old_result = json.loads((prior/"result.json").read_text())
    for name, digest in old_result["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    prior_frames = json.loads((prior/"frames.json").read_text())
    paths = (REGISTRATION, "sdcl/evaluation/label_gradient_sensitivity.py",
             "scripts/evaluation/diagnose_label_gradient.py", "tests/test_label_gradient_sensitivity.py",
             "sdcl/config.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    installed = Path(inspect.getfile(metrics_module))
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        z.write(installed, "installed/metrics.py")
    plan = dict(registration=registration, source_hashes=hashes, source_archive_sha256=file_hash(output/"source.zip"),
                installed_source=dict(path=str(installed), sha256=file_hash(installed)),
                prior_plan_sha256=file_hash(prior/"plan.json"), prior_result_sha256=file_hash(prior/"result.json"),
                images=previous["images"], signs=SIGNS.tolist())
    save_json(output/"plan.json", plan)
    traces, manifest = [], {}
    gradient_batches = 0
    for index, stem in enumerate(plan["images"], 1):
        item = prior_frames[stem]
        assert file_hash(prior/item["file"]) == item["sha256"]
        with np.load(prior/item["file"]) as z:
            f = {k: z[k].copy() for k in z.files}
        metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        assert file_hash(metadata_path) == item["metadata_sha256"]
        metadata = json.loads(metadata_path.read_text())
        gt = f["gt"].astype(float)
        absolute, relative, guard = amplitudes_and_guard(gt, metadata["original_size"])
        positive = f["foreground"] & (f["target_mass"] > 0)
        rows = np.flatnonzero(positive & guard[f["owner"]])
        owners = f["owner"][rows]
        prediction, target, stride = f["decoded_boxes"][rows], gt[owners], f["stride"][rows]
        mass = f["target_mass"][rows].astype(float)
        if len(rows):
            nominal = coordinate_gradient(prediction, target, stride)
            perturbed = np.stack([np.stack([coordinate_gradient(prediction, target+sign*amplitude[owners], stride)
                                           for sign in SIGNS]) for amplitude in (absolute, relative)])
            gradient_batches += 33
        else:
            nominal, perturbed = np.zeros((0, 4)), np.zeros((2, 16, 0, 4))
        valid = np.linalg.norm(nominal, axis=1) > 1e-12
        flip = np.stack([direction_flip(nominal, values) for values in perturbed])
        probabilities = flip.mean(1)
        areas = (gt[:, 2:]-gt[:, :2]).prod(1)*(640/960)**2
        trace = {}
        for group, limit in GROUPS.items():
            selected_gt = areas < limit
            selected = selected_gt[owners]
            selected_valid = selected & valid
            ids = np.unique(owners[selected_valid])
            affected = np.unique(owners[selected_valid & flip[0].any(0)])
            trace[group] = dict(gt=int(selected_gt.sum()), guard_gt=int((selected_gt & guard).sum()),
                productive_rows=int(selected.sum()), productive_covered_gt=int(len(ids)), productive_images=int(bool(len(ids))),
                zero_nominal_rows=int((selected & ~valid).sum()), zero_nominal_mass=float(mass[selected & ~valid].sum()),
                mass=float(mass[selected_valid].sum()), affected_gt=int(len(affected)), affected_images=int(bool(len(affected))),
                absolute_flip_mass=float((mass[selected_valid]*probabilities[0, selected_valid]).sum()),
                relative_flip_mass=float((mass[selected_valid]*probabilities[1, selected_valid]).sum()))
        traces.append(trace)
        path = output/"frames"/(stem+".npz")
        np.savez_compressed(path, rows=rows, owners=owners, prediction=prediction, target=target, stride=stride,
                            mass=mass, all_gt=gt, all_areas=areas, absolute=absolute, relative=relative, guard=guard,
                            nominal=nominal, perturbed=perturbed, valid=valid, flip=flip, probabilities=probabilities)
        manifest[stem] = dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path),
                             prior_frame_sha256=item["sha256"], metadata_sha256=item["metadata_sha256"])
        if index % 16 == 0:
            print(f"Label coordinate-gradient audit: {index}/64", flush=True)
    metrics = summarize(traces)
    decision = decide(metrics, registration["policy"])
    verification = dict(image_decodes=0, detector_forwards=0, coordinate_gradient_batches=gradient_batches,
                        detector_updates=0, AP_evaluations=0, complete64_cached_frames=True,
                        source_unchanged=all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                        installed_source_unchanged=file_hash(installed) == plan["installed_source"]["sha256"],
                        elapsed_seconds=perf_counter()-started)
    for name, value in (("frames.json", manifest), ("per_image.json", traces), ("metrics.json", metrics),
                        ("decision.json", decision), ("verification.json", verification)):
        save_json(output/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-14", output=str(output),
                  metrics=metrics, decision=decision, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output/"result.json", result)
    return result
