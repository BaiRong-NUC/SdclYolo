"""Measure actual tiny-positive supervision outside nonnegative ltrb support."""

import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import loss as loss_module
from ultralytics.utils import tal as tal_module
from ultralytics.utils.tal import make_anchors
import yaml

from sdcl.config import SDCLConfig, project_path
from sdcl.data.dataset import IgnoreDataset
from sdcl.evaluation.phase_screen import BASELINE, BASELINE_SHA256
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json
from sdcl.training.criterion import SDCLLoss
from sdcl.training.trainer import check_version


REGISTRATION = "docs/experiments/candidate50_assignment_reachability_registration_20261014.json"
OUTPUT = "output/analysis/assignment-reachability-fixed-20261014"
EXCLUSIONS = ("output/analysis/visual-preflight-fixed-20261014/plan.json",
              "output/analysis/side-composition-pilot-fixed-20261014/plan.json")
GROUPS = {"all": float("inf"), "tiny": 256., "small_all": 1024.}


def enclosure_upper_bound(boxes, points):
    boxes, points = np.asarray(boxes, float), np.asarray(points, float)
    if boxes.shape[-1] != 4 or points.shape[-1] != 2:
        raise ValueError("Expected xyxy boxes and xy points.")
    area = np.maximum(boxes[..., 2:]-boxes[..., :2], 0).prod(-1)
    enclosure = (np.maximum(boxes[..., 2:], points)-np.minimum(boxes[..., :2], points)).prod(-1)
    return area/np.maximum(enclosure, 1e-12)


def group_statistics(frame, ceiling):
    gt, owner = frame["gt"], frame["owner"]
    area = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)*(640/960)**2
    mass = frame["target_mass"]
    positive = frame["foreground"] & (mass > 0)
    point = frame["anchors"]
    assigned_boxes = gt[owner]
    outside = ((point < assigned_boxes[:, :2]) | (point > assigned_boxes[:, 2:])).any(1)
    low = positive & (ceiling < .75)
    assigned_max = np.zeros(len(gt))
    assigned_inside = np.zeros(len(gt), bool)
    for g in range(len(gt)):
        mask = positive & (owner == g)
        if mask.any():
            assigned_max[g] = ceiling[mask].max()
            assigned_inside[g] = (~outside[mask]).any()
    all_grid_max = np.array([enclosure_upper_bound(b, point).max() for b in gt])
    result = {}
    for name, limit in GROUPS.items():
        selected_gt = area < limit
        selected_pair = positive & selected_gt[owner]
        selected_low = low & selected_gt[owner]
        affected = np.unique(owner[selected_low])
        total_mass = float(mass[selected_pair].sum())
        gap_mass = float((mass[selected_pair]*(1-ceiling[selected_pair])).sum())
        result[name] = {
            "gt": int(selected_gt.sum()), "covered_gt": int((selected_gt & (assigned_max > 0)).sum()),
            "productive_positive_pairs": int(selected_pair.sum()), "target_mass": total_mass,
            "outside_pairs": int((selected_pair & outside).sum()),
            "low_ceiling_pairs": int(selected_low.sum()),
            "low_ceiling_mass": float(mass[selected_low].sum()),
            "unattainable_iou_gap_mass": gap_mass,
            "low_ceiling_affected_gt": int(len(affected)),
            "low_ceiling_affected_images": int(bool(len(affected))),
            "affected_with_inside_assigned_alternative": int(assigned_inside[affected].sum()),
            "affected_with_inside_grid_but_no_inside_assigned": int(((all_grid_max[affected] >= 1-1e-12) & ~assigned_inside[affected]).sum()),
            "affected_without_any_inside_grid_point": int((all_grid_max[affected] < 1-1e-12).sum()),
            "covered_gt_all_assigned_below075": int((selected_gt & (assigned_max > 0) & (assigned_max < .75)).sum()),
            "gt_all_grid_below075": int((selected_gt & (all_grid_max < .75)).sum()),
        }
    return result, assigned_max, all_grid_max


def decide(metrics, policy):
    tiny = metrics["tiny"]
    checks = {
        "affected_gt": tiny["low_ceiling_affected_gt"] >= policy["tiny_low_ceiling_affected_gt_min"],
        "affected_images": tiny["low_ceiling_affected_images"] >= policy["tiny_low_ceiling_affected_images_min"],
        "mass_fraction": tiny["low_ceiling_mass_fraction"] >= policy["tiny_low_ceiling_mass_fraction_min"],
        "weighted_gap": tiny["weighted_unattainable_iou_gap"] >= policy["tiny_weighted_unattainable_iou_gap_min"],
    }
    return {
        "decision": "MECHANISM_SUPPORTS_SEPARATE_BOUNDED_REPARAMETERIZATION_SCREEN" if all(checks.values()) else "STOP_THIS_NONNEGATIVE_SUPPORT_ROUTE_ON_FIXED_PROXY",
        "checks": checks,
        "scope": "Assignment/representation mechanism only, not harmful-gradient, AP, official evaluation, novelty or training authorization.",
    }


def run(output=OUTPUT, device="0"):
    import hashlib

    started = perf_counter()
    torch.set_num_threads(1)
    check_version()
    if device != "0" or not torch.cuda.is_available():
        raise ValueError("Fixed probe requires CUDA device0.")
    output = project_path(output)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = project_path(BASELINE)
    assert file_hash(checkpoint) == BASELINE_SHA256
    registration = json.loads(project_path(REGISTRATION).read_text())
    excluded = set()
    for name in EXCLUSIONS:
        selection = json.loads(project_path(name).read_text())["image_selection"]
        excluded.update(stem for stems in selection.values() for stem in stems)
    images = sorted(project_path("data/visdrone/images/train").glob("*.jpg"),
                    key=lambda p: hashlib.sha256(("candidate50-support-v1:"+p.stem).encode()).hexdigest())
    selected = [p for p in images if p.stem not in excluded][:64]
    assert len(selected) == 64
    paths = (REGISTRATION, "sdcl/evaluation/assignment_reachability.py",
             "scripts/evaluation/diagnose_assignment_reachability.py", "tests/test_assignment_reachability.py",
             "sdcl/training/criterion.py", "sdcl/data/dataset.py", "sdcl/training/trainer.py",
             "sdcl/config.py", "sdcl/evaluation/size_analysis.py")
    hashes = {name: file_hash(project_path(name)) for name in paths}
    installed = {name: Path(inspect.getfile(module)) for name, module in
                 (("tal.py", tal_module), ("loss.py", loss_module))}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(project_path(name), name)
        for name, path in installed.items():
            archive.write(path, "installed/"+name)
    plan = {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output/"source.zip"),
        "installed_sources": {name: {"path": str(p), "sha256": file_hash(p)} for name, p in installed.items()},
        "checkpoint": BASELINE, "checkpoint_sha256": BASELINE_SHA256,
        "exclusions": {name: file_hash(project_path(name)) for name in EXCLUSIONS},
        "excluded_images": len(excluded), "images": [p.stem for p in selected],
        "selection_saved_before_image_decode_or_model_forward": True,
    }
    save_json(output/"plan.json", plan)
    args = get_cfg(overrides={"imgsz": 960, "batch": 1, "workers": 0, "rect": False,
                              "plots": False, "device": "0"})
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = IgnoreDataset(img_path=str(data_path.parent/"images/train"), imgsz=960,
                            batch_size=1, augment=False, hyp=args, rect=False, cache=False,
                            single_cls=False, stride=32, pad=0., prefix="support-proxy: ",
                            task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    model = YOLO(str(checkpoint)).model.float().cuda().eval()
    model.args = args
    model.sdcl_settings = SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions = True
    model.coverage_settings = {"enabled": False}
    criterion = SDCLLoss(model)
    assert type(criterion.assigner).__name__ == "TaskAlignedAssigner"
    policy = {k: getattr(criterion.assigner, k) for k in ("topk", "topk2", "alpha", "beta", "stride_val", "eps")}
    plan["actual_assigner"] = policy
    save_json(output/"plan.json", plan)
    totals, traces, manifest = None, [], {}
    for n, image in enumerate(selected, 1):
        batch = dataset.collate_fn([dataset[indices[image.stem]]])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float()/255
        assert tuple(batch["img"].shape) == (1, 3, 960, 960)
        with torch.no_grad():
            raw = model(batch["img"])[1]
            anchors, stride = make_anchors(raw["feats"], criterion.stride, .5)
            distances = raw["boxes"].permute(0, 2, 1).contiguous()
            probabilities = raw["scores"].permute(0, 2, 1).sigmoid().contiguous()
            target_input = torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1)
            targets = criterion.preprocess(target_input, 1, probabilities.new_tensor([960]*4))
            labels, gt = targets.split((1, 4), 2)
            valid = gt.sum(2, keepdim=True) > 0
            decoded = criterion.bbox_decode(anchors, distances)*stride
            _, assigned, scores, foreground, owner = criterion.assigner(
                probabilities, decoded, anchors*stride, labels, gt, valid)
        assert valid.all() and len(gt[0]) > 0
        as_numpy = lambda t: t.detach().cpu().numpy().copy()
        frame = dict(gt=as_numpy(gt[0]), classes=as_numpy(labels[0, :, 0]),
                     anchors=as_numpy(anchors*stride), stride=as_numpy(stride),
                     probabilities=as_numpy(probabilities[0]), decoded_boxes=as_numpy(decoded[0]),
                     owner=as_numpy(owner[0]), foreground=as_numpy(foreground[0]),
                     target_mass=as_numpy(scores[0].sum(-1)), assigned_boxes=as_numpy(assigned[0]))
        ceiling = enclosure_upper_bound(frame["gt"][frame["owner"]], frame["anchors"])
        metrics, assigned_max, grid_max = group_statistics(frame, ceiling)
        frame.update(ceiling=ceiling, assigned_max=assigned_max, grid_max=grid_max)
        path = output/"frames"/(image.stem+".npz")
        np.savez_compressed(path, **frame)
        metadata = data_path.parent/"metadata/train"/(image.stem+".json")
        label = data_path.parent/"labels/train"/(image.stem+".txt")
        manifest[image.stem] = dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path),
                                    image_sha256=file_hash(image), metadata_sha256=file_hash(metadata),
                                    label_sha256=file_hash(label))
        traces.append(metrics)
        if totals is None:
            totals = {g: dict.fromkeys(metrics[g], 0) for g in GROUPS}
        for group, values in metrics.items():
            for key, value in values.items():
                totals[group][key] += value
        if n % 16 == 0:
            print(f"Assignment reachability: {n}/64", flush=True)
    for values in totals.values():
        mass = values["target_mass"]
        values["low_ceiling_mass_fraction"] = values["low_ceiling_mass"]/mass if mass else 0.
        values["weighted_unattainable_iou_gap"] = values["unattainable_iou_gap_mass"]/mass if mass else 0.
    decision = decide(totals, registration["policy"])
    verification = dict(image_decodes=64, frozen_detector_forwards=64, detector_updates=0,
                        fitting_steps=0, AP_evaluations=0, no_validation_or_test_images=True,
                        source_unchanged=all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                        installed_source_unchanged=all(file_hash(p) == plan["installed_sources"][n]["sha256"] for n, p in installed.items()),
                        elapsed_seconds=perf_counter()-started)
    for name, value in (("frames.json", manifest), ("metrics.json", totals), ("per_image.json", traces),
                        ("decision.json", decision), ("verification.json", verification)):
        save_json(output/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-14", output=str(output),
                  decision=decision, metrics=totals, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output/"result.json", result)
    return result
