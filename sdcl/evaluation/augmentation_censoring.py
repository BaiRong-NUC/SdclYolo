"""Audit actual label rejection in a frozen geometric augmentation prefix."""

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
from time import perf_counter
import zipfile

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import augment as augment_module
from ultralytics.data.augment import Mosaic, RandomPerspective
from ultralytics.utils import loss as loss_module
from ultralytics.utils import tal as tal_module
from ultralytics.utils.tal import make_anchors
import yaml

from sdcl.config import SDCLConfig, project_path
from sdcl.evaluation.phase_screen import BASELINE, BASELINE_SHA256
from sdcl.evaluation.size_analysis import file_hash
from sdcl.data.dataset import IgnoreDataset
from sdcl.postprocess.rescue import save_json
from sdcl.training.criterion import SDCLLoss
from sdcl.training.trainer import check_version


REGISTRATION = "docs/experiments/candidate54_augmentation_censoring_registration_20261015.json"
OUTPUT = "output/analysis/augmentation-censoring-fixed-20261015"
BASELINE_ARGS = "output/runs/baseline_yolo11s_960_100ep_seed0/args.yaml"
GROUPS = {"all": float("inf"), "tiny": 256., "small_all": 1024.}
STEM_PATTERN = r"\d{7}_\d{5}_[a-z]_\d{7}"


def contains(boxes, rectangle):
    boxes = np.asarray(boxes)
    return ((boxes[:, :2] >= rectangle[:2]).all(1) &
            (boxes[:, 2:] <= rectangle[2:]).all(1) &
            (boxes[:, 2:] > boxes[:, :2]).all(1))


class TrackedMosaic(Mosaic):
    def __init__(self, dataset, donors):
        super().__init__(dataset, imgsz=960, p=1., n=4)
        self.donors = donors

    def get_indexes(self):
        return self.donors

    def apply_instances(self, labels, params=None):
        boxes, flags, classes, records = [], [], [], []
        for item in params["layout"]:
            source = item["labels_patch"]
            instances = deepcopy(source["instances"])
            instances.convert_bbox(format="xyxy")
            h, w = item["img_shape"]
            instances.denormalize(w, h)
            original = instances.bboxes.copy()
            crop = np.array([item[k] for k in ("x1b", "y1b", "x2b", "y2b")])
            flags.append(contains(original, crop))
            shifted = original.copy()
            shifted[:, [0, 2]] += item["padw"]
            shifted[:, [1, 3]] += item["padh"]
            boxes.append(shifted)
            classes.append(source["cls"].copy().flatten())
            records.append(dict(stem=Path(source["im_file"]).stem, resized_shape=[h, w],
                                crop=crop.tolist(), offset=[item["padw"], item["padh"]],
                                count=len(original)))
        boxes, flags, classes = np.concatenate(boxes), np.concatenate(flags), np.concatenate(classes)
        clipped = boxes.clip(0, 1920)
        good = (clipped[:, 2:] > clipped[:, :2]).all(1)
        result = super().apply_instances(labels, params)
        np.testing.assert_array_equal(result["instances"].bboxes, clipped[good])
        np.testing.assert_array_equal(result["cls"].flatten(), classes[good])
        self.audit = dict(source_boxes=boxes, source_classes=classes, source_fully_visible=flags,
                          mosaic_good=good, mosaic_boxes=clipped[good], mosaic_classes=classes[good],
                          mosaic_fully_visible=flags[good])
        self.layout = records
        return result


class TrackedPerspective(RandomPerspective):
    def get_params(self, labels):
        params = super().get_params(labels)
        self.params = params
        return params

    def apply_instances(self, labels, params=None):
        instances = deepcopy(labels["instances"])
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])
        self.unclipped = self.apply_bboxes(instances.bboxes, params["M"])
        return super().apply_instances(labels, params)

    def box_candidates(self, box1, box2, **kwargs):
        good = RandomPerspective.box_candidates(box1, box2, **kwargs)
        self.box1, self.box2, self.keep = box1.T.copy(), box2.T.copy(), good.copy()
        return good


def geometry_frame(mosaic, affine):
    frame = dict(mosaic.audit)
    frame.update(matrix=affine.params["M"], scale=np.array(affine.params["scale"]),
                 affine_unclipped=affine.unclipped, scaled_reference=affine.box1,
                 transformed_boxes=affine.box2, keep=affine.keep)
    boxes = frame["transformed_boxes"]
    w, h = (boxes[:, 2:]-boxes[:, :2]).T
    rw, rh = (frame["scaled_reference"][:, 2:]-frame["scaled_reference"][:, :2]).T
    frame["fully_visible"] = frame["mosaic_fully_visible"] & contains(frame["affine_unclipped"], [0, 0, 960, 960])
    frame["valid_gt"] = (frame["mosaic_classes"] >= 0) & (frame["mosaic_classes"] < 10)
    frame["areas"] = w*h*(640/960)**2
    frame["size_rejected"] = (w <= 2) | (h <= 2)
    frame["area_rejected"] = w*h/(rw*rh+1e-16) <= .1
    frame["aspect_rejected"] = np.maximum(w/(h+1e-16), h/(w+1e-16)) >= 100
    result = {}
    for group, limit in GROUPS.items():
        selected = frame["valid_gt"] & frame["fully_visible"] & (frame["areas"] < limit)
        rejected = selected & ~frame["keep"]
        result[group] = dict(fully_visible_gt=int(selected.sum()), rejected_gt=int(rejected.sum()),
                             affected_mosaics=int(rejected.any()),
                             size_rejected_gt=int((rejected & frame["size_rejected"]).sum()),
                             area_rejected_gt=int((rejected & frame["area_rejected"]).sum()),
                             aspect_rejected_gt=int((rejected & frame["aspect_rejected"]).sum()))
    return frame, result


def aggregate(traces):
    result = {}
    for half, items in (("screen", traces[:32]), ("confirmation", traces[32:])):
        result[half] = {}
        for group in GROUPS:
            total = {k: sum(t[group][k] for t in items) for k in items[0][group]}
            denominator = total["fully_visible_gt"]
            total["rejected_fraction"] = total["rejected_gt"]/denominator if denominator else 0.
            if "supported_gt" in total:
                total["supported_fraction"] = total["supported_gt"]/denominator if denominator else 0.
            result[half][group] = total
    return result


def geometry_checks(metrics, policy):
    return {half: dict(denominator=values["tiny"]["fully_visible_gt"] >= policy["fully_visible_tiny_gt_per_half_min"],
                       rejected=values["tiny"]["rejected_gt"] >= policy["fully_visible_discarded_tiny_per_half_min"],
                       mosaics=values["tiny"]["affected_mosaics"] >= policy["affected_mosaics_per_half_min"],
                       fraction=values["tiny"]["rejected_fraction"] >= policy["discarded_tiny_fraction_per_half_min"])
            for half, values in metrics.items()}


def pairwise_iou(a, b):
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])-
                              np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    union = (a[:, 2:]-a[:, :2]).prod(1)[:, None]+(b[:, 2:]-b[:, :2]).prod(1)[None]-intersection
    return intersection/np.maximum(union, 1e-12)


def native_followup(frame, raw, criterion):
    device = raw["scores"].device
    anchors, stride = make_anchors(raw["feats"], criterion.stride, .5)
    probability = raw["scores"].permute(0, 2, 1).sigmoid()
    decoded = criterion.bbox_decode(anchors, raw["boxes"].permute(0, 2, 1))*stride
    gt, cls = frame["transformed_boxes"], frame["mosaic_classes"].astype(int)
    retained = frame["keep"] & frame["valid_gt"]
    tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
    _, _, _, foreground, _ = criterion.assigner(
        probability, decoded, anchors*stride, tensor(cls[retained])[None, :, None],
        tensor(gt[retained])[None], torch.ones((1, int(retained.sum()), 1), device=device, dtype=torch.bool))
    to_numpy = lambda t: t.detach().cpu().numpy()
    frame.update(native_probability=to_numpy(probability[0]), native_boxes=to_numpy(decoded[0]),
                 native_foreground=to_numpy(foreground[0]), native_anchors=to_numpy(anchors*stride))
    ignores = gt[frame["keep"] & (cls == 10)]
    ignored = np.zeros(len(anchors), bool)
    for box in ignores:
        ignored |= ((frame["native_anchors"] >= box[:2]) & (frame["native_anchors"] <= box[2:])).all(1)
    frame["native_ignored"] = ignored
    eligible = ~frame["native_foreground"] & ~ignored
    supported = np.zeros(len(gt), bool)
    argmax = frame["native_probability"].argmax(1)
    for g in np.flatnonzero(frame["fully_visible"] & frame["valid_gt"] & ~frame["keep"]):
        valid = eligible & (argmax == cls[g]) & (frame["native_probability"][:, cls[g]] >= .05)
        supported[g] = bool((pairwise_iou(gt[g:g+1], frame["native_boxes"][valid]) >= .5).any())
    frame["native_supported"] = supported
    return frame


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    check_version()
    torch.set_num_threads(1)
    output = project_path(output)
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(project_path(REGISTRATION).read_text())
    exclusions, excluded = {}, set()
    for p in sorted(project_path("output/analysis").glob("*/plan.json")):
        exclusions[p.relative_to(project_path(".")).as_posix()] = file_hash(p)
        excluded.update(re.findall(STEM_PATTERN, p.read_text(encoding="utf-8")))
    images = sorted(project_path("data/visdrone/images/train").glob("*.jpg"),
                    key=lambda p: hashlib.sha256(("candidate54-visible-censor-v1:"+p.stem).encode()).hexdigest())
    chosen = [p for p in images if p.stem not in excluded][:256]
    assert len(chosen) == 256 and len(set(p.stem for p in chosen)) == 256
    args_file = project_path(BASELINE_ARGS)
    old_args = yaml.safe_load(args_file.read_text(encoding="utf-8"))
    affine_args = {k: old_args[k] for k in ("degrees", "translate", "scale", "shear", "perspective")}
    assert affine_args == dict(degrees=0., translate=.1, scale=.5, shear=0., perspective=0.)
    paths = (REGISTRATION, "sdcl/evaluation/augmentation_censoring.py",
             "scripts/evaluation/diagnose_augmentation_censoring.py", "tests/test_augmentation_censoring.py",
             "sdcl/config.py", "sdcl/data/dataset.py", "sdcl/training/criterion.py",
             "sdcl/training/trainer.py", "sdcl/evaluation/phase_screen.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/rescue.py", BASELINE_ARGS)
    hashes = {name: file_hash(project_path(name)) for name in paths}
    installed = {name: Path(inspect.getfile(module)) for name, module in
                 (("augment.py", augment_module), ("tal.py", tal_module), ("loss.py", loss_module))}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in paths:
            z.write(project_path(name), name)
        for name, path in installed.items():
            z.write(path, "installed/"+name)
    plan = dict(registration=registration, source_hashes=hashes,
                source_archive_sha256=file_hash(output/"source.zip"),
                installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
                exclusions=exclusions, excluded_unique_stems=len(excluded),
                images=[p.stem for p in chosen], affine_args=affine_args,
                checkpoint=BASELINE, checkpoint_sha256=BASELINE_SHA256,
                sources={p.stem: {kind: file_hash(path) for kind, path in
                         (("image", p), ("metadata", project_path(f"data/visdrone/metadata/train/{p.stem}.json")),
                          ("label", project_path(f"data/visdrone/labels/train/{p.stem}.txt")))} for p in chosen},
                seeds=[int.from_bytes(hashlib.sha256(f"candidate54-warp-v1:{i}".encode()).digest()[:4], "little")
                       for i in range(64)], selection_saved_before_first_decode=True)
    save_json(output/"plan.json", plan)
    args = get_cfg(overrides=dict(imgsz=960, batch=1, workers=0, rect=False, plots=False))
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = IgnoreDataset(img_path=str(data_path.parent/"images/train"), imgsz=960, batch_size=1,
                            augment=False, hyp=args, rect=False, cache=False, single_cls=False,
                            stride=32, pad=0., prefix="censoring: ", task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    traces, manifest, layouts = [], {}, []
    for i in range(64):
        seed = plan["seeds"][i]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ids = [indices[p.stem] for p in chosen[4*i:4*i+4]]
        mosaic = TrackedMosaic(dataset, ids[1:])
        affine = TrackedPerspective(**affine_args, size=(960, 960))
        augmented = affine(mosaic(dataset.get_image_and_label(ids[0])))
        frame, trace = geometry_frame(mosaic, affine)
        np.testing.assert_array_equal(augmented["instances"].bboxes, frame["transformed_boxes"][frame["keep"]])
        np.testing.assert_array_equal(augmented["cls"].flatten(), frame["mosaic_classes"][frame["keep"]])
        assert augmented["img"].shape == (960, 960, 3)
        frame["image"] = augmented["img"]
        path = output/"frames"/f"{i:02d}.npz"
        np.savez_compressed(path, **frame)
        manifest[str(i)] = dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path))
        layouts.append(mosaic.layout)
        traces.append(trace)
        if (i+1) % 16 == 0:
            print(f"Augmentation label audit: {i+1}/64", flush=True)
    metrics = aggregate(traces)
    first_checks = geometry_checks(metrics, registration["first_stage_policy"])
    first_pass = all(all(values.values()) for values in first_checks.values())
    native_checks, forwards = None, 0
    if first_pass:
        assert device == "0" and torch.cuda.is_available()
        assert file_hash(project_path(BASELINE)) == BASELINE_SHA256
        model = YOLO(str(project_path(BASELINE))).model.float().cuda().eval()
        model.args = args
        model.sdcl_settings = SDCLConfig(enabled=False).to_dict()
        model.sdcl_ignore_regions = True
        model.coverage_settings = {"enabled": False}
        criterion = SDCLLoss(model)
        for i in range(64):
            path = output/manifest[str(i)]["file"]
            with np.load(path, allow_pickle=False) as z:
                frame = {k: z[k].copy() for k in z.files}
            image = torch.from_numpy(frame["image"][:, :, ::-1].transpose(2, 0, 1).copy()).cuda()[None].float()/255
            with torch.no_grad():
                raw = model(image)[1]
                frame = native_followup(frame, raw, criterion)
            forwards += 1
            for group, limit in GROUPS.items():
                selected = frame["fully_visible"] & frame["valid_gt"] & (frame["areas"] < limit)
                supported = selected & frame["native_supported"]
                traces[i][group].update(supported_gt=int(supported.sum()), supported_mosaics=int(supported.any()))
            np.savez_compressed(path, **frame)
            manifest[str(i)]["sha256"] = file_hash(path)
        metrics = aggregate(traces)
        p = registration["second_stage_policy"]
        native_checks = {half: dict(supported=v["tiny"]["supported_gt"] >= p["supported_discarded_tiny_per_half_min"],
                                   mosaics=v["tiny"]["supported_mosaics"] >= p["supported_mosaics_per_half_min"],
                                   fraction=v["tiny"]["supported_fraction"] >= p["supported_tiny_fraction_per_half_min"])
                         for half, v in metrics.items()}
    passed = first_pass and all(all(values.values()) for values in native_checks.values())
    decision = dict(decision="REPLICATED_VISIBLE_LABEL_CENSORING_SUPPORTS_SEPARATE_CONTROLLED_PILOT" if passed else
                    "STOP_THIS_FULLY_VISIBLE_AUGMENTATION_CENSORING_ROUTE",
                    geometry_checks=first_checks, native_checks=native_checks,
                    native_stage_measured=first_pass,
                    scope="Actual geometric-prefix label filtering only; no historical replay, harmful total-gradient, detection gain, AP, training or novelty proof.")
    verification = dict(source_JPEG_load_requests=256, geometric_augmentations=64,
                        frozen_detector_forwards=forwards, native_TAL_calls=forwards, parameter_updates=0,
                        AP_evaluations=0, previous_diagnostic_stems_excluded=True,
                        source_unchanged=all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                        installed_source_unchanged=all(file_hash(p) == plan["installed_sources"][n]["sha256"] for n, p in installed.items()),
                        elapsed_seconds=perf_counter()-started)
    for name, value in (("frames.json", manifest), ("layouts.json", layouts), ("per_image.json", traces),
                        ("metrics.json", metrics), ("decision.json", decision), ("verification.json", verification)):
        save_json(output/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(output),
                  metrics=metrics, decision=decision, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output/"result.json", result)
    return result
