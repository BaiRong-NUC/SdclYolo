"""Reproducible feature and detection sensitivity on sub-stride translations."""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision  # Enables the compiled NMS kernel in Ultralytics.
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import nms
from ultralytics.utils.ops import xywh2xyxy

from .config import load_experiment, project_path
from .geometry import prediction_ignore_mask
from .phase import PhaseConfig, SHIFTS, object_samples, phase_distance, translate_images
from .size_analysis import box_iou_matrix, file_hash, match_at_threshold
from .trainer import check_version, make_dataset


PROBE_SHIFTS = SHIFTS[:4]


def probe_detections(prediction, boxes, classes, ignores):
    detections = nms.non_max_suppression(
        prediction, conf_thres=0.25, iou_thres=0.7, multi_label=True, max_det=500,
    )[0]
    detections = detections[~prediction_ignore_mask(detections[:, :4], ignores)]
    detections = detections.float().cpu().numpy()
    matches, _ = match_at_threshold(
        box_iou_matrix(boxes, detections[:, :4]), classes, detections,
        confidence=0.25, threshold=0.5,
    )
    return matches >= 0


def run_phase_probe(checkpoint, output, device="0", images=64):
    check_version()
    checkpoint, output = project_path(checkpoint), project_path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite probe: {output}")
    if images < 1:
        raise ValueError("images must be positive.")
    args, _, _ = load_experiment("configs/experiments/baseline_960_100.yaml")
    args.update(batch=1, rect=True)
    args = get_cfg(overrides=args)
    data_path = Path(args.data)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = make_dataset(args, str(data_path.parent / "images/val"), data, "val", 1, 32, True)
    indices = np.linspace(0, len(dataset)-1, min(images, len(dataset))).astype(int)
    model = YOLO(str(checkpoint)).model.to("cuda:"+str(device) if str(device) != "cpu" else "cpu").float().eval()
    config = PhaseConfig(max_objects_per_image=10000)
    rows = []
    selected_hashes = {}
    totals = {group: {"distance_sum": 0.0, "objects": 0,
                      "flips": 0, "comparisons": 0}
              for group in ("tiny", "small", "medium", "large")}
    with torch.inference_mode():
        for index in indices:
            batch = dataset.collate_fn([dataset[int(index)]])
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch["img"] = batch["img"].float() / 255
            image_path = Path(batch["im_file"][0])
            for path in (image_path, data_path.parent / "labels/val" / f"{image_path.stem}.txt",
                         data_path.parent / "metadata/val" / f"{image_path.stem}.json"):
                selected_hashes[str(path)] = file_hash(path)
            height, width = batch["img"].shape[-2:]
            scale = batch["bboxes"].new_tensor([width, height] * 2)
            boxes = xywh2xyxy(batch["bboxes"]) * scale
            ignores = xywh2xyxy(batch["ignore_bboxes"]) * scale
            classes = batch["cls"].flatten().cpu().numpy()
            sides = (boxes[:, 2:] - boxes[:, :2]).prod(-1).sqrt() * 640 / max(height, width)
            original = model.predict(batch["img"])
            original_hits = probe_detections(original, boxes.cpu().numpy(), classes, ignores)
            row = {"image": Path(batch["im_file"][0]).name, "shifts": []}
            for shift in PROBE_SHIFTS:
                shifted = model.predict(translate_images(batch["img"], shift))
                groups, stats = object_samples(batch, shift, config, small_only=False)
                _, distances, object_sides = phase_distance(
                    original[1]["feats"][0], shifted[1]["feats"][0], groups, shift, (height, width))
                delta = boxes.new_tensor([*shift, *shift])
                shifted_hits = probe_detections(
                    shifted, (boxes+delta).cpu().numpy(), classes, ignores+delta)
                margin = config.border_margin
                safe = (boxes[:, :2] >= margin).all(-1)
                safe &= (boxes[:, 2:] <= boxes.new_tensor([width, height])-margin).all(-1)
                safe &= (boxes[:, :2]+delta[:2] >= margin).all(-1)
                safe &= (boxes[:, 2:]+delta[:2] <= boxes.new_tensor([width, height])-margin).all(-1)
                group_stats = {}
                for group, low, high in (("tiny", 0, 16), ("small", 16, 32),
                                         ("medium", 32, 96), ("large", 96, float("inf"))):
                    features_mask = (object_sides >= low) & (object_sides < high)
                    detection_mask = (safe & (sides >= low) & (sides < high)).cpu().numpy()
                    result = {
                        "distance_sum": float(distances[features_mask].sum()),
                        "objects": int(features_mask.sum()),
                        "flips": int(((original_hits != shifted_hits) & detection_mask).sum()),
                        "comparisons": int(detection_mask.sum()),
                    }
                    group_stats[group] = result
                    for key, value in result.items():
                        totals[group][key] += value
                row["shifts"].append({"shift": list(shift), **stats, "groups": group_stats})
            rows.append(row)
            if len(rows) % 16 == 0:
                print(f"Phase probe: {len(rows)}/{len(indices)} images", flush=True)
    for values in totals.values():
        values["mean_cosine_distance"] = values["distance_sum"] / max(1, values["objects"])
        values["hit_flip_rate"] = values["flips"] / max(1, values["comparisons"])
    result = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "settings": {"imgsz": 960, "reference_size": 640, "images": len(indices),
                     "indices": indices.tolist(), "shifts": [list(s) for s in PROBE_SHIFTS],
                     "conf": 0.25, "iou": 0.5, "nms_iou": 0.7, "max_det": 500,
                     "border_margin": 32, "dtype": "float32", "device": str(device)},
        "groups": totals, "per_image": rows,
        "data_hashes": {**selected_hashes, **{str(path): file_hash(path) for path in [
            data_path, *sorted((data_path.parent / "manifests").glob("*.json"))]},
                       },
        "source_hashes": {p.name: file_hash(p) for p in [
            Path(__file__), Path(__file__).with_name("phase.py"),
            Path(__file__).with_name("trainer.py"), Path(__file__).with_name("dataset.py"),
            Path(__file__).with_name("geometry.py")]},
        "scope": "64 fixed validation images, four shifts, correlated object observations; "
                 "feature error and hit flips are diagnostics, not AP or evidence of causality.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"output": str(output), "images": len(indices), "groups": totals}
