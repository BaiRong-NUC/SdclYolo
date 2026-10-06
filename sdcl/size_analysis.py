"""Paired size diagnostics with COCO area matching and fixed-threshold misses."""

import csv
from contextlib import redirect_stdout
from datetime import datetime
from functools import partial
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import yaml

from .config import project_path


GROUPS = {
    "all": (0.0, float("inf")),
    "tiny": (0.0, 16.0**2),
    "small": (16.0**2, 32.0**2),
    "small_all": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}
PARTITION = ("tiny", "small", "medium", "large")


def area_group(area):
    for name in PARTITION:
        lower, upper = GROUPS[name]
        if lower <= area < upper:
            return name
    raise ValueError(f"Invalid box area: {area}")


def group_mask(areas, group):
    lower, upper = GROUPS[group]
    return (areas >= lower) & (areas < upper)


def box_areas(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)


def box_iou_matrix(ground_truth, predictions):
    gt = np.asarray(ground_truth, dtype=float).reshape(-1, 4)
    dt = np.asarray(predictions, dtype=float).reshape(-1, 4)
    low = np.maximum(gt[:, None, :2], dt[None, :, :2])
    high = np.minimum(gt[:, None, 2:], dt[None, :, 2:])
    overlap = np.maximum(high - low, 0).prod(axis=2)
    union = box_areas(gt)[:, None] + box_areas(dt)[None, :] - overlap
    return overlap / np.maximum(union, 1e-12)


def match_at_threshold(ious, gt_classes, detections, confidence=0.25, threshold=0.5):
    """Match in descending score order across all size groups, one GT per DT."""
    detections = np.asarray(detections, dtype=float).reshape(-1, 6)
    matched = np.full(len(gt_classes), -1, dtype=int)
    used_predictions = np.zeros(len(detections), dtype=bool)
    for index in np.argsort(-detections[:, 4], kind="stable"):
        if detections[index, 4] < confidence:
            break
        candidates = (np.asarray(gt_classes) == int(detections[index, 5])) & (matched < 0)
        candidates &= ious[:, index] >= threshold
        if candidates.any():
            eligible = np.flatnonzero(candidates)
            chosen = eligible[np.argmax(ious[eligible, index])]
            matched[chosen] = index
            used_predictions[index] = True
    return matched, used_predictions


def miss_reasons(ious, classes, detections, matches, confidence=0.25, threshold=0.5):
    detections = np.asarray(detections, dtype=float).reshape(-1, 6)
    high = detections[:, 4] >= confidence
    causes = []
    for index, class_id in enumerate(classes):
        if matches[index] >= 0:
            causes.append("detected")
            continue
        same_class = detections[:, 5] == class_id
        overlaps = ious[index]
        if np.any(same_class & high & (overlaps >= threshold)):
            cause = "matching_competition"
        elif np.any(same_class & ~high & (overlaps >= threshold)):
            cause = "below_confidence"
        elif np.any(~same_class & high & (overlaps >= threshold)):
            cause = "class_confusion"
        elif np.any(same_class & high & (overlaps >= 0.1)):
            cause = "localization_error"
        else:
            cause = "no_nearby_detection"
        causes.append(cause)
    return causes


def load_records(dataset, split):
    records = []
    images = sorted((dataset / "images" / split).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images: {dataset / 'images' / split}")
    for index, image in enumerate(images, 1):
        metadata_path = dataset / "metadata" / split / f"{image.stem}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append({"id": index, "stem": image.stem, "image": str(image),
                        "size": metadata["original_size"], "objects": metadata["valid_objects"],
                        "metadata_sha256": file_hash(metadata_path),
                        "labels_sha256": file_hash(dataset / "labels" / split / f"{image.stem}.txt"),
                        "image_sha256": file_hash(image)})
    return records


def record_hash(records):
    content = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_predictions(checkpoint, data_yaml, records, output, device, imgsz, batch, cache=None):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy
    from .geometry import prediction_ignore_mask
    from .trainer import SDCLValidator, check_version

    check_version()
    output.mkdir(parents=True)
    settings = {"checkpoint_sha256": file_hash(checkpoint), "records_sha256": record_hash(records),
                "ultralytics": ultralytics.__version__, "imgsz": imgsz, "batch": batch,
                "device": str(device), "conf": 0.001, "nms_iou": 0.7, "max_det": 500,
                "half": True, "split": "val", "ignore_iof": 0.5,
                "dataset_yaml_sha256": file_hash(data_yaml),
                "adapter_hashes": {name: file_hash(Path(__file__).parent / name)
                                   for name in ("trainer.py", "dataset.py", "geometry.py")}}
    cached = Path(cache) if cache else None
    if cached:
        saved = json.loads((cached / "manifest.json").read_text(encoding="utf-8"))
        if saved["settings"] != settings:
            raise ValueError(f"Prediction cache does not match checkpoint/data/settings: {cached}")
        if file_hash(cached / "predictions.jsonl") != saved["predictions_sha256"]:
            raise ValueError(f"Prediction cache content hash mismatch: {cached}")
        predictions = {}
        with (cached / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                predictions[item["image"]] = np.asarray(item["boxes"], dtype=float).reshape(-1, 6)
        metrics = saved["ultralytics_metrics"]
        ignored_count = saved["ignore_filtered_predictions"]
    else:
        expected = {record["stem"]: record for record in records}
        predictions = {}
        ignored_count = 0

        class CaptureValidator(SDCLValidator):
            def update_metrics(self, preds, batch_data):
                nonlocal ignored_count
                filtered = []
                for index, pred in enumerate(preds):
                    ignored = batch_data["ignore_bboxes"][batch_data["ignore_batch_idx"] == index]
                    scale = ignored.new_tensor([batch_data["img"].shape[-1],
                                                batch_data["img"].shape[-2]] * 2)
                    drop = prediction_ignore_mask(pred["bboxes"], xywh2xyxy(ignored) * scale)
                    ignored_count += int(drop.sum())
                    pred = {key: value[~drop] for key, value in pred.items()}
                    filtered.append(pred)
                    prepared = self._prepare_batch(index, batch_data)
                    stem = Path(prepared["im_file"]).stem
                    gt_classes = prepared["cls"].cpu().numpy().astype(int)
                    reference = expected[stem]["objects"]
                    if not np.array_equal(np.sort(gt_classes),
                                          np.sort([obj["class_id"] for obj in reference])):
                        raise ValueError(f"Metadata/validation label mismatch: {stem}")
                    boxes = scale_boxes(prepared["imgsz"], pred["bboxes"].clone(),
                                        prepared["ori_shape"], prepared["ratio_pad"])
                    values = torch.cat((boxes, pred["conf"][:, None], pred["cls"][:, None]), 1)
                    predictions[stem] = values.float().cpu().numpy().astype(float)
                # Ignore filtering runs once; the same filtered tensors feed both metrics.
                DetectionValidator.update_metrics(self, filtered, batch_data)

        result = YOLO(str(checkpoint)).val(
            data=str(data_yaml), validator=partial(CaptureValidator, ignore_regions=True),
            device=device, imgsz=imgsz, batch=batch, workers=0, half=True, plots=False,
            conf=0.001, iou=0.7, max_det=500, split="val", verbose=False,
            project=str(output / "validation"), name="capture", exist_ok=False,
        )
        metrics = {key: float(value) for key, value in result.results_dict.items()}
    if set(predictions) != {record["stem"] for record in records}:
        raise ValueError("Prediction images do not match the complete validation split.")
    class_count = len(yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))["names"])
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            boxes = predictions[record["stem"]]
            if (not np.isfinite(boxes).all() or len(boxes) > 500
                    or np.any((boxes[:, 4] < 0.001) | (boxes[:, 4] > 1))
                    or np.any((boxes[:, 5] < 0) | (boxes[:, 5] >= class_count))
                    or np.any(boxes[:, 5] != np.floor(boxes[:, 5]))):
                raise ValueError(f"Invalid cached predictions: {record['stem']}")
            stream.write(json.dumps({"image": record["stem"], "boxes": boxes.tolist()}) + "\n")
    manifest = {"settings": settings, "ultralytics_metrics": metrics,
                "ignore_filtered_predictions": ignored_count,
                "predictions_sha256": file_hash(output / "predictions.jsonl")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return predictions, manifest


def coco_size_metrics(records, predictions, names, reference_size=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    counts = {group: 0 for group in GROUPS}
    annotation_id = 1
    for record in records:
        gain = reference_size / max(record["size"]) if reference_size else 1.0
        images.append({"id": record["id"], "width": record["size"][0] * gain,
                       "height": record["size"][1] * gain})
        for obj in record["objects"]:
            x1, y1, x2, y2 = np.asarray(obj["box_xyxy"], dtype=float)
            width, height = x2 - x1, y2 - y1
            # Size bins depend on widths, not subtraction of scaled absolute positions.
            area = float(width * height * gain**2)
            annotations.append({"id": annotation_id, "image_id": record["id"],
                                "category_id": obj["class_id"] + 1,
                                "bbox": [float(x1 * gain), float(y1 * gain),
                                         float(width * gain), float(height * gain)],
                                "area": area, "iscrowd": 0})
            annotation_id += 1
            for group, (lower, upper) in GROUPS.items():
                counts[group] += int(lower <= area < upper)
        for x1, y1, x2, y2, score, class_id in predictions[record["stem"]]:
            detections.append({"image_id": record["id"], "category_id": int(class_id) + 1,
                               "bbox": [float(x1 * gain), float(y1 * gain),
                                        float((x2 - x1) * gain), float((y2 - y1) * gain)],
                               "score": float(score)})
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {"images": images, "annotations": annotations, "info": {},
                                "categories": [{"id": int(index) + 1, "name": name}
                                               for index, name in names.items()]}
        ground_truth.createIndex()
        if detections:
            detected = ground_truth.loadRes(detections)
        else:
            detected = COCO()
            detected.dataset = {**ground_truth.dataset, "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(ground_truth, detected, "bbox")
        evaluator.params.maxDets = [500]
        evaluator.params.areaRngLbl = list(GROUPS)
        # COCO has inclusive upper bounds; nextafter implements our half-open bins.
        evaluator.params.areaRng = [[low, np.nextafter(high, -np.inf) if np.isfinite(high) else 1e10]
                                    for low, high in GROUPS.values()]
        evaluator.evaluate()
        evaluator.accumulate()

    def mean_valid(values):
        valid = values[values >= 0]
        return float(valid.mean()) if len(valid) else None

    result = {}
    for index, group in enumerate(GROUPS):
        precision = evaluator.eval["precision"][:, :, :, index, 0]
        recall = evaluator.eval["recall"][:, :, index, 0]
        result[group] = {"gt": counts[group], "ap50": mean_valid(precision[0]),
                         "ap50_95": mean_valid(precision), "ar50_95_max500": mean_valid(recall),
                         "per_class_ap50_95": {
                             name: mean_valid(precision[:, :, class_index])
                             for class_index, name in enumerate(names.values())}}
    return result


def paired_misses(records, predictions, names, reference_size=640, confidence=0.25):
    rows, image_counts = [], []
    thresholds = sorted({0.1, confidence, 0.5})
    sensitivity = {str(value): {group: {"gt": 0, "baseline_tp": 0, "sdcl_tp": 0}
                                for group in GROUPS} for value in thresholds}
    for record in records:
        objects = record["objects"]
        gt_boxes = np.asarray([obj["box_xyxy"] for obj in objects], dtype=float).reshape(-1, 4)
        gt_classes = np.asarray([obj["class_id"] for obj in objects], dtype=int)
        gain = reference_size / max(record["size"])
        areas = box_areas(gt_boxes) * gain**2
        image_row = {"image": record["stem"]}
        matched, causes, matrices = {}, {}, {}
        for name, prediction_map in predictions.items():
            boxes = prediction_map[record["stem"]]
            matrices[name] = box_iou_matrix(gt_boxes, boxes[:, :4])
            matched[name], used = match_at_threshold(matrices[name], gt_classes, boxes, confidence)
            causes[name] = miss_reasons(matrices[name], gt_classes, boxes, matched[name], confidence)
            image_row[f"{name}_fp"] = int(((boxes[:, 4] >= confidence) & ~used).sum())
            for threshold in thresholds:
                at_threshold, _ = match_at_threshold(matrices[name], gt_classes, boxes, threshold)
                for group in GROUPS:
                    mask = group_mask(areas, group)
                    sensitivity[str(threshold)][group][f"{name}_tp"] += int((mask & (at_threshold >= 0)).sum())
        for group in GROUPS:
            mask = group_mask(areas, group)
            image_row[f"{group}_gt"] = int(mask.sum())
            for name in predictions:
                image_row[f"{group}_{name}_tp"] = int((mask & (matched[name] >= 0)).sum())
            for threshold in thresholds:
                sensitivity[str(threshold)][group]["gt"] += int(mask.sum())
        image_counts.append(image_row)
        for index, obj in enumerate(objects):
            baseline_hit = matched["baseline"][index] >= 0
            sdcl_hit = matched["sdcl"][index] >= 0
            transition = ("both_detected" if baseline_hit and sdcl_hit else
                          "recovered" if sdcl_hit else "lost" if baseline_hit else "both_missed")
            row = {"image": record["stem"], "object_id": obj["object_id"],
                   "class_name": names[obj["class_id"]], "scale_group": area_group(areas[index]),
                   "original_area": float(box_areas(gt_boxes[index:index + 1])[0]),
                   "reference_area": float(areas[index]),
                   "reference_side": float(np.sqrt(areas[index])), "transition": transition,
                   "baseline_cause": causes["baseline"][index], "sdcl_cause": causes["sdcl"][index],
                   "occlusion": obj["occlusion"], "truncation": obj["truncation"],
                   **dict(zip(("x1", "y1", "x2", "y2"), obj["box_xyxy"]))}
            rows.append(row)
    return rows, image_counts, sensitivity


def summarize_misses(rows, image_counts, bootstrap_samples=2000):
    from collections import Counter

    groups = {}
    image_total = len(image_counts)
    generator = np.random.default_rng(0)
    samples = generator.multinomial(image_total, np.full(image_total, 1 / image_total),
                                    size=bootstrap_samples) if bootstrap_samples else None
    areas = np.asarray([row["reference_area"] for row in rows])
    for group in GROUPS:
        selected = [row for row, keep in zip(rows, group_mask(areas, group)) if keep]
        transitions = Counter(row["transition"] for row in selected)
        gt = len(selected)
        baseline_tp = transitions["both_detected"] + transitions["lost"]
        sdcl_tp = transitions["both_detected"] + transitions["recovered"]
        counts = np.array([item[f"{group}_gt"] for item in image_counts])
        differences = np.array([item[f"{group}_sdcl_tp"] - item[f"{group}_baseline_tp"]
                                for item in image_counts])
        confidence_interval = None
        if samples is not None and gt:
            denominators = samples @ counts
            valid = denominators > 0
            delta = (samples[valid] @ differences) / denominators[valid] * 100
            confidence_interval = np.quantile(delta, [0.025, 0.975]).tolist()
        groups[group] = {
            "gt": gt, "baseline_tp": baseline_tp, "sdcl_tp": sdcl_tp,
            "baseline_fn": gt - baseline_tp, "sdcl_fn": gt - sdcl_tp,
            "baseline_recall": baseline_tp / gt if gt else None,
            "sdcl_recall": sdcl_tp / gt if gt else None,
            "delta_recall_pp": (sdcl_tp - baseline_tp) / gt * 100 if gt else None,
            "recovered": transitions["recovered"], "lost": transitions["lost"],
            "both_missed": transitions["both_missed"],
            "paired_image_bootstrap_delta_recall_95ci_pp": confidence_interval,
            "miss_causes": {name: dict(Counter(row[f"{name}_cause"] for row in selected
                                              if row[f"{name}_cause"] != "detected"))
                            for name in ("baseline", "sdcl")},
        }
    return groups


def recall_at_fp_budgets(records, predictions, reference_size=640, anchor_budget=None,
                         budgets=(5, 10, 15, 20), bootstrap_samples=2000):
    """Use whole confidence tie groups and compare recall at common FP budgets."""
    models = {}
    totals = {group: 0 for group in GROUPS}
    for name, prediction_map in predictions.items():
        scores, false, per_image = [], [], []
        true = {group: [] for group in GROUPS}
        for record in records:
            gt = np.asarray([obj["box_xyxy"] for obj in record["objects"]]).reshape(-1, 4)
            classes = [obj["class_id"] for obj in record["objects"]]
            dt = prediction_map[record["stem"]]
            matched, used = match_at_threshold(box_iou_matrix(gt, dt[:, :4]), classes, dt, 0.001)
            areas = box_areas(gt) * (reference_size / max(record["size"]))**2
            scores.extend(dt[:, 4])
            false.extend(~used)
            image = {}
            for group in GROUPS:
                selected = group_mask(areas, group)
                if name == "baseline":
                    totals[group] += int(selected.sum())
                tp = np.zeros(len(dt), dtype=bool)
                matched_indices = matched[selected & (matched >= 0)]
                tp[matched_indices] = True
                true[group].extend(tp)
                image[group] = {"gt": int(selected.sum()), "scores": dt[matched_indices, 4]}
            per_image.append(image)
        scores = np.asarray(scores)
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True]) if len(scores) else []
        models[name] = {"confidence": sorted_scores[ends],
                        "fp": np.cumsum(np.asarray(false, dtype=int)[order])[ends],
                        "tp": {group: np.cumsum(np.asarray(values, dtype=int)[order])[ends]
                               for group, values in true.items()},
                        "per_image": per_image}
    count = len(records)
    requested = [(f"fppi_{budget}", int(budget * count)) for budget in budgets]
    if anchor_budget is not None:
        requested.append(("baseline_confidence_anchor", int(anchor_budget)))
    generator = np.random.default_rng(0)
    samples = generator.multinomial(count, np.full(count, 1 / count),
                                    size=bootstrap_samples) if bootstrap_samples else None
    comparisons = []
    for label, budget in requested:
        thresholds, selected = {}, {}
        for name, curve in models.items():
            eligible = np.flatnonzero(curve["fp"] <= budget)
            index = int(eligible[-1]) if len(eligible) else None
            thresholds[name] = (float(curve["confidence"][index]) if index is not None else
                                float(np.nextafter(curve["confidence"][0], np.inf))
                                if len(curve["confidence"]) else 1.0)
            selected[name] = {"confidence": thresholds[name],
                              "actual_fp": int(curve["fp"][index]) if index is not None else 0,
                              "actual_fppi": float(curve["fp"][index] / count) if index is not None else 0,
                              "recall": {group: float(curve["tp"][group][index] / total)
                                         if total and index is not None else 0.0 if total else None
                                         for group, total in totals.items()}}
        delta, intervals = {}, {}
        for group, total in totals.items():
            delta[group] = ((selected["sdcl"]["recall"][group] - selected["baseline"]["recall"][group])
                            * 100 if total else None)
            intervals[group] = None
            if total and samples is not None:
                gt = np.array([image[group]["gt"] for image in models["baseline"]["per_image"]])
                difference = np.array([
                    int((b[group]["scores"] >= thresholds["sdcl"]).sum())
                    - int((a[group]["scores"] >= thresholds["baseline"]).sum())
                    for a, b in zip(models["baseline"]["per_image"], models["sdcl"]["per_image"])])
                denominator = samples @ gt
                valid = denominator > 0
                bootstrap = (samples[valid] @ difference) / denominator[valid] * 100
                intervals[group] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        comparisons.append({"label": label, "maximum_fp": budget, "maximum_fppi": budget / count,
                            "models": selected, "delta_recall_pp": delta,
                            "paired_image_bootstrap_delta_recall_95ci_pp": intervals})
    curves = {name: [{"confidence": float(score), "fp": int(curve["fp"][index]),
                     "fppi": float(curve["fp"][index] / count),
                     **{f"{group}_recall": float(curve["tp"][group][index] / total) if total else None
                        for group, total in totals.items()}}
                    for index, score in enumerate(curve["confidence"])]
              for name, curve in models.items()}
    return comparisons, curves


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def make_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(PARTITION)
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for axis, metric, title in ((axes[0], "ap50_95", "COCO-style AP50-95"),
                                (axes[1], "recall", f"Recall @ conf {report['confidence']}, IoU 0.50")):
        for name, offset, color in (("baseline", -0.18, "#25677d"), ("sdcl", 0.18, "#d47728")):
            values = [(report["coco_reference"][name][group][metric] if metric == "ap50_95"
                       else report["misses"][group][f"{name}_recall"]) for group in groups]
            axis.bar(x + offset, [100 * value if value is not None else np.nan for value in values],
                     width=0.36, color=color, label=name)
        axis.set(title=title, ylabel="Percent", xticks=x, xticklabels=groups)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
    recovered = [report["misses"][group]["recovered"] for group in groups]
    lost = [report["misses"][group]["lost"] for group in groups]
    axes[2].bar(x - 0.18, recovered, width=0.36, color="#2a8766", label="Recovered by SDCL")
    axes[2].bar(x + 0.18, lost, width=0.36, color="#b55a48", label="Lost by SDCL")
    axes[2].set(title="Paired GT transitions", ylabel="Objects", xticks=x, xticklabels=groups)
    axes[2].legend(frameon=False)
    fig.suptitle(f"Checkpoint comparison | size after resize to {report['reference_size']} | diagnostics")
    fig.savefig(output / "size_comparison.png", dpi=170)
    plt.close(fig)


def plot_fp_curves(curves, report, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for axis, group in zip(axes, ("tiny", "small_all")):
        for name, color in (("baseline", "#25677d"), ("sdcl", "#d47728")):
            selected = [row for row in curves[name] if row["fppi"] <= 30
                        and row[f"{group}_recall"] is not None]
            axis.plot([row["fppi"] for row in selected],
                      [100 * row[f"{group}_recall"] for row in selected], label=name, color=color)
            recall = report["misses"][group][f"{name}_recall"]
            if recall is not None:
                axis.scatter(report["false_positives"][name] / report["images"],
                             100 * recall, color=color, s=45)
        axis.set(title=group, xlabel="False positives per image", ylabel="Recall (%)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(f"Micro recall at IoU 0.50 | dots: confidence {report['confidence']} | diagnostic")
    fig.savefig(output / "recall_vs_fppi.png", dpi=170)
    plt.close(fig)


def render_examples(records, rows, predictions, names, output, reference_size=640, confidence=0.25):
    from PIL import Image, ImageDraw

    examples = []
    record_map = {record["stem"]: record for record in records}
    for transition in ("recovered", "lost", "both_missed"):
        selected = [row for row in rows if row["transition"] == transition
                    and row["scale_group"] in {"tiny", "small"}]
        selected.sort(key=lambda row: (abs(row["reference_side"] - 20), row["image"], row["object_id"]))
        seen_images = set()
        for row in selected:
            if row["image"] in seen_images:
                continue
            seen_images.add(row["image"])
            examples.append(row)
            if len(seen_images) == 3:
                break
    example_dir = output / "examples"
    example_dir.mkdir()
    width, height = 720, 410
    contact = Image.new("RGB", (width * 2, height * len(examples)), "white")
    for row_index, row in enumerate(examples):
        record = record_map[row["image"]]
        factor = reference_size / max(record["size"])
        margin = 75 / factor
        left = max(0, int(row["x1"] - margin))
        top = max(0, int(row["y1"] - margin))
        right = min(record["size"][0], int(row["x2"] + margin))
        bottom = min(record["size"][1], int(row["y2"] + margin))
        with Image.open(record["image"]) as source:
            crop = source.crop((left, top, right, bottom)).convert("RGB")
        gain = min(width / crop.width, (height - 58) / crop.height)
        size = (max(1, int(crop.width * gain)), max(1, int(crop.height * gain)))
        for column, name in enumerate(("baseline", "sdcl")):
            panel = Image.new("RGB", (width, height), "#f1f1ee")
            panel.paste(crop.resize(size), (0, 58))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 7), f"{name} | {row['transition']} | {row['class_name']} | "
                      f"{row['reference_side']:.1f}px | {row[name + '_cause']}", fill="black")
            draw.text((8, 25), f"{row['image']} / GT {row['object_id']} | crop magnified | "
                      f"green: selected GT; orange: predictions >={confidence}", fill="black")
            for x1, y1, x2, y2, score, class_id in predictions[name][row["image"]]:
                if score < confidence or x2 < left or x1 > right or y2 < top or y1 > bottom:
                    continue
                coordinates = [(x1 - left) * gain, (y1 - top) * gain + 58,
                               (x2 - left) * gain, (y2 - top) * gain + 58]
                draw.rectangle(coordinates, outline="#d47728", width=2)
                draw.text((coordinates[0], coordinates[1]),
                          f"{names[int(class_id)]} {score:.2f}", fill="#a44305")
            target = [(row["x1"] - left) * gain, (row["y1"] - top) * gain + 58,
                      (row["x2"] - left) * gain, (row["y2"] - top) * gain + 58]
            draw.rectangle(target, outline="#00845a", width=4)
            contact.paste(panel, (column * width, row_index * height))
        contact.crop((0, row_index * height, width * 2, (row_index + 1) * height)).save(
            example_dir / f"{row_index + 1:02d}_{row['transition']}.png")
    if examples:
        contact.save(example_dir / "paired_small_objects.png")
    write_csv(example_dir / "selected_examples.csv", examples)


def run_size_analysis(baseline, sdcl, dataset="data/visdrone", output=None, device="0",
                      imgsz=640, batch=16, confidence=0.25, prediction_cache=None):
    dataset = project_path(dataset)
    output = project_path(output or f"output/analysis/size-{datetime.now():%Y%m%d-%H%M%S}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    if not 0.001 <= confidence <= 1 or imgsz < 32 or batch < 1:
        raise ValueError("Invalid confidence/imgsz/batch.")
    output.mkdir(parents=True)
    records = load_records(dataset, "val")
    data = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))
    source_names = data["names"]
    names = {int(key): value for key, value in
             (source_names.items() if isinstance(source_names, dict) else enumerate(source_names))}
    names = dict(sorted(names.items()))
    predictions, captures = {}, {}
    for name, checkpoint in (("baseline", baseline), ("sdcl", sdcl)):
        print(f"Capturing {name} predictions on {len(records)} validation images.", flush=True)
        cache = project_path(prediction_cache) / "predictions" / name if prediction_cache else None
        predictions[name], captures[name] = capture_predictions(
            project_path(checkpoint), dataset / "dataset.yaml", records,
            output / "predictions" / name, device, imgsz, batch, cache)
    report = {
        "protocol": "COCO-style size diagnostics with existing IoF ignore filtering; NOT official VisDrone",
        "images": len(records), "reference_size": imgsz, "confidence": confidence,
        "matching_iou": 0.5, "captures": captures, "coco_reference": {}, "coco_original": {},
        "analysis_source_sha256": file_hash(__file__),
        "size_groups_area_half_open": {key: [lo, hi if np.isfinite(hi) else None]
                                       for key, (lo, hi) in GROUPS.items()},
        "definitions": {
            "reference_area": "original_bbox_area * (imgsz / max(original_width, original_height))**2",
            "ap": "pycocotools 101-point class-macro AP; IoU 0.50:0.05:0.95; confidence >=0.001",
            "area_matching": "Keep all GT; out-of-range GT/matches are ignored by COCO area evaluation.",
            "recall": "Class-aware score-greedy 1:1 matching against all GT; micro TP/GT at fixed confidence.",
            "bootstrap": "2000 paired resamples of validation images, seed 0; recall delta only.",
            "causes": "Heuristic mutually exclusive causes from retained predictions; not causal attribution.",
            "ap_comparison": "Compare these AP values within this report; they differ from Ultralytics/VisDrone AP.",
            "inference": "Same validation/NMS path as previous analysis; at most 500 predictions per image.",
        },
    }
    for view, reference in (("coco_reference", imgsz), ("coco_original", None)):
        for name in predictions:
            print(f"Evaluating {view}/{name} area AP.", flush=True)
            report[view][name] = coco_size_metrics(records, predictions[name], names, reference)
    print("Matching individual GT and bootstrapping image-level recall differences.", flush=True)
    rows, image_counts, sensitivity = paired_misses(records, predictions, names, imgsz, confidence)
    report["misses"] = summarize_misses(rows, image_counts)
    for group in GROUPS:
        if report["misses"][group]["gt"] != report["coco_reference"]["baseline"][group]["gt"]:
            raise AssertionError(f"AP/Recall size-group GT counts disagree: {group}")
    report["confidence_sensitivity"] = sensitivity
    report["false_positives"] = {
        name: sum(item[f"{name}_fp"] for item in image_counts) for name in predictions}
    print("Comparing recalls at shared false-positive budgets.", flush=True)
    report["matched_fp_budgets"], curves = recall_at_fp_budgets(
        records, predictions, imgsz, report["false_positives"]["baseline"])
    report["definitions"]["matched_fp"] = (
        "Same maximum global FP count; threshold includes whole confidence ties. "
        "Bootstrap holds selected thresholds fixed; it does not include threshold-selection uncertainty.")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "objects.csv", rows)
    write_csv(output / "per_image.csv", image_counts)
    comparison = []
    for group in GROUPS:
        item = {"group": group, **{key: value for key, value in report["misses"][group].items()
                                   if key not in {"miss_causes", "paired_image_bootstrap_delta_recall_95ci_pp"}}}
        for metric in ("ap50", "ap50_95", "ar50_95_max500"):
            a = report["coco_reference"]["baseline"][group][metric]
            b = report["coco_reference"]["sdcl"][group][metric]
            item.update({f"baseline_{metric}": a, f"sdcl_{metric}": b,
                         f"delta_{metric}_pp": (b - a) * 100 if a is not None and b is not None else None})
        comparison.append(item)
    write_csv(output / "size_comparison.csv", comparison)
    make_figures(report, output)
    for name, curve in curves.items():
        write_csv(output / f"{name}_recall_fp_curve.csv", curve)
    plot_fp_curves(curves, report, output)
    render_examples(records, rows, predictions, names, output, imgsz, confidence)
    return {"output": str(output), "size_metrics": comparison}
