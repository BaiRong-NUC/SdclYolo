"""Frozen, training-free screen for local magnification of the 960 baseline."""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import yaml

from sdcl.config import project_path
from sdcl.evaluation.resolution import resolution_deltas
from sdcl.evaluation.size_analysis import (
    analyze_predictions, file_hash, load_records, record_hash, write_csv,
)


BASELINE = "output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt"
BASELINE_SHA256 = "eac88f7e348134d3bdaf8b8d3c93845002b634b66a4fa28bd6246abbbeba1a25"
RECORDS_SHA256 = "551e5afc2d589fc0b8086b391e69e137913a13781794257e3e5089ada80acf09"
DATA_YAML_SHA256 = "72373a49b320ddc76db6f864ac4a3d6abaf5b615448a1c72cf6764ddd4914f91"
EXPECTED_IMAGES = 548
GATES = {
    "tiny_ap50_95_pp": 1.0,
    "small_all_ap50_95_pp": 0.5,
    "small_all_recall_fppi10_pp": 1.0,
    "small_all_recall_fppi10_ci_low_pp": 0.0,
    "all_ap50_95_pp": -0.1,
}


@dataclass(frozen=True)
class SlicePolicy:
    imgsz: int = 960
    reference_size: int = 640
    grid: int = 2
    overlap: float = 0.20
    conf: float = 0.001
    nms_iou: float = 0.7
    max_det: int = 500
    ignore_iof: float = 0.5
    half: bool = True
    rect: bool = False
    batch: int = 1


def tile_windows(width, height, overlap=0.20):
    """Four fixed crops derived only from image shape; overlap is crop-relative."""
    if width < 2 or height < 2 or not 0 <= overlap < 1:
        raise ValueError("Tiles require dimensions >=2 and overlap in [0, 1).")
    tile_width = min(width, math.ceil(width / (2 - overlap)))
    tile_height = min(height, math.ceil(height / (2 - overlap)))
    return [(x, y, x + tile_width, y + tile_height)
            for y in (0, height - tile_height)
            for x in (0, width - tile_width)]


def map_tile_predictions(predictions, window):
    """Map view coordinates back to the original image after clipping to the crop."""
    values = np.asarray(predictions, dtype=np.float32).reshape(-1, 6).copy()
    x1, y1, x2, y2 = window
    values[:, [0, 2]] = values[:, [0, 2]].clip(0, x2 - x1) + x1
    values[:, [1, 3]] = values[:, [1, 3]].clip(0, y2 - y1) + y1
    return values


def merge_predictions(views, size, ignores, policy=SlicePolicy()):
    """Filter ignored boxes, apply class-aware global NMS, then cap at 500."""
    import torch
    from torchvision.ops import batched_nms
    from sdcl.data.geometry import prediction_ignore_mask

    values = np.concatenate(
        [np.asarray(view, dtype=np.float32).reshape(-1, 6) for view in views], axis=0)
    if not len(values):
        return np.empty((0, 6), dtype=float), 0
    values = values[np.isfinite(values).all(axis=1)]
    width, height = size
    values[:, [0, 2]] = values[:, [0, 2]].clip(0, width)
    values[:, [1, 3]] = values[:, [1, 3]].clip(0, height)
    valid = ((values[:, 2] > values[:, 0]) & (values[:, 3] > values[:, 1])
             & (values[:, 4] >= policy.conf) & (values[:, 4] <= 1)
             & (values[:, 5] >= 0) & (values[:, 5] == np.floor(values[:, 5])))
    tensor = torch.from_numpy(values[valid].copy())
    ignored = prediction_ignore_mask(
        tensor[:, :4], torch.as_tensor(np.asarray(ignores), dtype=torch.float32).reshape(-1, 4),
        policy.ignore_iof,
    )
    ignored_count = int(ignored.sum())
    tensor = tensor[~ignored]
    keep = batched_nms(tensor[:, :4], tensor[:, 4], tensor[:, 5], policy.nms_iou)
    return tensor[keep[:policy.max_det]].numpy().astype(float), ignored_count


def decide_slicing(report, full_validation=True):
    captures = report["captures"]
    first, second = (captures[name]["settings"] for name in ("baseline", "sdcl"))
    if first != second or first["policy"] != asdict(SlicePolicy()):
        raise ValueError("Screen requires identical frozen checkpoint/data/predictor policies.")
    if first["checkpoint_sha256"] != BASELINE_SHA256:
        raise ValueError("Screen requires the frozen 960 baseline checkpoint.")
    if captures["baseline"]["views_per_image"] != 1 or captures["sdcl"]["views_per_image"] != 5:
        raise ValueError("Screen requires global versus global plus four fixed tiles.")
    if report["inference_sizes"] != {"baseline": 960, "sdcl": 960}:
        raise ValueError("Screen requires 960 for every view.")
    measured = resolution_deltas(report)
    checks = [
        {"metric": key, "value_pp": measured[key], "minimum_pp": minimum,
         "comparison": ">" if key.endswith("ci_low_pp") else ">=",
         "passed": measured[key] > minimum if key.endswith("ci_low_pp")
                   else measured[key] >= minimum}
        for key, minimum in GATES.items()
    ]
    complete = (full_validation and report["images"] == EXPECTED_IMAGES
                and first["records_sha256"] == RECORDS_SHA256
                and first["dataset_yaml_sha256"] == DATA_YAML_SHA256)
    return {
        "decision": ("CONTINUE" if all(check["passed"] for check in checks) else "STOP")
                    if complete else "SMOKE_ONLY",
        "full_validation_verified": complete,
        "checks": checks,
        "rule": "All five gates must pass on all 548 frozen validation images.",
        "scope": "Local magnification headroom screen with one frozen checkpoint. "
                 "CONTINUE permits a bounded compute-budget experiment; it does not "
                 "establish a training improvement, novelty, or publication readiness.",
        "metric_protocol": report["protocol"],
        "bootstrap_scope": "2000 paired image resamples, fixed selected thresholds. "
                           "No training-seed or threshold-selection uncertainty.",
    }


def capture_slicing(checkpoint, dataset, records, output, device, settings):
    import cv2
    import torch
    from ultralytics import YOLO
    from sdcl.training.trainer import check_version

    check_version()
    policy = SlicePolicy()
    model = YOLO(str(checkpoint))
    kwargs = {
        "imgsz": policy.imgsz, "conf": policy.conf, "iou": policy.nms_iou,
        "max_det": policy.max_det, "quantize": 16 if policy.half else None, "rect": policy.rect,
        "batch": policy.batch, "device": device, "verbose": False,
        "save": False, "augment": False, "agnostic_nms": False,
    }

    def synchronize():
        if torch.cuda.is_available() and str(device).lower() != "cpu":
            torch.cuda.synchronize(torch.device(f"cuda:{device}"))

    def predict(image):
        result = model.predict(source=image, **kwargs)[0]
        boxes = result.boxes
        return torch.cat((boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]), 1).float().cpu().numpy()

    # Warm up model, square letterboxing and the CPU merge path before timing.
    dummy = np.zeros((960, 960, 3), dtype=np.uint8)
    for _ in range(3):
        predict(dummy)
    if not model.predictor.model.fp16:
        raise ValueError("Frozen screen requires effective FP16 inference on a supported device.")
    merge_predictions([np.array([[0, 0, 10, 10, 0.5, 0]], dtype=np.float32)],
                      (960, 960), [])
    synchronize()
    predictions = {"baseline": {}, "sdcl": {}}
    ignored_counts = {"baseline": 0, "sdcl": 0}
    timings = []
    for index, record in enumerate(records, 1):
        metadata = json.loads(
            (dataset / "metadata/val" / f"{record['stem']}.json").read_text(encoding="utf-8"))
        ignores = metadata["ignore_boxes_xyxy"]
        synchronize()
        started = perf_counter()
        image = cv2.imread(record["image"])
        if image is None:
            raise ValueError(f"Cannot decode image: {record['image']}")
        height, width = image.shape[:2]
        if record["size"] != [width, height]:
            raise ValueError(f"Image/metadata dimension mismatch: {record['stem']}")
        global_predictions = predict(image)
        synchronize()
        global_seconds = perf_counter() - started
        baseline, baseline_ignored = merge_predictions([global_predictions], record["size"], ignores)
        baseline_seconds = perf_counter() - started
        started = perf_counter()
        views = [global_predictions]
        for window in tile_windows(width, height, policy.overlap):
            x1, y1, x2, y2 = window
            view = predict(image[y1:y2, x1:x2].copy())
            views.append(map_tile_predictions(view, window))
        candidate, candidate_ignored = merge_predictions(views, record["size"], ignores)
        synchronize()
        extra_seconds = perf_counter() - started
        predictions["baseline"][record["stem"]] = baseline
        predictions["sdcl"][record["stem"]] = candidate
        ignored_counts["baseline"] += baseline_ignored
        ignored_counts["sdcl"] += candidate_ignored
        timings.append({
            "image": record["stem"], "baseline_ms": baseline_seconds * 1000,
            "candidate_ms": (global_seconds + extra_seconds) * 1000,
            "global_decode_predict_ms": global_seconds * 1000,
            "tiles_predict_merge_ms": extra_seconds * 1000,
            "baseline_boxes": len(baseline), "candidate_boxes": len(candidate),
        })
        if index == 1 or index % 50 == 0 or index == len(records):
            print(f"Captured paired views: {index}/{len(records)} images.", flush=True)
    write_csv(output / "timings.csv", timings)
    captures = {}
    for name, count in (("baseline", 1), ("sdcl", 5)):
        directory = output / "predictions" / name
        directory.mkdir(parents=True)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"image": record["stem"],
                                         "boxes": predictions[name][record["stem"]].tolist()}) + "\n")
        milliseconds = [row[f"{'baseline' if name == 'baseline' else 'candidate'}_ms"] for row in timings]
        captures[name] = {
            "settings": settings, "views_per_image": count,
            "ignore_filtered_predictions": ignored_counts[name],
            "predictions_sha256": file_hash(directory / "predictions.jsonl"),
            "performance": {
                "mean_ms_per_image": float(np.mean(milliseconds)),
                "median_ms_per_image": float(np.median(milliseconds)),
                "p95_ms_per_image": float(np.quantile(milliseconds, 0.95)),
                "note": "Synchronized serial batch-1 timing, includes image decode, "
                        "preprocessing, prediction, crop copies, ignore filtering and merge. "
                        "Candidate reuses global predictions: sum of shared global and "
                        "tile/merge components. Excludes setup, warmup, metadata, JSON "
                        "writes and metric evaluation; not a deployment throughput claim.",
            },
        }
        (directory / "manifest.json").write_text(json.dumps(captures[name], indent=2), encoding="utf-8")
    return predictions, captures


def run_slicing_screen(output=None, device="0", limit=None):
    import ultralytics

    if limit is not None and not 1 <= limit <= EXPECTED_IMAGES:
        raise ValueError("Smoke limit must be between 1 and 548.")
    checkpoint = project_path(BASELINE)
    dataset = project_path("data/visdrone")
    if file_hash(checkpoint) != BASELINE_SHA256:
        raise ValueError("960 baseline checkpoint changed; do not silently change the screen.")
    if file_hash(dataset / "dataset.yaml") != DATA_YAML_SHA256:
        raise ValueError("Dataset YAML changed since the frozen baseline evaluation.")
    all_records = load_records(dataset, "val")
    if len(all_records) != EXPECTED_IMAGES or record_hash(all_records) != RECORDS_SHA256:
        raise ValueError("Validation images/labels/metadata changed since the frozen baseline evaluation.")
    records = all_records[:limit] if limit is not None else all_records
    output = project_path(output or f"output/analysis/slicing-{datetime.now():%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite slicing screen: {output}")
    output.mkdir(parents=True)
    source_paths = sorted(project_path("sdcl").rglob("*.py")) + [
        project_path("scripts/evaluation/diagnose_slicing.py"), project_path("scripts/_bootstrap.py")]
    settings = {
        "checkpoint_sha256": BASELINE_SHA256, "records_sha256": record_hash(records),
        "dataset_yaml_sha256": DATA_YAML_SHA256, "policy": asdict(SlicePolicy()),
        "ultralytics": ultralytics.__version__, "device": str(device),
        "source_hashes": {str(path.relative_to(project_path("."))): file_hash(path)
                          for path in source_paths},
    }
    plan = {
        "version": "2026-10-12-slicing-v1", "fixed_gates_pp": GATES,
        "checkpoint": str(checkpoint), "settings": settings,
        "images": len(records), "full_validation_requested": limit is None,
        "comparison": "Same 960-trained checkpoint: square 960 global vs global + 2x2 "
                      "tiles, crop-relative overlap 0.20, every view square 960.",
        "fairness": "Both arms use rect=False, batch=1 and the same final max_det=500. "
                    "Recompute both arms; do not reuse rectangular validator captures.",
        "restriction": "Tile locations depend only on image shape, never annotations. "
                       "Ignored regions are used only for diagnostic scoring.",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "evaluation_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(project_path("."))))
    predictions, captures = capture_slicing(checkpoint, dataset, records, output, device, settings)
    names = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(key): value for key, value in
             (names.items() if isinstance(names, dict) else enumerate(names))}
    analyze_predictions(
        records, predictions, captures, names, output, {"baseline": 960, "sdcl": 960},
        model_labels={"baseline": "Global 960", "sdcl": "Global + 4 tiles 960"},
        inference_description=plan["comparison"] + " " + plan["fairness"],
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    decision = decide_slicing(report, full_validation=limit is None)
    performance = {name: capture["performance"] for name, capture in captures.items()}
    decision["performance"] = performance
    decision["mean_latency_ratio"] = (
        performance["sdcl"]["mean_ms_per_image"] / performance["baseline"]["mean_ms_per_image"])
    decision["report"] = str(output / "report.json")
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# Frozen local magnification screen", "", f"Decision: **{decision['decision']}**", "",
             plan["comparison"], "", plan["fairness"], "",
             "| Metric | Delta (pp) | Required | Pass |", "|---|---:|---:|---|"]
    for check in decision["checks"]:
        lines.append(f"| {check['metric']} | {check['value_pp']:.4f} | "
                     f"{check['comparison']} {check['minimum_pp']:.4f} | {check['passed']} |")
    lines.extend(["", f"Mean latency ratio: {decision['mean_latency_ratio']:.3f}x", "",
                  performance["sdcl"]["note"], "", decision["scope"], "",
                  "COCO-style / IoF-ignore diagnostics; not official VisDrone AP.",
                  "AP comparisons are paired within this square-predictor report. "
                  "Do not compare absolute AP to older rectangular-validator reports."])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), **decision}
