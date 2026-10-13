"""One frozen Gaussian Soft-NMS control, with no detector training or GT fusion."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
from time import perf_counter
import zipfile

import numpy as np
import yaml

from sdcl.config import project_path
from sdcl.evaluation.slicing_screen import (
    BASELINE, BASELINE_SHA256, DATA_YAML_SHA256, EXPECTED_IMAGES, RECORDS_SHA256,
    SlicePolicy, map_tile_predictions, merge_predictions, tile_windows,
)
from sdcl.evaluation.size_analysis import (
    analyze_predictions, file_hash, load_records, recall_at_fp_budgets,
    record_hash, write_csv,
)


HISTORICAL = "output/analysis/slicing-960-fixed-20261012"
GATES = {
    "small_all_recall_fppi10_pp": 1.0,
    "small_all_recall_fppi10_ci_low_pp": 0.0,
    "all_ap50_95_pp": -0.1,
}


@dataclass(frozen=True)
class FusionPolicy:
    method: str = "class_aware_gaussian_soft_nms"
    sigma: float = 0.5
    score_floor: float = 0.001
    max_det: int = 500
    coordinates: str = "continuous_xyxy_no_plus_one"


def gaussian_soft_nms(values, sigma=0.5, score_floor=0.001, max_det=500):
    """Repeatedly select the current max; decay same-class scores by exp(-IoU^2/sigma)."""
    if not math.isfinite(sigma) or sigma <= 0 or not 0 < score_floor <= 1 or max_det < 1:
        raise ValueError("Invalid Soft-NMS parameters.")
    pending = np.asarray(values, dtype=float).reshape(-1, 6).copy()
    if (not np.isfinite(pending).all()
            or np.any(pending[:, 2:4] <= pending[:, :2])
            or np.any(pending[:, 4] < 0) or np.any(pending[:, 4] > 1)
            or np.any(pending[:, 5] < 0)
            or np.any(pending[:, 5] != np.floor(pending[:, 5]))):
        raise ValueError("Soft-NMS requires finite valid boxes, scores and integer classes.")
    pending = pending[pending[:, 4] >= score_floor]
    selected = []
    while len(pending) and len(selected) < max_det:
        index = int(np.argmax(pending[:, 4]))
        best = pending[index].copy()
        selected.append(best)
        pending = np.delete(pending, index, axis=0)
        same_class = pending[:, 5] == best[5]
        other = pending[same_class]
        low = np.maximum(best[:2], other[:, :2])
        high = np.minimum(best[2:4], other[:, 2:4])
        intersection = np.maximum(high - low, 0).prod(axis=1)
        area = (best[2] - best[0]) * (best[3] - best[1])
        other_area = (other[:, 2] - other[:, 0]) * (other[:, 3] - other[:, 1])
        iou = intersection / np.maximum(area + other_area - intersection, 1e-12)
        pending[same_class, 4] *= np.exp(-(iou * iou) / sigma)
        pending = pending[pending[:, 4] >= score_floor]
    # Once 500 maxima are selected, later decays cannot create a higher score.
    return np.asarray(selected, dtype=float).reshape(-1, 6)


def soft_merge_predictions(views, size, ignores):
    import torch
    from sdcl.data.geometry import prediction_ignore_mask

    values = np.concatenate(
        [np.asarray(view, dtype=np.float32).reshape(-1, 6) for view in views], axis=0)
    values = values[np.isfinite(values).all(axis=1)].copy()
    width, height = size
    values[:, [0, 2]] = values[:, [0, 2]].clip(0, width)
    values[:, [1, 3]] = values[:, [1, 3]].clip(0, height)
    valid = ((values[:, 2] > values[:, 0]) & (values[:, 3] > values[:, 1])
             & (values[:, 4] >= 0.001) & (values[:, 4] <= 1)
             & (values[:, 5] >= 0) & (values[:, 5] == np.floor(values[:, 5])))
    tensor = torch.from_numpy(values[valid].copy())
    ignored = prediction_ignore_mask(
        tensor[:, :4], torch.as_tensor(np.asarray(ignores), dtype=torch.float32).reshape(-1, 4), 0.5)
    policy = FusionPolicy()
    result = gaussian_soft_nms(tensor[~ignored].numpy(), policy.sigma,
                              policy.score_floor, policy.max_det)
    return result, int(ignored.sum())


def fusion_measures(report):
    if report["reference_size"] != 640:
        raise ValueError("Fusion decision requires fixed 640 size groups.")
    budget = next(item for item in report["matched_fp_budgets"] if item["label"] == "fppi_10")
    values = {
        "small_all_recall_fppi10_pp": budget["delta_recall_pp"]["small_all"],
        "small_all_recall_fppi10_ci_low_pp":
            budget["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0],
        "all_ap50_95_pp": 100 * (report["coco_reference"]["sdcl"]["all"]["ap50_95"]
                               - report["coco_reference"]["baseline"]["all"]["ap50_95"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite decision metrics.")
    return values


def decide_fusion(report, full_validation=True):
    captures = report["captures"]
    settings = captures["baseline"]["settings"]
    if settings != captures["sdcl"]["settings"]:
        raise ValueError("Both arms must share checkpoint, raw inputs and predictor settings.")
    if (settings["checkpoint_sha256"] != BASELINE_SHA256
            or settings["policy"] != asdict(SlicePolicy())):
        raise ValueError("Frozen checkpoint/predictor policy mismatch.")
    if (captures["baseline"]["views_per_image"] != 1
            or captures["sdcl"]["views_per_image"] != 5
            or captures["baseline"]["fusion"] != {"method": "hard_nms", "iou": 0.7}
            or captures["sdcl"]["fusion"] != asdict(FusionPolicy())
            or report["inference_sizes"] != {"baseline": 960, "sdcl": 960}):
        raise ValueError("Frozen fusion comparison mismatch.")
    measured = fusion_measures(report)
    checks = [
        {"metric": key, "value_pp": measured[key], "minimum_pp": minimum,
         "comparison": ">" if key.endswith("ci_low_pp") else ">=",
         "passed": measured[key] > minimum if key.endswith("ci_low_pp")
                   else measured[key] >= minimum}
        for key, minimum in GATES.items()
    ]
    complete = (full_validation and report["images"] == EXPECTED_IMAGES
                and settings["records_sha256"] == RECORDS_SHA256
                and settings["dataset_yaml_sha256"] == DATA_YAML_SHA256)
    return {
        "decision": ("CONTINUE" if all(check["passed"] for check in checks) else "STOP")
                    if complete else "SMOKE_ONLY",
        "full_validation_verified": complete, "checks": checks,
        "rule": "All three previously declared primary gates must pass. FPPI<=5 is observational.",
        "scope": "Independent standard-fusion screen; previous tiled-inference STOP is unchanged. "
                 "CONTINUE only justifies considering a bounded follow-up. This is not "
                 "a trained method, a novelty claim, or official VisDrone evaluation.",
        "bootstrap_scope": "2000 paired image resamples, seed 0, selected thresholds fixed; "
                           "no training-seed or threshold-selection uncertainty.",
    }


def capture_fusion(checkpoint, dataset, records, output, device, settings):
    import cv2
    import torch
    from ultralytics import YOLO
    from sdcl.training.trainer import check_version

    check_version()
    model = YOLO(str(checkpoint))
    policy = SlicePolicy()
    kwargs = {
        "imgsz": policy.imgsz, "conf": policy.conf, "iou": policy.nms_iou,
        "max_det": policy.max_det, "quantize": 16, "rect": False, "batch": 1,
        "device": device, "verbose": False, "save": False,
        "augment": False, "agnostic_nms": False,
    }

    def sync():
        if torch.cuda.is_available() and str(device) != "cpu":
            torch.cuda.synchronize(torch.device(f"cuda:{device}"))

    def predict(image):
        boxes = model.predict(source=image, **kwargs)[0].boxes
        return torch.cat((boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]), 1).float().cpu().numpy()

    for _ in range(3):
        predict(np.zeros((960, 960, 3), dtype=np.uint8))
    if not model.predictor.model.fp16:
        raise ValueError("Frozen screen requires effective FP16.")
    warmup = [np.array([[0, 0, 10, 10, 0.9, 0], [1, 1, 11, 11, 0.8, 0]])]
    merge_predictions(warmup, (960, 960), [])
    soft_merge_predictions(warmup, (960, 960), [])
    sync()
    predictions = {name: {} for name in ("baseline", "hard_nms", "soft_nms")}
    ignored_counts = {name: 0 for name in predictions}
    timings = []
    raw_path = output / "raw_views.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for index, record in enumerate(records, 1):
            metadata = json.loads(
                (dataset / "metadata/val" / f"{record['stem']}.json").read_text(encoding="utf-8"))
            ignores = metadata["ignore_boxes_xyxy"]
            sync()
            started = perf_counter()
            image = cv2.imread(record["image"])
            if image is None or record["size"] != [image.shape[1], image.shape[0]]:
                raise ValueError(f"Invalid image/dimensions: {record['stem']}")
            global_boxes = predict(image)
            sync()
            global_seconds = perf_counter() - started
            started = perf_counter()
            baseline, ignored = merge_predictions([global_boxes], record["size"], ignores)
            baseline_merge_seconds = perf_counter() - started
            ignored_counts["baseline"] += ignored
            started = perf_counter()
            windows = tile_windows(*record["size"])
            views = [global_boxes]
            for window in windows:
                x1, y1, x2, y2 = window
                views.append(map_tile_predictions(
                    predict(image[y1:y2, x1:x2].copy()), window))
            sync()
            tile_seconds = perf_counter() - started
            started = perf_counter()
            control, ignored = merge_predictions(views, record["size"], ignores)
            hard_merge_seconds = perf_counter() - started
            ignored_counts["hard_nms"] += ignored
            started = perf_counter()
            candidate, ignored = soft_merge_predictions(views, record["size"], ignores)
            soft_merge_seconds = perf_counter() - started
            ignored_counts["soft_nms"] += ignored
            for name, values in (("baseline", baseline), ("hard_nms", control),
                                 ("soft_nms", candidate)):
                predictions[name][record["stem"]] = values
            timings.append({
                "image": record["stem"],
                "global_decode_predict_ms": global_seconds * 1000,
                "tiles_predict_ms": tile_seconds * 1000,
                "baseline_merge_ms": baseline_merge_seconds * 1000,
                "hard_nms_merge_ms": hard_merge_seconds * 1000,
                "soft_nms_merge_ms": soft_merge_seconds * 1000,
                "baseline_ms": (global_seconds + baseline_merge_seconds) * 1000,
                "hard_nms_ms": (global_seconds + tile_seconds + hard_merge_seconds) * 1000,
                "soft_nms_ms": (global_seconds + tile_seconds + soft_merge_seconds) * 1000,
                **{f"{name}_boxes": len(values[record["stem"]])
                   for name, values in predictions.items()},
            })
            raw.write(json.dumps({
                "image": record["stem"], "size": record["size"],
                "windows": [[0, 0, *record["size"]], *windows],
                "views": [view.tolist() for view in views],
            }) + "\n")
            if index == 1 or index % 50 == 0 or index == len(records):
                print(f"Captured all three arms from shared views: {index}/{len(records)}.", flush=True)
    write_csv(output / "timings.csv", timings)
    captures = {}
    for name in predictions:
        directory = output / "predictions" / name
        directory.mkdir(parents=True)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"image": record["stem"],
                                         "boxes": predictions[name][record["stem"]].tolist()}) + "\n")
        times = [row[f"{name}_ms"] for row in timings]
        captures[name] = {
            "settings": settings, "views_per_image": 1 if name == "baseline" else 5,
            "fusion": asdict(FusionPolicy()) if name == "soft_nms"
                      else {"method": "hard_nms", "iou": 0.7},
            "raw_views_sha256": file_hash(raw_path),
            "ignore_filtered_predictions": ignored_counts[name],
            "predictions_sha256": file_hash(directory / "predictions.jsonl"),
            "performance": {
                "mean_ms_per_image": float(np.mean(times)),
                "median_ms_per_image": float(np.median(times)),
                "p95_ms_per_image": float(np.quantile(times, 0.95)),
                "mean_merge_ms_per_image": float(np.mean([row[f"{name}_merge_ms"] for row in timings])),
                "note": "Synchronized warmed serial batch-1 component timing: shared image "
                        "decode/global prediction plus shared tile prediction where applicable "
                        "plus each arm's own ignore/merge. Includes crop copies; excludes "
                        "metadata, raw/JSON writes, setup and metrics. Not deployment throughput.",
            },
        }
        (directory / "manifest.json").write_text(json.dumps(captures[name], indent=2), encoding="utf-8")
    return predictions, captures


def run_fusion_screen(output=None, device="0", limit=None):
    import torch
    from sdcl.training.trainer import check_version

    check_version()
    if not str(device).isdigit():
        raise ValueError("Use a single CUDA device index, e.g. --device 0.")
    if limit is not None and not 1 <= limit <= EXPECTED_IMAGES:
        raise ValueError("Smoke limit must be in [1, 548].")
    checkpoint, dataset = project_path(BASELINE), project_path("data/visdrone")
    if (file_hash(checkpoint) != BASELINE_SHA256
            or file_hash(dataset / "dataset.yaml") != DATA_YAML_SHA256):
        raise ValueError("Frozen checkpoint/dataset YAML changed.")
    all_records = load_records(dataset, "val")
    if len(all_records) != EXPECTED_IMAGES or record_hash(all_records) != RECORDS_SHA256:
        raise ValueError("Frozen validation image/label/metadata hashes changed.")
    records = all_records[:limit] if limit is not None else all_records
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(output or f"output/analysis/fusion-{now:%Y%m%d-%H%M%S-%f}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fusion screen: {output}")
    output.mkdir(parents=True)
    sources = sorted(project_path("sdcl").rglob("*.py")) + [
        project_path("scripts/evaluation/diagnose_fusion.py"), project_path("scripts/_bootstrap.py")]
    settings = {
        "checkpoint_sha256": BASELINE_SHA256, "records_sha256": record_hash(records),
        "dataset_yaml_sha256": DATA_YAML_SHA256, "policy": asdict(SlicePolicy()),
        "device": str(device),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "pycocotools")},
        "source_hashes": {str(path.relative_to(project_path("."))): file_hash(path) for path in sources},
    }
    historical = project_path(HISTORICAL)
    old_report_path = historical / "report.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    historical_proof = {
        "report_path": str(old_report_path), "report_sha256": file_hash(old_report_path),
        "decision_sha256": file_hash(historical / "decision.json"),
        "predictions_sha256": {},
    }
    for name in ("baseline", "sdcl"):
        capture = old_report["captures"][name]
        if (capture["settings"]["checkpoint_sha256"] != BASELINE_SHA256
                or capture["settings"]["records_sha256"] != RECORDS_SHA256
                or capture["settings"]["policy"] != asdict(SlicePolicy())):
            raise ValueError("Historical control settings mismatch.")
        path = historical / "predictions" / name / "predictions.jsonl"
        if file_hash(path) != capture["predictions_sha256"]:
            raise ValueError("Historical predictions corrupted.")
        historical_proof["predictions_sha256"][name] = file_hash(path)
    plan = {
        "version": "2026-10-13-fusion-v1", "created_at": now.isoformat(),
        "fixed_primary_gates_pp": GATES, "fixed_fusion": asdict(FusionPolicy()),
        "secondary_only": ["tiny AP", "small_all AP", "FPPI<=5 recall", "latency"],
        "settings": settings, "images": len(records),
        "full_validation_requested": limit is None,
        "arms": ["global hard NMS 0.7", "global + 4 tiles hard NMS 0.7",
                 "global + 4 tiles Gaussian Soft-NMS sigma 0.5"],
        "scope": "One standard fusion, no parameter sweep, training or GT-assisted fusion.",
        "historical_control": historical_proof,
        "references": ["arXiv:1704.04503v2", "bharatsingh430/soft-nms lib/nms/cpu_nms.pyx"],
        "coordinate_adapter": "Continuous YOLO xyxy IoU, no legacy inclusive-pixel +1.",
        "environment": {"python": platform.python_version(), "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(int(device))},
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "evaluation_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(project_path("."))))
    predictions, captures = capture_fusion(checkpoint, dataset, records, output, device, settings)
    control_matches = {
        "baseline": captures["baseline"]["predictions_sha256"]
                    == historical_proof["predictions_sha256"]["baseline"],
        "hard_nms": captures["hard_nms"]["predictions_sha256"]
                    == historical_proof["predictions_sha256"]["sdcl"],
    }
    names = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(key): value for key, value in
             (names.items() if isinstance(names, dict) else enumerate(names))}
    common = {"inference_sizes": {"baseline": 960, "sdcl": 960}, "reference_size": 640}
    candidate_dir = output / "soft_nms_vs_global"
    candidate_dir.mkdir()
    analyze_predictions(
        records, {"baseline": predictions["baseline"], "sdcl": predictions["soft_nms"]},
        {"baseline": captures["baseline"], "sdcl": captures["soft_nms"]}, names, candidate_dir,
        model_labels={"baseline": "Global 960", "sdcl": "Tiles Gaussian Soft-NMS"},
        inference_description="Square 960 for all views; shared captures; final max_det=500. "
                              "Same per-view NMS 0.7; candidate fusion Gaussian sigma 0.5.",
        **common,
    )
    candidate_report = json.loads((candidate_dir / "report.json").read_text(encoding="utf-8"))
    control_dir = output / "hard_nms_vs_global"
    control_dir.mkdir()
    if limit is None and all(control_matches.values()):
        shutil.copyfile(old_report_path, control_dir / "report.json")
        print("Both hard-NMS prediction hashes exactly match the historical run; reusing its metrics.", flush=True)
    else:
        analyze_predictions(
            records, {"baseline": predictions["baseline"], "sdcl": predictions["hard_nms"]},
            {"baseline": captures["baseline"], "sdcl": captures["hard_nms"]},
            names, control_dir, model_labels={"baseline": "Global 960", "sdcl": "Tiles hard NMS"},
            inference_description="Shared square-960 views, hard NMS 0.7, final max_det=500.",
            **common,
        )
    control_report = json.loads((control_dir / "report.json").read_text(encoding="utf-8"))
    fusion_vs_control, _ = recall_at_fp_budgets(
        records, {"baseline": predictions["hard_nms"], "sdcl": predictions["soft_nms"]},
        640, budgets=(5, 10, 15, 20),
    )
    (output / "soft_nms_vs_hard_nms_fp_budgets.json").write_text(
        json.dumps(fusion_vs_control, indent=2), encoding="utf-8")
    decision = decide_fusion(candidate_report, full_validation=limit is None)
    decision.update({
        "performance": {name: capture["performance"] for name, capture in captures.items()},
        "historical_control_prediction_hash_matches": control_matches,
        "historical_control_metrics_reused": limit is None and all(control_matches.values()),
        "secondary": {
            "ap_delta_vs_global_pp": {
                group: 100 * (candidate_report["coco_reference"]["sdcl"][group]["ap50_95"]
                              - candidate_report["coco_reference"]["baseline"][group]["ap50_95"])
                for group in ("all", "tiny", "small_all")},
            "ap_delta_vs_hard_nms_pp": {
                group: 100 * (candidate_report["coco_reference"]["sdcl"][group]["ap50_95"]
                              - control_report["coco_reference"]["sdcl"][group]["ap50_95"])
                for group in ("all", "tiny", "small_all")},
            "fp_budgets_vs_global": candidate_report["matched_fp_budgets"],
            "fp_budgets_vs_hard_nms": fusion_vs_control,
        },
        "reports": {"soft_nms_vs_global": str(candidate_dir / "report.json"),
                    "hard_nms_vs_global": str(control_dir / "report.json")},
    })
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    verification = {
        "images": len(records),
        "source_hashes_match": all(file_hash(project_path(name)) == digest
                                 for name, digest in settings["source_hashes"].items()),
        "checkpoint_hash_matches": file_hash(checkpoint) == BASELINE_SHA256,
        "records_hash_matches": record_hash(load_records(dataset, "val")) == RECORDS_SHA256,
        "prediction_hashes_match": {
            name: file_hash(output / "predictions" / name / "predictions.jsonl")
                  == capture["predictions_sha256"] for name, capture in captures.items()},
        "historical_control_report_hash_matches": file_hash(old_report_path)
                                                 == historical_proof["report_sha256"],
        "raw_views_sha256": file_hash(output / "raw_views.jsonl"),
        "all_final_caps_valid": all(len(boxes) <= 500 for values in predictions.values()
                                   for boxes in values.values()),
        "old_stop_decision_sha256": file_hash(historical / "decision.json"),
        "old_stop_decision_unchanged": file_hash(historical / "decision.json")
                                       == historical_proof["decision_sha256"],
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    if (not verification["source_hashes_match"] or not verification["checkpoint_hash_matches"]
            or not verification["records_hash_matches"]
            or not all(verification["prediction_hashes_match"].values())
            or not verification["historical_control_report_hash_matches"]
            or not verification["old_stop_decision_unchanged"]
            or not verification["all_final_caps_valid"]):
        raise RuntimeError("Post-run verification failed; do not use this decision.")
    lines = ["# Frozen Gaussian Soft-NMS control", "", f"Decision: **{decision['decision']}**", "",
             "One checkpoint; shared square-960 global + 4 tile predictions; final max_det=500.",
             "Gaussian sigma=0.5, score floor=0.001, class-aware continuous-coordinate IoU.", "",
             "| Primary metric | Delta (pp) | Required | Pass |", "|---|---:|---:|---|"]
    for check in decision["checks"]:
        lines.append(f"| {check['metric']} | {check['value_pp']:.4f} | "
                     f"{check['comparison']} {check['minimum_pp']:.4f} | {check['passed']} |")
    lines.extend(["", "## Secondary observations", "",
                  "| Budget | Small-all recall delta vs global (pp) |", "|---|---:|"])
    for budget in candidate_report["matched_fp_budgets"]:
        lines.append(f"| {budget['label']} | {budget['delta_recall_pp']['small_all']:.4f} |")
    lines.extend(["", "| Arm | Mean ms/image | Mean merge ms/image |", "|---|---:|---:|"])
    for name, capture in captures.items():
        perf = capture["performance"]
        lines.append(f"| {name} | {perf['mean_ms_per_image']:.3f} | {perf['mean_merge_ms_per_image']:.3f} |")
    lines.extend(["", captures["soft_nms"]["performance"]["note"], "", decision["scope"], "",
                  decision["bootstrap_scope"], "", "FPPI<=5 and latency are observations, not added gates.",
                  "COCO-style / IoF-ignore diagnostics, not official VisDrone AP.",
                  "If STOP, end this fixed slicing/fusion route; do not tune sigma or restart training."])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "decision": decision["decision"], "checks": decision["checks"],
            "historical_control_prediction_hash_matches": control_matches,
            "ap_delta_vs_global_pp": decision["secondary"]["ap_delta_vs_global_pp"]}
