"""Export frozen controls and run the pinned VisDrone MATLAB evaluation toolkit."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.fusion_screen import soft_merge_predictions
from sdcl.evaluation.size_analysis import file_hash, load_records, record_hash
from sdcl.postprocess.localization import (
    calibrated_candidates, frozen_control, load_calibrator,
)
from sdcl.postprocess.rescue import STRONG


TOOLKIT = "reference/VisDrone2018-DET-toolkit"
TOOLKIT_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
ARMS = ("frozen_calibration", "calibration_no_gt", "soft_nms_no_gt", "baseline_no_gt")
METRICS = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def detection_rows(boxes, size):
    """Use full precision, original pixel coordinates and official categories."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    if not np.isfinite(boxes).all() or len(boxes) > 500:
        raise ValueError("Non-finite detections or maxDets exceeded.")
    width, height = size
    if len(boxes) and (
        np.any(boxes[:, :2] < 0)
        or np.any(boxes[:, 2] > width)
        or np.any(boxes[:, 3] > height)
        or np.any(boxes[:, 2:4] <= boxes[:, :2])
        or np.any(boxes[:, 4] < .001)
        or np.any(boxes[:, 4] > 1)
        or np.any(boxes[:, 5] != np.floor(boxes[:, 5]))
        or np.any((boxes[:, 5] < 0) | (boxes[:, 5] > 9))
        or np.any(np.diff(boxes[:, 4]) > 0)
    ):
        raise ValueError("Invalid coordinates, classes, scores or detection order.")
    rows = np.empty((len(boxes), 8), dtype=np.float64)
    rows[:, :2] = boxes[:, :2]
    rows[:, 2:4] = boxes[:, 2:4] - boxes[:, :2]
    rows[:, 4] = boxes[:, 4]
    rows[:, 5] = boxes[:, 5] + 1
    rows[:, 6:] = -1
    return rows


def write_detection(path, boxes, size):
    rows = detection_rows(boxes, size)
    # The official loader uses load(), whose empty-text behavior is runtime-specific.
    if not len(rows):
        raise ValueError("Empty prediction file needs an explicitly verified loader.")
    text = "".join(",".join(format(v, ".17g") for v in row) + "\n" for row in rows)
    Path(path).write_text(text, encoding="ascii")
    parsed = np.loadtxt(path, delimiter=",", ndmin=2)
    if not np.array_equal(rows, parsed):
        raise AssertionError("Export changed detection precision.")


def checked_toolkit():
    toolkit = project_path(TOOLKIT)
    commit = subprocess.check_output(
        ["git", "-C", str(toolkit), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != TOOLKIT_COMMIT:
        raise ValueError("The registered official toolkit revision changed.")
    dirty = subprocess.check_output(
        ["git", "-C", str(toolkit), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("Official toolkit must remain unmodified.")
    return toolkit, {
        str(path.relative_to(toolkit)).replace("\\", "/"): file_hash(path)
        for path in sorted(toolkit.rglob("*.m"))
    }


def prepare_official(output=None):
    control = frozen_control()
    toolkit, hashes = checked_toolkit()
    dataset = project_path("data/visdrone")
    records = load_records(dataset, "val")
    raw_path = project_path(STRONG) / "raw_views.jsonl"
    validation = control["frozen_validation"]
    if (
        len(records) != 548
        or record_hash(records) != validation["records_sha256"]
        or file_hash(raw_path) != validation["raw_views_sha256"]
        or file_hash(dataset / "dataset.yaml") != validation["data_yaml_sha256"]
    ):
        raise ValueError("Frozen detector inputs or validation data changed.")
    manifest = json.loads((dataset / "manifests/val.json").read_text())
    annotation_hashes = {}
    for entry in manifest:
        stem = Path(entry["image"]).stem
        annotation = dataset / "original_annotations/val" / f"{stem}.txt"
        if (
            file_hash(annotation) != entry["annotation_sha256"]
            or file_hash(dataset / "images/val" / entry["image"]) != entry["image_sha256"]
        ):
            raise ValueError(f"Original annotation/image manifest mismatch: {stem}")
        annotation_hashes[stem] = entry["annotation_sha256"]
    if set(annotation_hashes) != {r["stem"] for r in records}:
        raise ValueError("Original annotations must cover exactly 548 validation images.")

    captures = {}
    prediction_path = Path(control["prediction_capture"]["path"])
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["image"] in captures:
            raise ValueError("Duplicate frozen prediction.")
        captures[item["image"]] = np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 6)
    if set(captures) != set(annotation_hashes):
        raise ValueError("Frozen predictions do not cover the validation split.")
    calibrator = load_calibrator(control)
    output = project_path(output or f"output/analysis/visdrone-official-{datetime.now():%Y%m%d-%H%M%S}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite official evaluation: {output}")
    output.mkdir(parents=True)
    shutil.copytree(toolkit, output / "toolkit", ignore=shutil.ignore_patterns(".git"))
    for arm in ARMS:
        (output / "detections" / arm).mkdir(parents=True)
    gt_dir, img_dir = output / "dataset/annotations", output / "dataset/images"
    gt_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for r in records:
        annotation = dataset / "original_annotations/val" / f"{r['stem']}.txt"
        shutil.copyfile(annotation, gt_dir / annotation.name)
        try:
            (img_dir / Path(r["image"]).name).hardlink_to(r["image"])
        except OSError:
            shutil.copyfile(r["image"], img_dir / Path(r["image"]).name)

    expected = {r["stem"]: r for r in records}
    counts = {arm: 0 for arm in ARMS}
    seen, different_images = set(), 0
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            stem = raw["image"]
            if stem not in expected or stem in seen:
                raise ValueError("Unexpected or duplicate raw image.")
            seen.add(stem)
            r = expected[stem]
            ignores = json.loads((dataset / "metadata/val" / f"{stem}.json").read_text())[
                "ignore_boxes_xyxy"
            ]
            replay, _, _, _ = calibrated_candidates(
                raw["views"], raw["windows"], r["size"], ignores, calibrator
            )
            if not np.array_equal(replay, captures[stem]):
                raise AssertionError("Frozen calibration no longer replays exactly.")
            clean, _, _, _ = calibrated_candidates(
                raw["views"], raw["windows"], r["size"], [], calibrator
            )
            soft, _ = soft_merge_predictions(raw["views"], r["size"], [])
            global_boxes = np.asarray(raw["views"][0], dtype=np.float64).reshape(-1, 6)
            width, height = r["size"]
            global_boxes[:, [0, 2]] = global_boxes[:, [0, 2]].clip(0, width)
            global_boxes[:, [1, 3]] = global_boxes[:, [1, 3]].clip(0, height)
            valid = np.all(global_boxes[:, 2:4] > global_boxes[:, :2], axis=1)
            global_boxes = global_boxes[valid]
            global_boxes = global_boxes[np.argsort(-global_boxes[:, 4], kind="stable")][:500]
            image = dict(zip(ARMS, (captures[stem], clean, soft, global_boxes)))
            different_images += not np.array_equal(clean, captures[stem])
            for arm, values in image.items():
                write_detection(output / "detections" / arm / f"{stem}.txt", values, r["size"])
                counts[arm] += len(values)
            if len(seen) == 1 or len(seen) % 50 == 0 or len(seen) == len(records):
                print(f"Prepared official exports: {len(seen)}/{len(records)}", flush=True)
    if seen != set(expected):
        raise ValueError("Incomplete raw prediction capture.")

    file_hashes = {
        str(p.relative_to(output)).replace("\\", "/"): file_hash(p)
        for p in sorted(output.rglob("*.txt"))
    }
    evaluator = Path(__file__).resolve()
    shutil.copyfile(evaluator, output / "evaluation_source.py")
    plan = {
        "split": "val", "images": len(records), "toolkit_commit": TOOLKIT_COMMIT,
        "toolkit_url": "https://github.com/VisDrone/VisDrone2018-DET-toolkit",
        "toolkit_hashes": hashes, "control": control, "records_sha256": record_hash(records),
        "annotation_hashes": annotation_hashes, "input_txt_hashes": file_hashes,
        "image_hashes": {r["stem"]: r["image_sha256"] for r in records},
        "evaluation_source_sha256": file_hash(evaluator),
        "counts": counts, "no_gt_calibration_changed_images": int(different_images),
        "arms": {
            "frozen_calibration": "Exact historical capture; validation ignore annotations influenced fusion and features.",
            "calibration_no_gt": "Same detector and calibration weights; fusion/features receive an empty ignore list.",
            "soft_nms_no_gt": "Same five raw views and Soft-NMS; no calibrator or validation annotations in prediction.",
            "baseline_no_gt": "Same cached global 960 view, per-view NMS, confidence .001 and max500; no annotation filtering.",
        },
        "protocol": "Pinned official MATLAB toolkit, original 8-column annotations and coordinates.",
        "limits": [
            "Validation was previously used for method selection; this is not independent test evidence.",
            "The calibration head was fitted with ignored-region filtering; no_gt is a frozen-weight inference correction.",
            "The official toolkit evalClass vector repeats each class once per image where it occurs; preserve this upstream aggregation.",
            "No official tiny/small AP is produced by calcAccuracy; do not reuse project size buckets as official metrics.",
        ],
    }
    save_json(output / "plan.json", plan)
    write_runner(output)
    save_json(output / "preparation_verification.json", {
        "frozen_calibration_replay_exact": True, "raw_prediction_hash_matches": True,
        "original_annotations_and_images_match_manifest": True,
        "full_precision_export_roundtrip_exact": True, "all_548_images_exported": True,
        "toolkit_unmodified": True, "prediction_generation_for_no_gt_arms_uses_no_annotations": True,
    })
    return output


def matlab_string(path):
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def write_runner(output):
    root = matlab_string(output)
    code = f"""root = '{root}';
addpath(fullfile(root, 'toolkit', 'utils'));
if exist('mean2', 'file') == 0
  addpath(fullfile(root, 'compat'));
end
args = argv();
arm = args{{1}};
if ~any(strcmp(arm, {{'frozen_calibration','calibration_no_gt','soft_nms_no_gt','baseline_no_gt'}}))
  error('Unknown evaluation arm');
end
result = fullfile(root, ['official_' arm '.mat']);
if exist(result, 'file'), error('Refusing to overwrite result'); end
gtPath = fullfile(root, 'dataset', 'annotations');
imgPath = fullfile(root, 'dataset', 'images');
resPath = fullfile(root, 'detections', arm);
nameImgs = findImageList(gtPath);
numImgs = length(nameImgs);
assert(numImgs == 548);
tic;
[allgt, alldet] = saveAnnoRes(gtPath, resPath, imgPath, numImgs, nameImgs);
preprocess_seconds = toc;
save('-mat7-binary', fullfile(root, ['preprocessed_' arm '.mat']), 'allgt', 'alldet', 'nameImgs');
tic;
[AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500] = calcAccuracy(numImgs, allgt, alldet);
evaluate_seconds = toc;
values = [AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500];
save('-mat7-binary', result, 'values', 'preprocess_seconds', 'evaluate_seconds');
fprintf('OFFICIAL_RESULT %s %.12f %.12f %.12f %.12f %.12f %.12f %.12f\\n', arm, values);
"""
    (output / "run_official.m").write_text(code, encoding="ascii")
    compat = output / "compat"
    compat.mkdir()
    (compat / "mean2.m").write_text(
        "function value = mean2(array)\nvalue = mean(array(:));\nend\n", encoding="ascii"
    )


def verify_prepared(output):
    output = project_path(output)
    plan = json.loads((output / "plan.json").read_text())
    for name, digest in plan["input_txt_hashes"].items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Official exported input changed: {name}")
    for name, digest in plan["toolkit_hashes"].items():
        if file_hash(output / "toolkit" / name) != digest:
            raise ValueError(f"Official evaluator changed: {name}")
    for stem, digest in plan["image_hashes"].items():
        if file_hash(output / "dataset/images" / f"{stem}.jpg") != digest:
            raise ValueError("Official image input changed.")
    if file_hash(output / "evaluation_source.py") != plan["evaluation_source_sha256"]:
        raise ValueError("Saved export source changed.")
    for arm in ARMS:
        if {p.stem for p in (output / "detections" / arm).glob("*.txt")} != set(plan["annotation_hashes"]):
            raise ValueError("Official prediction file set changed.")
    return plan


def run_official(output, octave, arm):
    output = project_path(output)
    verify_prepared(output)
    if arm not in ARMS:
        raise ValueError("Unknown official evaluation arm.")
    command = [str(Path(octave).resolve()), "--no-gui", "--quiet", str(output / "run_official.m"), arm]
    log_path = output / f"official_{arm}.log"
    if log_path.exists() or (output / f"official_{arm}.mat").exists():
        raise FileExistsError("Refusing to overwrite an official result/log.")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Official toolkit failed; inspect {log_path}")
    verify_prepared(output)
    return collect_results(output)


def collect_results(output):
    from scipy.io import loadmat

    output = project_path(output)
    verify_prepared(output)
    metrics = {}
    for arm in ARMS:
        path = output / f"official_{arm}.mat"
        if not path.exists():
            continue
        saved = loadmat(path)
        values = saved["values"].ravel()
        if len(values) != 7 or not np.isfinite(values).all():
            raise ValueError("Official metrics missing/non-finite.")
        metrics[arm] = {
            **dict(zip(METRICS, values.tolist())),
            "preprocess_seconds": float(saved["preprocess_seconds"].item()),
            "evaluate_seconds": float(saved["evaluate_seconds"].item()),
            "mat_sha256": file_hash(path),
        }
    report = {
        "protocol": "Official VisDrone toolkit " + TOOLKIT_COMMIT,
        "split": "val", "images": 548, "metrics_percent": metrics,
        "complete": set(metrics) == set(ARMS), "units": "percent; differences are percentage points",
    }
    validation_path = output / "runtime/validation.json"
    if validation_path.exists():
        report["execution"] = "Original official aggregation/preprocessing with cross-validated native hot-loop helpers."
        report["native_validation"] = json.loads(validation_path.read_text())
    save_json(output / "official_metrics.json", report)
    return report
