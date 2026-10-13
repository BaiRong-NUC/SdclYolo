"""One frozen test-dev transfer screen, with no training or method selection."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import zipfile

import cv2
import numpy as np
from scipy.io import loadmat
import torch

from sdcl.config import project_path
from sdcl.evaluation.fusion_screen import soft_merge_predictions
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import file_hash, load_records, record_hash
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.postprocess.class_margin_probe import checked_control
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.rescue import save_json
from sdcl.training.trainer import check_version


ORIGINAL = "output/analysis/visdrone-official-control-20261013"
OUTPUT = "output/analysis/testdev-transfer-fixed-20261013"
OCTAVE = "output/tools/octave/octave-11.3.0-w64/mingw64/bin/octave-cli.exe"
RECORD_SHA = "eed9255c175075143c52a4f4a75632b9df09275882e6f6408ca02a1b40e20a5d"
ARMS = ("baseline", "soft_nms", "calibration")
GATES = {
    "official_ap_vs_soft_nms_pp": .30,
    "official_ap_vs_baseline_pp": 1.,
    "official_ar500_vs_soft_nms_pp": -.10,
}


def transfer_decision(metrics, complete):
    if set(metrics) != set(ARMS) or any(
        not np.isfinite(metrics[arm].get(key, np.nan))
        for arm in ARMS for key in METRICS
    ):
        raise ValueError("Incomplete/non-finite official metrics.")
    values = {
        "official_ap_vs_soft_nms_pp": metrics["calibration"]["AP"] - metrics["soft_nms"]["AP"],
        "official_ap_vs_baseline_pp": metrics["calibration"]["AP"] - metrics["baseline"]["AP"],
        "official_ar500_vs_soft_nms_pp": metrics["calibration"]["AR500"] - metrics["soft_nms"]["AR500"],
    }
    checks = [
        dict(metric=key, value=values[key], threshold=value, comparison=">=",
             passed=values[key] >= value)
        for key, value in GATES.items()
    ]
    return dict(
        decision=("TRANSFER_PASS" if all(r["passed"] for r in checks) else "TRANSFER_STOP")
        if complete else "SMOKE_ONLY",
        checks=checks, full_testdev=complete,
        scope="Frozen engineering-control transfer only; no novelty or permission to resume stopped recipes.",
    )


def checked_testdev():
    dataset = project_path("data/visdrone")
    records = load_records(dataset, "test")
    if len(records) != 1610 or record_hash(records) != RECORD_SHA:
        raise ValueError("The registered 1610-image test-dev split changed.")
    manifest = json.loads((dataset / "manifests/test.json").read_text())
    expected = {Path(row["image"]).stem: row for row in manifest}
    if len(expected) != 1610 or set(expected) != {r["stem"] for r in records}:
        raise ValueError("Incomplete original test-dev manifest.")
    for r in records:
        row = expected[r["stem"]]
        original = dataset / "original_annotations/test" / f"{r['stem']}.txt"
        if r["image_sha256"] != row["image_sha256"] or file_hash(original) != row["annotation_sha256"]:
            raise ValueError("Original test-dev images/annotations changed.")
    other = {s: load_records(dataset, s) for s in ("train", "val")}
    test_bytes = {r["image_sha256"] for r in records}
    test_names = {r["stem"] for r in records}
    prefix = {r["stem"].split("_")[0] for r in records}
    audit = {}
    for split, rows in other.items():
        overlap = dict(
            identical_image_bytes=len(test_bytes & {r["image_sha256"] for r in rows}),
            image_names=len(test_names & {r["stem"] for r in rows}),
            filename_prefix=len(prefix & {r["stem"].split("_")[0] for r in rows}),
        )
        if overlap["identical_image_bytes"] or overlap["image_names"] or overlap["filename_prefix"]:
            raise ValueError("Test-dev overlap with train/val; not this registered experiment.")
        audit[split] = overlap
    return records, expected, audit


def source_hashes():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             *project_path("sdcl/evaluation/native").glob("*.cc")]
    return {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}


def global_predictions(values, size):
    values = np.asarray(values, dtype=np.float64).reshape(-1, 6).copy()
    width, height = size
    values[:, [0, 2]] = values[:, [0, 2]].clip(0, width)
    values[:, [1, 3]] = values[:, [1, 3]].clip(0, height)
    values = values[np.all(values[:, 2:4] > values[:, :2], axis=1)]
    return values[np.argsort(-values[:, 4], kind="stable")][:500]


def prepare(output=OUTPUT, device="0", limit=None):
    from ultralytics import YOLO

    if limit is not None and not 1 <= limit <= 1610:
        raise ValueError("Smoke limit must be in 1..1610.")
    check_version()
    control = checked_control()
    runtime = verify_runtime_files(ORIGINAL)
    if not json.loads((project_path(ORIGINAL) / "runtime/validation.json").read_text())["passed"]:
        raise ValueError("The reference native runtime is not validated.")
    all_records, annotation_manifest, audit = checked_testdev()
    records = all_records[:limit] if limit is not None else all_records
    output = project_path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite/reselect a test-dev experiment.")
    output.mkdir(parents=True)
    original = project_path(ORIGINAL)
    shutil.copytree(original / "toolkit", output / "toolkit")
    shutil.copytree(original / "compat", output / "compat")
    shutil.copyfile(original / "runtime/visdrone_helpers.oct", output / "visdrone_helpers.oct")
    plan = dict(
        version="2026-10-13-frozen-testdev-transfer-v1",
        registered_at=datetime.now().isoformat(), split="test-dev", local_split="test",
        images=len(records), full_testdev=limit is None, records_sha256=record_hash(records),
        full_records_sha256=RECORD_SHA, annotation_manifest=annotation_manifest, overlap_audit=audit,
        gates=GATES, control=control, source_hashes=source_hashes(),
        runtime_manifest_sha256=file_hash(original / "runtime/manifest.json"),
        native_validation_sha256=file_hash(original / "runtime/validation.json"),
        octave_path=runtime["octave_path"],
        protocol="Unmodified pinned VisDrone toolkit, including its presence-weighted aggregation.",
        limits=["Test-dev labels are public; this is local evaluation, not a leaderboard submission.",
                "No matching prior test predictions were found in output/analysis before registration.",
                "First-prefix nonoverlap is a naming audit, not proof of geographical independence.",
                "Three frozen engineering controls only; no training or hyperparameter selection.",
                "Processing times are measurements of this shared-view pipeline, not deployment FPS."],
    )
    save_json(output / "plan.json", plan)
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in plan["source_hashes"]:
            archive.write(project_path(name), name)
    for arm in ARMS:
        (output / "detections" / arm).mkdir(parents=True)
    for name in ("annotations", "images"):
        (output / "dataset" / name).mkdir(parents=True)
    calibrator = load_calibrator(frozen_control())
    model = YOLO(control["detector"]["path"])
    kwargs = dict(imgsz=960, conf=.001, iou=.7, max_det=500, quantize=16, rect=False,
                  batch=1, device=device, verbose=False, save=False, augment=False, agnostic_nms=False)

    def predict(image):
        boxes = model.predict(image, **kwargs)[0].boxes
        return torch.cat((boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]), 1).float().cpu().numpy()

    with torch.inference_mode():
        for _ in range(3):
            predict(np.zeros((960, 960, 3), dtype=np.uint8))
    if not model.predictor.model.fp16:
        raise ValueError("The frozen view policy requires effective FP16.")
    started = perf_counter()
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as capture:
        for index, r in enumerate(records, 1):
            image = cv2.imread(r["image"])
            if image is None or list(image.shape[1::-1]) != r["size"]:
                raise ValueError("Invalid test-dev image/dimensions.")
            windows = [[0, 0, *r["size"]], *tile_windows(*r["size"])]
            views = []
            tick = perf_counter()
            with torch.inference_mode():
                for view, window in enumerate(windows):
                    x1, y1, x2, y2 = window
                    values = predict(image[y1:y2, x1:x2].copy())
                    views.append(map_tile_predictions(values, window) if view else values)
            torch.cuda.synchronize()
            forward_ms = (perf_counter() - tick) * 1000
            tick = perf_counter()
            soft = soft_merge_predictions(views, r["size"], [])[0]
            soft_ms = (perf_counter() - tick) * 1000
            tick = perf_counter()
            calibrated = calibrated_candidates(views, windows, r["size"], [], calibrator)[0]
            calibration_ms = (perf_counter() - tick) * 1000
            predictions = dict(baseline=global_predictions(views[0], r["size"]),
                               soft_nms=soft, calibration=calibrated)
            for arm, boxes in predictions.items():
                write_detection(output / "detections" / arm / f"{r['stem']}.txt", boxes, r["size"])
            capture.write(json.dumps(dict(
                image=r["stem"], boxes={a: v.tolist() for a, v in predictions.items()},
                timing=dict(shared_five_views_ms=forward_ms, soft_nms_ms=soft_ms,
                            calibration_ms=calibration_ms))) + "\n")
            capture.flush()
            annotation = project_path("data/visdrone/original_annotations/test") / f"{r['stem']}.txt"
            shutil.copyfile(annotation, output / "dataset/annotations" / annotation.name)
            target = output / "dataset/images" / Path(r["image"]).name
            try:
                target.hardlink_to(r["image"])
            except OSError:
                shutil.copyfile(r["image"], target)
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"Frozen test-dev predictions: {index}/{len(records)}; {perf_counter()-started:.1f}s", flush=True)
    if source_hashes() != plan["source_hashes"] or record_hash(checked_testdev()[0]) != RECORD_SHA:
        raise ValueError("Sources or test data changed during prediction.")
    checked_control()
    input_hashes = {
        p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob("*")
        if p.is_file() and p.name != "plan.json"
    }
    save_json(output / "input_manifest.json", input_hashes)
    return output


def run_official(output=OUTPUT):
    output = project_path(output)
    plan = json.loads((output / "plan.json").read_text())
    original = project_path(ORIGINAL)
    runtime = verify_runtime_files(original)
    if plan["octave_path"] != runtime["octave_path"] or source_hashes() != plan["source_hashes"]:
        raise ValueError("Runtime or registered experiment source changed.")
    inputs = json.loads((output / "input_manifest.json").read_text())
    for name, digest in inputs.items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Evaluation input changed: {name}")
    root = matlab_string(output)
    lib = matlab_string(output / "visdrone_helpers.oct")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}};
assert(any(strcmp(arm,{{'baseline','soft_nms','calibration'}})));
names=findImageList(fullfile(root,'dataset','annotations')); assert(length(names)=={plan['images']});
[gt,dt]=saveAnnoRes(fullfile(root,'dataset','annotations'),fullfile(root,'detections',arm),fullfile(root,'dataset','images'),{plan['images']},names);
[a,b,c,d,e,f,g]=calcAccuracy({plan['images']},gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('TESTDEV_OFFICIAL_COMPLETED %s\\n',arm);
""", encoding="ascii")
    script_hash = file_hash(script)

    def evaluate_arm(arm):
        if (output / f"{arm}.mat").exists() or (output / f"{arm}.log").exists():
            raise FileExistsError("Official results/logs are write-once.")
        with (output / f"{arm}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), arm],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Official {arm} failed; inspect {arm}.log.")
        print(f"Official test-dev completed: {arm}", flush=True)
        return arm, dict(zip(METRICS, loadmat(output / f"{arm}.mat")["values"].ravel().tolist()))

    print("Running three pinned official test-dev arms; logs are saved per arm.", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        metrics = dict(pool.map(evaluate_arm, ARMS))
    for name, digest in inputs.items():
        if file_hash(output / name) != digest:
            raise ValueError("Official evaluation changed input content.")
    verify_runtime_files(original)
    checked_control()
    if source_hashes() != plan["source_hashes"] or file_hash(script) != script_hash:
        raise ValueError("Official source/script changed.")
    if record_hash(checked_testdev()[0]) != RECORD_SHA:
        raise ValueError("Test-dev data changed.")
    decision = transfer_decision(metrics, plan["full_testdev"])
    save_json(output / "official_metrics.json", metrics)
    save_json(output / "decision.json", decision)
    save_json(output / "verification.json", dict(
        images=plan["images"], all_input_hashes_unchanged=True, registered_source_unchanged=True,
        detector_calibrator_unchanged=True, official_toolkit_body_unchanged=True,
        native_runtime_validated=True, no_annotations_in_prediction=True,
        no_training_or_parameter_selection=True, octave_script_sha256=script_hash,
    ))
    lines = ["# Frozen test-dev transfer", "", f"Decision: {decision['decision']}", "",
             "| Metric | Measured | Requirement | Passed |", "|---|---:|---:|---|"]
    lines += [f"| {r['metric']} | {r['value']:.6f} | >= {r['threshold']:.2f} | {r['passed']} |"
              for r in decision["checks"]]
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    return dict(output=str(output), decision=decision["decision"], metrics=metrics, checks=decision["checks"])
