"""Registered low-cost confirmations of coordinate mechanisms, without retuning."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import cv2
import numpy as np
from scipy.io import loadmat
import torch
import yaml

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, record_hash, write_csv
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.evaluation.testdev_transfer import ORIGINAL, OUTPUT as TRANSFER, checked_testdev
from sdcl.postprocess.class_margin_probe import checked_control
from sdcl.postprocess.localization import (
    action_menu, apply_actions, calibrated_candidates, frozen_control, load_calibrator,
    normalized_actions, select_actions,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source
from sdcl.training.trainer import check_version


CACHE = "output/candidate_series/testdev_actions_20261013"
ARMS = ("calibration", "absolute", "gain", "gain_always", "uniform", "regret")
CANDIDATES = {
    "relative": {"arms": ("calibration", "absolute", "gain"), "method": "gain",
                 "matched": "absolute", "primary": "official AP: gain minus absolute"},
    "retention": {"arms": ("gain", "gain_always"), "method": "gain",
                  "matched": "gain_always", "primary": "official AR500: gain minus gain_always"},
    "ranking": {"arms": ("calibration", "uniform", "regret"), "method": "regret",
                "matched": "uniform", "primary": "official AP: regret minus uniform"},
}
POLICY = {
    "version": "2026-10-13-candidate-confirmations-v1",
    "ap_vs_calibration_min_pp": .10,
    "ap_vs_matched_min_exclusive_pp": 0.,
    "ar500_vs_calibration_min_pp": -.50,
    "retention_ap_vs_always_min_pp": 0.,
    "retention_ar500_vs_always_min_exclusive_pp": 0.,
    "scope": "Low-cost investment screen, not significance, seed robustness or publication proof.",
    "rationale": (
        "User requested modest exploratory gates. +0.10 AP is a budget cutoff, not a noise "
        "estimate; -0.50 AR500 is an exploratory tolerance, not an application requirement. "
        "Mechanism/size/timing diagnostics are reported, not additional hard vetoes."
    ),
}


def confirm_decision(candidate, metrics):
    if candidate not in CANDIDATES or set(metrics) != set(CANDIDATES[candidate]["arms"]):
        raise ValueError("Unknown candidate or incomplete matched controls.")
    for values in metrics.values():
        if set(values) != set(METRICS) or not all(
            np.isfinite(v) and 0 <= v <= 100 for v in values.values()
        ):
            raise ValueError("Invalid official metric vector.")
    item = CANDIDATES[candidate]
    method, matched = metrics[item["method"]], metrics[item["matched"]]
    matched_ap = method["AP"] - matched["AP"]
    matched_ar = method["AR500"] - matched["AR500"]
    if candidate == "retention":
        values = {"AP_vs_always_pp": matched_ap, "AR500_vs_always_pp": matched_ar}
        if matched_ap >= 0 and matched_ar > 0:
            verdict = "PROMISING_POINT_ESTIMATE"
        elif matched_ap < 0 and matched_ar <= 0:
            verdict = "NO_TRANSFERRED_BENEFIT"
        else:
            verdict = "TRADEOFF_OR_NO_CLEAR_BENEFIT"
    else:
        ap = method["AP"] - metrics["calibration"]["AP"]
        ar = method["AR500"] - metrics["calibration"]["AR500"]
        values = {"AP_vs_matched_pp": matched_ap, "AR500_vs_matched_pp": matched_ar,
                  "AP_vs_calibration_pp": ap, "AR500_vs_calibration_pp": ar}
        if matched_ap <= 0:
            verdict = "NO_TRANSFERRED_MECHANISM_ADVANTAGE"
        elif ap <= 0:
            verdict = "NO_TRANSFERRED_CONTROL_BENEFIT"
        elif ar < POLICY["ar500_vs_calibration_min_pp"]:
            verdict = "AP_RECALL_TRADEOFF_EXCEEDS_BUDGET"
        elif ap < POLICY["ap_vs_calibration_min_pp"]:
            verdict = "SMALL_POSITIVE_SIGNAL"
        else:
            verdict = "PROMISING_POINT_ESTIMATE"
    return {
        "candidate": candidate, "decision": verdict, "primary": item["primary"],
        "values": values, "registered_policy": POLICY,
        "limits": "Public test-dev control results were already seen; coordinate-head comparisons "
                  "were registered before this run. One seed; no official AP CI. Positive point "
                  "estimates do not prove novelty or reliable application performance.",
        "historical_STOP": "Preserved; no historical thresholds or decisions changed.",
    }


def sources():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             project_path("configs/experiments/localization_gain_960.yaml"),
             project_path("configs/experiments/regret_probe_960.yaml")]
    return {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}


def transfer_inputs():
    directory = project_path(TRANSFER)
    plan = json.loads((directory / "plan.json").read_text())
    verify_dependencies(plan["source_hashes"], directory / "source.zip")
    manifest = json.loads((directory / "input_manifest.json").read_text())
    for name, digest in manifest.items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Prior test-dev input changed: {name}")
    verification = json.loads((directory / "verification.json").read_text())
    if verification["images"] != 1610 or not all(
        value for key, value in verification.items() if key != "images"
    ):
        raise ValueError("Prior test-dev control was not fully verified.")
    return directory, plan


def prepare(cache=CACHE, device="0", limit=None):
    from ultralytics import YOLO

    if limit is not None and not 1 <= limit <= 1610:
        raise ValueError("Smoke limit must be in 1..1610.")
    check_version()
    torch.set_num_threads(1)
    control = checked_control()
    transfer, prior = transfer_inputs()
    runtime = verify_runtime_files(ORIGINAL)
    models, mean, std, heads = frozen_heads()
    all_records, _, overlap = checked_testdev()
    records = all_records if limit is None else all_records[:limit]
    expected = {r["stem"]: r for r in records}
    baseline = {}
    with (transfer / "predictions.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["image"] not in expected:
                continue
            if item["image"] in baseline:
                raise ValueError("Duplicate prior test-dev calibration.")
            baseline[item["image"]] = np.asarray(item["boxes"]["calibration"]).reshape(-1, 6)
    if set(baseline) != set(expected):
        raise ValueError("Incomplete prior test-dev calibration.")
    cache = project_path(cache)
    cache.mkdir(parents=True, exist_ok=False)
    plan = {
        "registered_at": datetime.now().isoformat(), "policy": POLICY,
        "candidates": CANDIDATES, "arms": ARMS, "split": "test-dev", "images": len(records),
        "full_testdev": limit is None, "records_sha256": record_hash(records),
        "heads": heads, "control": control, "overlap_audit": overlap,
        "source_hashes": sources(), "prior_transfer_plan_sha256": file_hash(transfer / "plan.json"),
        "prior_transfer_capture_sha256": file_hash(transfer / "predictions.jsonl"),
        "prior_official_metrics_sha256": file_hash(transfer / "official_metrics.json"),
        "runtime_manifest_sha256": file_hash(project_path(ORIGINAL) / "runtime/manifest.json"),
        "octave_path": runtime["octave_path"],
        "inference": "Frozen detector; FP16,960,conf=.001,IoU=.7,max500,rect=False,batch1, "
                     "original plus four fixed slices. ignores=[] throughout.",
        "timing": "Shared frozen five-view forward and calibration; per-head cached added CPU1 "
                  "menu+normalization+selection. Other heads excluded; not deployment FPS.",
        "selection_history": "Methods selected on reused val; ordinary test-dev controls already "
                             "observed. No coordinate test-dev results or retuning before registration.",
    }
    save_json(cache / "plan.json", plan)
    archive_source(cache, plan["source_hashes"])
    (cache / "actions").mkdir()
    for arm in ARMS:
        (cache / "detections" / arm).mkdir(parents=True)
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
        raise ValueError("The frozen detector policy requires effective FP16.")
    chunks, timing = {}, []
    with (cache / "predictions.jsonl").open("x") as capture, (cache / "raw_views.jsonl").open("x") as raw:
        for index, record in enumerate(records, 1):
            stem = record["stem"]
            image = cv2.imread(record["image"])
            if image is None or list(image.shape[1::-1]) != record["size"]:
                raise ValueError("Invalid test-dev image.")
            windows = [[0, 0, *record["size"]], *tile_windows(*record["size"])]
            views = []
            tick = perf_counter()
            with torch.inference_mode():
                for view, window in enumerate(windows):
                    x1, y1, x2, y2 = window
                    values = predict(image[y1:y2, x1:x2].copy())
                    views.append(map_tile_predictions(values, window) if view else values)
            torch.cuda.synchronize()
            row = {"image": stem, "shared_detector_ms": (perf_counter() - tick) * 1000}
            tick = perf_counter()
            c, origins, scores, base = calibrated_candidates(
                views, windows, record["size"], [], calibrator
            )
            row["shared_calibration_ms"] = (perf_counter() - tick) * 1000
            if not np.array_equal(c, baseline[stem]):
                raise ValueError(f"Prior test-dev strong control replay differs: {stem}")
            tick = perf_counter()
            boxes, valid, features = action_menu(c, origins, scores, base, views, record["size"], [])
            normalized = normalized_actions(features, mean, std)
            row["shared_menu_normalization_ms"] = (perf_counter() - tick) * 1000
            x = torch.from_numpy(normalized)
            outputs = {}
            with torch.inference_mode():
                for arm, scorer in models.items():
                    tick = perf_counter()
                    outputs[arm] = scorer(x).numpy()
                    row[arm + "_head_ms"] = (perf_counter() - tick) * 1000
            predictions = {"calibration": c}
            for arm in ARMS[1:]:
                tick = perf_counter()
                scorer = outputs["gain"] if arm == "gain_always" else outputs[arm]
                variant = "absolute" if arm in ("uniform", "regret") else arm
                predictions[arm] = apply_actions(c, boxes, select_actions(scorer, valid, variant))
                row[arm + "_selection_ms"] = (perf_counter() - tick) * 1000
            for arm, values in predictions.items():
                if values.shape != c.shape or not np.array_equal(values[:, 4:], c[:, 4:]):
                    raise ValueError("Coordinates-only action invariant failed.")
                path = cache / "detections" / arm / f"{stem}.txt"
                write_detection(path, values, record["size"])
                if arm == "calibration" and file_hash(path) != file_hash(
                    transfer / "detections/calibration" / f"{stem}.txt"
                ):
                    raise ValueError("Official strong-control export changed.")
            path = cache / "actions" / f"{stem}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, normalized=normalized, boxes=boxes,
                                    valid=valid, calibration=c)
            chunks[stem] = {"file": path.relative_to(cache).as_posix(), "sha256": file_hash(path)}
            capture.write(json.dumps({"image": stem, "arms": {a: p.tolist() for a, p in predictions.items()}}) + "\n")
            raw.write(json.dumps({"image": stem, "views": [v.tolist() for v in views], "windows": windows}) + "\n")
            timing.append(row)
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"Shared frozen candidate cache: {index}/{len(records)}", flush=True)
    write_csv(cache / "timings.csv", timing)
    save_json(cache / "chunks.json", chunks)
    if sources() != plan["source_hashes"] or record_hash(checked_testdev()[0]) != prior["records_sha256"]:
        raise ValueError("Sources/test-dev inputs changed during preparation.")
    checked_control()
    for entry in heads.values():
        if file_hash(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("Candidate weights changed.")
    hashes = {p.relative_to(cache).as_posix(): file_hash(p)
              for p in cache.rglob("*") if p.is_file()}
    save_json(cache / "input_manifest.json", hashes)
    save_json(cache / "preparation_verification.json", {
        "images": len(records), "complete": limit is None,
        "prior_calibration_exact": True, "official_calibration_txt_exact": True,
        "no_test_labels_in_predictions": True, "same_scores_classes_order_count": True,
        "frozen_weights_unchanged": True, "no_training": True,
    })
    return {"cache": str(cache), "images": len(records), "full_testdev": limit is None}


def checked_cache(cache):
    cache = project_path(cache)
    plan = json.loads((cache / "plan.json").read_text())
    if plan["policy"] != POLICY or plan["candidates"] != json.loads(json.dumps(CANDIDATES)):
        raise ValueError("Registered candidate policy changed.")
    verify_dependencies(plan["source_hashes"], cache / "source.zip")
    hashes = json.loads((cache / "input_manifest.json").read_text())
    for name, digest in hashes.items():
        if file_hash(cache / name) != digest:
            raise ValueError(f"Candidate cache changed: {name}")
    if not plan["full_testdev"] or plan["images"] != 1610:
        raise ValueError("Smoke caches cannot yield formal candidate decisions.")
    records, _, _ = checked_testdev()
    if record_hash(records) != plan["records_sha256"]:
        raise ValueError("Candidate test-dev records changed.")
    checked_control()
    for entry in plan["heads"].values():
        if file_hash(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("Candidate checkpoint changed.")
    return cache, plan, records


def evaluate(candidate, cache=CACHE, output=None, workers=2, reuse=None):
    if candidate not in CANDIDATES or not 1 <= workers <= 3:
        raise ValueError("Unknown candidate or invalid worker count.")
    cache, plan, records = checked_cache(cache)
    runtime = verify_runtime_files(ORIGINAL)
    transfer = project_path(TRANSFER)
    if file_hash(transfer / "official_metrics.json") != plan["prior_official_metrics_sha256"]:
        raise ValueError("Pinned test-dev official control changed.")
    if runtime["octave_path"] != plan["octave_path"]:
        raise ValueError("Registered official runtime changed.")
    output = project_path(output or f"output/analysis/candidate-{candidate}-{datetime.now():%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True, exist_ok=False)
    arms = CANDIDATES[candidate]["arms"]
    save_json(output / "plan.json", {
        "registered_at": datetime.now().isoformat(), "candidate": candidate,
        "policy": plan["policy"], "primary": CANDIDATES[candidate]["primary"],
        "arms": arms, "cache": str(cache), "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "candidate_source_sha256": file_hash(__file__),
    })
    shutil.copyfile(Path(__file__), output / "evaluation_source.py")
    for name in ("toolkit", "compat"):
        shutil.copytree(transfer / name, output / name)
    shutil.copyfile(transfer / "visdrone_helpers.oct", output / "visdrone_helpers.oct")
    root = matlab_string(output)
    dataset = matlab_string(transfer / "dataset")
    detections = matlab_string(cache / "detections")
    lib = matlab_string(output / "visdrone_helpers.oct")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    names = ",".join(f"'{arm}'" for arm in arms)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}'; dataset='{dataset}'; detections='{detections}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}}; assert(any(strcmp(arm,{{{names}}})));
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==1610);
[gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(detections,arm),fullfile(dataset,'images'),1610,names);
[a,b,c,d,e,f,g]=calcAccuracy(1610,gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('CANDIDATE_COMPLETED %s\\n',arm);
""", encoding="ascii")
    report, reuse_sources = {}, {}
    original_metrics = json.loads((transfer / "official_metrics.json").read_text())
    if "calibration" in arms:
        saved_calibration = dict(zip(
            METRICS, loadmat(transfer / "calibration.mat")["values"].ravel().tolist()
        ))
        if saved_calibration != original_metrics["calibration"]:
            raise ValueError("Pinned calibration MAT differs from its official report.")
        report["calibration"] = original_metrics["calibration"]
        shutil.copyfile(transfer / "calibration.mat", output / "calibration.mat")
        reuse_sources["calibration"] = {"path": str(transfer / "calibration.mat"),
                                        "sha256": file_hash(transfer / "calibration.mat"),
                                        "condition": "All1610calibration predictions and official TXT reproduced exactly."}
    if reuse is not None:
        reuse_path = project_path(reuse)
        reuse_plan = json.loads((reuse_path / "plan.json").read_text())
        reuse_verify = json.loads((reuse_path / "verification.json").read_text())
        reused = json.loads((reuse_path / "official_metrics.json").read_text())
        if (reuse_plan["cache_plan_sha256"] != file_hash(cache / "plan.json")
                or not all(reuse_verify.values())):
            raise ValueError("Reused results were not verified on this cache.")
        for arm in arms:
            if arm not in reused or arm in report:
                continue
            path = reuse_path / f"{arm}.mat"
            values = dict(zip(METRICS, loadmat(path)["values"].ravel().tolist()))
            if values != reused[arm]:
                raise ValueError("Reused MAT metrics differ from the verified report.")
            shutil.copyfile(path, output / path.name)
            report[arm] = values
            reuse_sources[arm] = {"path": str(path), "sha256": file_hash(path)}
    input_hashes = {p.relative_to(output).as_posix(): file_hash(p)
                    for p in output.rglob("*") if p.is_file()}
    save_json(output / "input_manifest.json", input_hashes)

    def run_arm(arm):
        with (output / f"{arm}.log").open("x") as log:
            done = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), arm],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT
            )
        if done.returncode:
            raise RuntimeError(f"Official evaluator failed: {arm}.log")
        values = loadmat(output / f"{arm}.mat")["values"].ravel()
        if len(values) != len(METRICS):
            raise ValueError("Incomplete official results.")
        print(f"Official {candidate} confirmation completed: {arm}", flush=True)
        return arm, dict(zip(METRICS, values.tolist()))

    remaining = [arm for arm in arms if arm not in report]
    print(f"Candidate {candidate}: evaluating {remaining}, reusing {list(report)}.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        report.update(dict(pool.map(run_arm, remaining)))
    save_json(output / "official_metrics.json", report)
    decision = confirm_decision(candidate, report)
    save_json(output / "decision.json", decision)
    # Size metrics are evaluation-only secondary evidence, never used to retune the heads.
    names = yaml.safe_load(project_path("data/visdrone/dataset.yaml").read_text())["names"]
    names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    predictions = {arm: {} for arm in arms}
    with (cache / "predictions.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            for arm in arms:
                predictions[arm][item["image"]] = np.asarray(item["arms"][arm]).reshape(-1, 6)
    diagnostic = {}
    for arm in arms:
        print(f"Secondary reference640 COCO-style size metrics: {candidate}/{arm}", flush=True)
        diagnostic[arm] = coco_size_metrics(records, predictions[arm], names, reference_size=640)
    save_json(output / "size_diagnostic.json", {
        "scope": "COCO-style/IoF-ignore,reference640; NOT official VisDrone tiny/small AP.",
        "metrics": diagnostic,
    })
    from csv import DictReader
    with (cache / "timings.csv").open() as stream:
        times = list(DictReader(stream))[3:]
    method = CANDIDATES[candidate]["method"]
    if method == "gain":
        head = "gain"
    else:
        head = method
    added = [sum(float(r[key]) for key in (
        "shared_menu_normalization_ms", head + "_head_ms", method + "_selection_ms"
    )) for r in times]
    save_json(output / "timing_summary.json", {
        "mean_added_cpu1_ms": float(np.mean(added)), "p95_added_cpu1_ms": float(np.quantile(added, .95)),
        "scope": plan["timing"], "first_three_images_excluded": True,
    })
    checked_cache(cache)
    verify_runtime_files(ORIGINAL)
    transfer_inputs()
    for name, digest in input_hashes.items():
        if file_hash(output / name) != digest:
            raise ValueError("Registered official evaluator inputs changed.")
    for arm, entry in reuse_sources.items():
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("A reused verified result changed.")
    verification = {
        "all1610testdev_images": True, "prior_control_exact": True,
        "input_source_checkpoint_hashes_match": True, "runtime_verified": True,
        "no_test_labels_in_prediction": True, "no_training_or_retuning": True,
        "official_inputs_unchanged": True, "reused_results_verified": True,
    }
    save_json(output / "verification.json", verification)
    save_json(output / "reuse_provenance.json", reuse_sources)
    lines = ["# Registered candidate confirmation", "", f"Candidate: {candidate}",
             f"Decision: {decision['decision']}", "", "| Arm | AP (%) | AR500 (%) |",
             "|---|---:|---:|"]
    lines += [f"| {a} | {report[a]['AP']:.6f} | {report[a]['AR500']:.6f} |" for a in arms]
    lines += ["", decision["limits"], "", decision["historical_STOP"]]
    (output / "decision.md").write_text("\n".join(lines) + "\n")
    return {"output": str(output), "decision": decision, "metrics": report}
