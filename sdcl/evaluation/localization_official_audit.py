"""No-training official audit of frozen coordinate-action heads and matched controls."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.io import loadmat
import torch

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.localization_screen import owned_diagnostic
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import (
    box_iou_matrix, file_hash, load_records, match_at_threshold, record_hash, write_csv,
)
from sdcl.postprocess.class_margin_probe import checked_control
from sdcl.postprocess.localization import (
    ActionScorer, action_menu, apply_actions, calibrated_candidates, frozen_control,
    load_calibrator, normalized_actions, select_actions,
)
from sdcl.postprocess.rescue import STRONG, save_json
from sdcl.training.rescue import archive_source


ORIGINAL = "output/analysis/visdrone-official-control-20261013"
RUNS = {
    "localization": "output/runs/localization_gain_960_seed0",
    "ranking": "output/runs/regret_objective_probe_960_seed0",
}
ARMS = ("calibration", "vote", "absolute", "gain_always", "gain", "uniform", "regret")


def audit_summary(metrics):
    """Report prospective comparisons, without relabeling historical STOP screens."""
    if set(metrics) != set(ARMS):
        raise ValueError("All registered official arms are required.")
    for arm in ARMS:
        values = metrics[arm]
        if set(values) != set(METRICS) or not all(
            np.isfinite(v) and 0 <= v <= 100 for v in values.values()
        ):
            raise ValueError("Incomplete or invalid official metrics.")
    comparisons = {}
    for candidate, control in (
        ("regret", "uniform"), ("gain", "absolute"), ("gain", "gain_always"),
        ("regret", "calibration"), ("gain", "calibration"), ("vote", "calibration"),
        ("uniform", "calibration"), ("regret", "vote"), ("gain", "vote"),
    ):
        comparisons[f"{candidate}_vs_{control}"] = {
            metric + "_pp": metrics[candidate][metric] - metrics[control][metric]
            for metric in METRICS
        }
    primary = comparisons["regret_vs_uniform"]["AP_pp"]
    overall = comparisons["regret_vs_calibration"]["AP_pp"]
    return {
        "status": "NO_TRAINING_AUDIT_ONLY",
        "primary_comparison": "regret_vs_uniform official AP",
        "secondary_comparison": "gain_vs_absolute official AP",
        "ranking_positive_point_estimates": bool(primary > 0 and overall > 0),
        "comparisons": comparisons,
        "interpretation": (
            "Positive point estimates are a candidate signal only. No AP confidence interval, "
            "independent confirmation, seed robustness or novelty proof is supplied. "
            "Report AR tradeoffs explicitly; do not invent an application tolerance."
        ),
        "historical_gates": "Unchanged. This audit does not convert a prior STOP to CONTINUE.",
    }


def interference_diagnostic(record, calibration, predictions, selected, origins, boxes):
    """Annotation-assisted evidence only; never feeds the prediction pipeline."""
    gt = np.asarray([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
    classes = np.asarray([o["class_id"] for o in record["objects"]])
    matched, _ = match_at_threshold(
        box_iou_matrix(gt, calibration[:, :4]), classes, calibration, .001
    )
    owner = {int(row): int(i) for i, row in enumerate(matched) if row >= 0}
    rows = []
    for arm in ("gain", "uniform", "regret"):
        values, actions = predictions[arm], selected[arm]
        reused = {}
        result = {"image": record["stem"], "arm": arm, "owned": len(owner),
                  "changed_owned": 0, "worsened_owned": 0,
                  "other_gt_better_after_harm": 0, "peer_reused_distinct_owners": 0}
        for row, index in owner.items():
            if not actions[row]:
                continue
            result["changed_owned"] += 1
            old = box_iou_matrix(gt[index:index + 1], calibration[row:row + 1, :4])[0, 0]
            new = box_iou_matrix(gt[index:index + 1], values[row:row + 1, :4])[0, 0]
            if new < old - 1e-9:
                result["worsened_owned"] += 1
                other = np.flatnonzero((classes == classes[index]) & (np.arange(len(gt)) != index))
                if len(other) and box_iou_matrix(gt[other], values[row:row + 1, :4]).max() > new:
                    result["other_gt_better_after_harm"] += 1
            action = int(actions[row])
            if 1 <= action <= 4:
                peer = action - 1 if action <= origins[row] else action
                key = (int(peer), int(classes[index]), *boxes[row, action].tolist())
                reused.setdefault(key, set()).add(index)
        result["peer_reused_distinct_owners"] = sum(
            len(owners) for owners in reused.values() if len(owners) > 1
        )
        rows.append(result)
    return rows


def frozen_heads():
    saved, models, provenance = {}, {}, {}
    for family, run in RUNS.items():
        directory = project_path(run)
        checkpoint = directory / "last.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["epoch"] != 20 or not state["full_training_data"]:
            raise ValueError("Only the frozen complete final epoch20 heads are eligible.")
        archive = directory / "source.zip"
        if file_hash(archive) != state["source_archive_sha256"]:
            raise ValueError("Training source archive changed.")
        verify_dependencies(state["source_hashes"], archive)
        cache = project_path(state["cache_path"]) / "manifest.json"
        if file_hash(cache) != state["cache_sha256"]:
            raise ValueError("Frozen training cache manifest changed.")
        saved[family] = state
        provenance[family] = {"checkpoint": str(checkpoint), "sha256": file_hash(checkpoint),
                              "archive_sha256": file_hash(archive),
                              "cache_sha256": file_hash(cache)}
        expected = ("absolute", "gain") if family == "localization" else ("uniform", "regret")
        if set(state["models"]) != set(expected):
            raise ValueError("Unexpected frozen branches.")
        for arm in expected:
            models[arm] = ActionScorer().eval()
            models[arm].load_state_dict(state["models"][arm])
    for key in ("mean", "std"):
        if not torch.equal(saved["localization"][key], saved["ranking"][key]):
            raise ValueError("Matched action normalization differs.")
    return models, saved["localization"]["mean"].numpy(), saved["localization"]["std"].numpy(), provenance


def sources():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             project_path("configs/experiments/localization_gain_960.yaml"),
             project_path("configs/experiments/regret_probe_960.yaml")]
    return {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}


def evaluate(output=None, workers=3):
    if not 1 <= workers <= 4:
        raise ValueError("Use between one and four official evaluator workers.")
    torch.set_num_threads(1)
    control = checked_control()
    original = project_path(ORIGINAL)
    runtime = verify_runtime_files(original)
    if not json.loads((original / "runtime/validation.json").read_text())["passed"]:
        raise ValueError("Native helpers have not been validated.")
    models, mean, std, heads = frozen_heads()
    dataset = project_path("data/visdrone")
    records = load_records(dataset, "val")
    historical = frozen_control()
    raw_path = project_path(STRONG) / "raw_views.jsonl"
    if (len(records) != 548
            or record_hash(records) != historical["frozen_validation"]["records_sha256"]
            or file_hash(raw_path) != historical["frozen_validation"]["raw_views_sha256"]):
        raise ValueError("Frozen validation inputs changed.")
    capture = Path(control["prediction_capture"]["path"])
    baseline = {}
    for line in capture.read_text().splitlines():
        item = json.loads(line)
        if item["image"] in baseline:
            raise ValueError("Duplicate corrected-control capture.")
        baseline[item["image"]] = np.asarray(item["boxes"], dtype=float).reshape(-1, 6)
    expected = {r["stem"]: r for r in records}
    if set(baseline) != set(expected):
        raise ValueError("Incomplete corrected strong control.")
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(output or f"output/analysis/localization-official-audit-{now:%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at": now.isoformat(), "split": "val", "images": 548, "arms": ARMS,
        "primary": "regret versus uniform official AP; secondary gain versus absolute official AP",
        "scope": "Exploratory protocol audit after reused-val screening; no training or retuning.",
        "historical_STOP": "Preserved; no new arbitrary practical-effect cutoff.",
        "prediction_generation": "ignores=[] throughout calibration and action construction; labels only in diagnostics/evaluation.",
        "control_manifest_sha256": file_hash(original / "corrected_control_manifest.json"),
        "control_capture_sha256": file_hash(capture), "heads": heads,
        "raw_views_sha256": file_hash(raw_path), "records_sha256": record_hash(records),
        "source_hashes": sources(), "runtime_manifest_sha256": file_hash(original / "runtime/manifest.json"),
        "limitations": ["One training seed, reused val, no official AP uncertainty interval.",
                       "Toolkit category aggregation preserved; this is not a leaderboard submission.",
                       "Heads were trained with train ignored regions; no_gt inference uses fixed weights.",
                       "Conditional harm diagnostics cannot prove identity switches or small-object specificity."],
    }
    save_json(output / "plan.json", plan)
    archive_source(output, plan["source_hashes"])
    for name in ("toolkit", "compat"):
        shutil.copytree(original / name, output / name)
    shutil.copyfile(original / "runtime/visdrone_helpers.oct", output / "visdrone_helpers.oct")
    for arm in ARMS:
        (output / "detections" / arm).mkdir(parents=True)
    calibrator = load_calibrator(historical)
    owned, interference, timings = [], [], []
    seen = set()
    with raw_path.open(encoding="utf-8") as stream, (output / "predictions.jsonl").open("x") as cache:
        for line in stream:
            raw = json.loads(line)
            stem = raw["image"]
            if stem not in expected or stem in seen:
                raise ValueError("Unexpected or duplicate raw image.")
            seen.add(stem)
            record = expected[stem]
            c, origins, scores, features = calibrated_candidates(
                raw["views"], raw["windows"], record["size"], [], calibrator
            )
            if not np.array_equal(c, baseline[stem]):
                raise ValueError(f"Corrected control replay differs: {stem}")
            started = perf_counter()
            boxes, valid, features = action_menu(
                c, origins, scores, features, raw["views"], record["size"], []
            )
            x = torch.from_numpy(normalized_actions(features, mean, std))
            common_ms = (perf_counter() - started) * 1000
            outputs, times = {}, {}
            with torch.inference_mode():
                for arm, model in models.items():
                    started = perf_counter()
                    outputs[arm] = model(x).numpy()
                    times[arm] = (perf_counter() - started) * 1000
            image, selected = {"calibration": c}, {}
            for arm in ARMS[1:]:
                started = perf_counter()
                scorer = outputs.get(arm, outputs["gain"])
                variant = "absolute" if arm in ("uniform", "regret") else arm
                selected[arm] = select_actions(scorer, valid, variant)
                image[arm] = apply_actions(c, boxes, selected[arm])
                times[arm + "_selection"] = (perf_counter() - started) * 1000
            if any(not np.array_equal(v[:, 4:], c[:, 4:]) for v in image.values()):
                raise ValueError("Action selection changed scores, classes, order or count.")
            # All inference is complete before any GT-derived diagnostic is computed.
            owned.append(owned_diagnostic(record, image))
            interference.extend(interference_diagnostic(record, c, image, selected, origins, boxes))
            timings.append({"image": stem, "menu_normalization_ms": common_ms, **times})
            for arm, values in image.items():
                write_detection(output / "detections" / arm / f"{stem}.txt", values, record["size"])
            cache.write(json.dumps({"image": stem, "arms": {a: v.tolist() for a, v in image.items()}}) + "\n")
            if len(seen) == 1 or len(seen) % 50 == 0 or len(seen) == 548:
                print(f"No-GT frozen coordinate audit: {len(seen)}/548", flush=True)
    if seen != set(expected):
        raise ValueError("Incomplete prediction replay.")
    write_csv(output / "owned_gt_per_image.csv", owned)
    write_csv(output / "interference_per_image.csv", interference)
    write_csv(output / "timings.csv", timings)
    totals = {
        arm: {key: sum(row[key] for row in interference if row["arm"] == arm)
              for key in interference[0] if key not in ("image", "arm")}
        for arm in ("gain", "uniform", "regret")
    }
    save_json(output / "interference_diagnostic.json", {
        "all_sizes": totals,
        "scope": "Annotation-assisted conditional diagnostic, all owned GT sizes; not deployable efficacy.",
        "other_gt_better_after_harm": "Same-class alternative GT has higher IoU after harming original owned GT; not proof of a switch.",
        "peer_reused_distinct_owners": "Selected non-vote peer geometry shared by distinct owned GT within image/view/class.",
    })
    root, lib = matlab_string(output), matlab_string(output / "visdrone_helpers.oct")
    data = matlab_string(original / "dataset")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    names = ",".join(f"'{arm}'" for arm in ARMS)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}'; dataset='{data}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}}; assert(any(strcmp(arm,{{{names}}})));
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==548);
[gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(root,'detections',arm),fullfile(dataset,'images'),548,names);
[a,b,c,d,e,f,g]=calcAccuracy(548,gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('COORDINATE_AUDIT_COMPLETED %s\\n',arm);
""", encoding="ascii")
    inputs = {p.relative_to(output).as_posix(): file_hash(p)
              for p in output.rglob("*") if p.is_file()}
    save_json(output / "input_manifest.json", inputs)

    def run_arm(arm):
        with (output / f"{arm}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), arm],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"Official evaluator failed: {arm}.log")
        values = loadmat(output / f"{arm}.mat")["values"].ravel()
        if len(values) != len(METRICS):
            raise ValueError("Incomplete official metric vector.")
        print(f"Official coordinate audit completed: {arm}", flush=True)
        return arm, dict(zip(METRICS, values.tolist()))

    print(f"Running seven fixed official arms with {workers} workers.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metrics = dict(pool.map(run_arm, ARMS))
    pinned = control["evaluation"]["headline_metrics_percent"]
    if any(abs(metrics["calibration"][key] - pinned[key]) > 1e-12 for key in METRICS):
        raise ValueError("Official corrected calibration did not reproduce.")
    if any(file_hash(output / name) != digest for name, digest in inputs.items()):
        raise ValueError("Official audit inputs changed.")
    if sources() != plan["source_hashes"]:
        raise ValueError("Registered audit source changed.")
    verify_runtime_files(original)
    checked_control()
    if (record_hash(load_records(dataset, "val")) != plan["records_sha256"]
            or file_hash(raw_path) != plan["raw_views_sha256"]):
        raise ValueError("Validation/raw inputs changed.")
    for family, entry in heads.items():
        if file_hash(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError(f"Frozen {family} checkpoint changed.")
    summary = audit_summary(metrics)
    save_json(output / "official_metrics.json", metrics)
    save_json(output / "audit_summary.json", summary)
    save_json(output / "verification.json", {
        "all_548_images": True, "corrected_control_replay_exact": True,
        "corrected_official_metrics_reproduced": True, "all_inputs_unchanged": True,
        "same_scores_classes_order_count": True, "annotations_only_in_diagnostics_evaluation": True,
        "no_training_no_retuning": True, "source_and_checkpoints_unchanged": True,
    })
    return {"output": str(output), "metrics": metrics, "summary": summary, "interference": totals}
