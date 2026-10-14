"""No-training official audit of a passed stable-order tiny-head pilot."""

from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.io import loadmat
import torch

from sdcl.config import project_path
from sdcl.evaluation.localization_official_audit import ORIGINAL, frozen_heads
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import file_hash, load_records, record_hash, write_csv
from sdcl.postprocess.class_margin_probe import checked_control
from sdcl.postprocess.localization import (
    ActionScorer, action_menu, apply_actions, calibrated_candidates, frozen_control,
    load_calibrator, normalized_actions, select_actions,
)
from sdcl.postprocess.rescue import STRONG, save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.stable_order_pilot import ARMS as HEAD_ARMS, OUTPUT as PRIOR


REGISTRATION = "docs/experiments/candidate20_stable_order_official_registration_20261014.json"
OUTPUT = "output/analysis/stable-order-official-val-fixed-20261014"
HISTORY = "output/analysis/localization-official-audit-fixed-20261013"
ARMS = ("calibration", "uniform", "regret",
        *(f"{arm}_seed{seed}" for seed in range(3) for arm in HEAD_ARMS))


def decide(metrics, policy):
    rows = []
    for seed in range(3):
        stable = metrics[f"stable_order_seed{seed}"]
        nominal = metrics[f"nominal_order_seed{seed}"]
        thinned = metrics[f"random_thinned_order_seed{seed}"]
        rows.append({
            "seed": seed,
            "AP_vs_uniform_pp": stable["AP"] - metrics["uniform"]["AP"],
            "AP_vs_regret_pp": stable["AP"] - metrics["regret"]["AP"],
            "AP_vs_nominal_pp": stable["AP"] - nominal["AP"],
            "AP_vs_thinned_pp": stable["AP"] - thinned["AP"],
            "AR500_vs_uniform_pp": stable["AR500"] - metrics["uniform"]["AR500"],
        })
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
    positives = sum(r["AP_vs_nominal_pp"] > 0 and r["AP_vs_thinned_pp"] > 0 for r in rows)
    checks = {
        "baseline_AP_gain": mean["AP_vs_uniform_pp"] >= policy["mean_stable_AP_vs_uniform_min_pp"],
        "regret_AP_gain": mean["AP_vs_regret_pp"] >
        policy["mean_stable_AP_vs_regret_min_exclusive_pp"],
        "ordinary_objective_AP_gain": mean["AP_vs_nominal_pp"] >
        policy["mean_stable_AP_vs_nominal_min_exclusive_pp"],
        "matched_count_AP_gain": mean["AP_vs_thinned_pp"] >
        policy["mean_stable_AP_vs_thinned_min_exclusive_pp"],
        "positive_matched_seeds": positives >= policy["positive_vs_both_matched_seeds_min"],
        "AR500_budget": mean["AR500_vs_uniform_pp"] >=
        -policy["mean_stable_AR500_vs_uniform_max_decrease_pp"],
    }
    return {
        "decision": "WIDER_TRAIN_ONLY_PILOT_ALLOWED" if all(checks.values()) else
        "STOP_THIS_STABLE_ORDER_OFFICIAL_RECIPE",
        "checks": checks, "mean": mean, "positive_matched_seeds": positives, "seeds": rows,
        "scope": "Official toolkit on reused validation; no independent confirmation or novelty claim.",
    }


def evaluate(output=OUTPUT, workers=3):
    started = perf_counter()
    if not 1 <= workers <= 3:
        raise ValueError("Use between one and three evaluator workers.")
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior, original, history = project_path(PRIOR), project_path(ORIGINAL), project_path(HISTORY)
    gate, prior_plan = verify_record(
        prior, "docs/experiments/candidate19_stable_order_pilot_result_20261014.json")
    if gate["decision"]["decision"] != "BOUNDED_OFFICIAL_AUDIT_ALLOWED":
        raise ValueError("Matched pilot did not pass.")
    runtime = verify_runtime_files(original)
    if not json.loads((original / "runtime/validation.json").read_text())["passed"]:
        raise ValueError("Original official-runtime numerical validation failed.")
    control, historical = checked_control(), frozen_control()
    frozen, mean, std, provenance = frozen_heads()
    if provenance != prior_plan["heads"]:
        raise ValueError("Pilot normalization or frozen weights changed.")
    models = {a: frozen[a] for a in ("uniform", "regret")}
    for seed in range(3):
        saved = torch.load(prior / f"seed{seed}/last.pt", map_location="cpu", weights_only=False)
        if saved["epoch"] != 8 or saved["plan_sha256"] != file_hash(prior / "plan.json"):
            raise ValueError("Pilot final checkpoint provenance differs.")
        for arm in HEAD_ARMS:
            model = ActionScorer().eval()
            model.load_state_dict(saved["models"][arm])
            models[f"{arm}_seed{seed}"] = model
    torch.set_num_threads(1)
    records = load_records(project_path("data/visdrone"), "val")
    raw_path = project_path(STRONG) / "raw_views.jsonl"
    if len(records) != 548 or record_hash(records) != historical["frozen_validation"]["records_sha256"]:
        raise ValueError("Frozen val image records differ.")
    if file_hash(raw_path) != historical["frozen_validation"]["raw_views_sha256"]:
        raise ValueError("Frozen view capture differs.")
    prior_official_plan = json.loads((history / "plan.json").read_text())
    prior_official_metrics = json.loads((history / "official_metrics.json").read_text())
    if (prior_official_plan["raw_views_sha256"] != file_hash(raw_path) or
            prior_official_plan["records_sha256"] != record_hash(records)):
        raise ValueError("Historical official baseline inputs differ.")
    expected = {r["stem"]: r for r in records}
    capture = {}
    with open(control["prediction_capture"]["path"], encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            capture[row["image"]] = np.asarray(row["boxes"], dtype=float).reshape(-1, 6)
    if set(capture) != set(expected):
        raise ValueError("Corrected calibrated capture is incomplete.")
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/evaluation/stable_order_official.py",
             "scripts/evaluation/evaluate_stable_order_official.py",
             "sdcl/evaluation/official_visdrone.py", "sdcl/evaluation/official_runtime.py",
             "sdcl/postprocess/localization.py", "sdcl/postprocess/rescue.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    checkpoints = {f"seed{s}/last.pt": file_hash(prior / f"seed{s}/last.pt") for s in range(3)}
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate19_stable_order_pilot_result_20261014.json")),
        "prior_plan_sha256": file_hash(prior / "plan.json"),
        "pilot_checkpoint_hashes": checkpoints, "frozen_heads": provenance,
        "raw_views_sha256": file_hash(raw_path), "records_sha256": record_hash(records),
        "official_runtime_sha256": file_hash(original / "runtime/manifest.json"),
        "original_metrics_sha256": file_hash(history / "official_metrics.json"),
    })
    for name in ("toolkit", "compat"):
        shutil.copytree(original / name, output / name)
    shutil.copyfile(original / "runtime/visdrone_helpers.oct", output / "visdrone_helpers.oct")
    for arm in ARMS:
        (output / "detections" / arm).mkdir(parents=True)
    calibrator = load_calibrator(historical)
    times, seen = [], set()
    with torch.inference_mode(), raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            stem = item["image"]
            if stem not in expected or stem in seen:
                raise ValueError("Unexpected/duplicate view capture row.")
            seen.add(stem)
            record = expected[stem]
            views = [np.asarray(v, float).reshape(-1, 6) for v in item["views"]]
            c, origins, scores, base = calibrated_candidates(
                views, item["windows"], record["size"], [], calibrator)
            if not np.array_equal(c, capture[stem]):
                raise ValueError("Calibration raw replay differs; stop without changing control.")
            begin = perf_counter()
            boxes, valid, features = action_menu(c, origins, scores, base, views, record["size"], [])
            x = torch.from_numpy(normalized_actions(features, mean, std))
            row = {"image": stem, "menu_normalization_ms": (perf_counter()-begin)*1000}
            predictions = {"calibration": c}
            for arm, model in models.items():
                begin = perf_counter()
                selected = select_actions(model(x).numpy(), valid, "absolute")
                predictions[arm] = apply_actions(c, boxes, selected)
                row[arm + "_score_select_ms"] = (perf_counter()-begin)*1000
            times.append(row)
            for arm, values in predictions.items():
                path = output / "detections" / arm / f"{stem}.txt"
                write_detection(path, values, record["size"])
                if arm in ("calibration", "uniform", "regret") and (
                        file_hash(path) != file_hash(history / "detections" / arm / path.name)):
                    raise ValueError("Frozen official detection export is not byte-identical.")
            if len(seen) == 1 or len(seen) % 64 == 0 or len(seen) == 548:
                print(f"Stable-order no-GT official replay: {len(seen)}/548, "
                      f"{perf_counter()-started:.1f}s", flush=True)
    if seen != set(expected):
        raise ValueError("Incomplete official input coverage.")
    write_csv(output / "timings.csv", times)
    root = matlab_string(output)
    data = matlab_string(original / "dataset")
    lib = matlab_string(output / "visdrone_helpers.oct")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    names = ",".join(f"'{a}'" for a in ARMS)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}'; dataset='{data}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}}; assert(any(strcmp(arm,{{{names}}})));
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==548);
[gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(root,'detections',arm),fullfile(dataset,'images'),548,names);
[a,b,c,d,e,f,g]=calcAccuracy(548,gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('STABLE_ORDER_AUDIT_COMPLETED %s\\n',arm);
""", encoding="ascii")
    inputs = {p.relative_to(output).as_posix(): file_hash(p)
              for p in output.rglob("*") if p.is_file()}
    save_json(output / "input_manifest.json", inputs)

    def run_arm(arm):
        begin = perf_counter()
        with (output / f"{arm}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), arm],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Official evaluator failed for {arm}; inspect its log.")
        values = loadmat(output / f"{arm}.mat")["values"].ravel()
        if len(values) != 7 or not np.isfinite(values).all():
            raise ValueError("Official metric vector is incomplete.")
        metrics = dict(zip(METRICS, values.tolist()))
        if arm in ("calibration", "uniform", "regret") and any(
                abs(metrics[k] - prior_official_metrics[arm][k]) > 1e-12 for k in METRICS):
            raise ValueError("Historical official control metrics did not reproduce.")
        save_json(output / f"{arm}_official.json", metrics)
        print(f"Official stable-order arm completed: {arm}, AP={metrics['AP']:.6f}, "
              f"{perf_counter()-begin:.1f}s", flush=True)
        return arm, metrics

    print(f"Running12 official arms, workers={workers}.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metrics = dict(pool.map(run_arm, ARMS))
    verdict = decide(metrics, registration["policy"])
    if any(file_hash(output / p) != h for p, h in inputs.items()):
        raise ValueError("Official evaluation inputs changed.")
    if any(file_hash(project_path(p)) != h for p, h in hashes.items()):
        raise ValueError("Audit source changed.")
    if any(file_hash(prior / p) != h for p, h in checkpoints.items()):
        raise ValueError("Audit checkpoints changed.")
    verify_runtime_files(original)
    timing = {key: {"mean": float(np.mean([r[key] for r in times])),
                    "median": float(np.median([r[key] for r in times]))}
              for key in times[0] if key != "image"}
    save_json(output / "official_metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "timing_summary.json", {
        "cpu_ms_per_image": timing,
        "limit": "Cached CPU replay stage timings; exclude detector views and evaluator. Not end-to-end deployment latency.",
    })
    save_json(output / "verification.json", {
        "all548": True, "all12_arms": True, "no_training": True, "detector_forwards": 0,
        "three_historical_controls_export_bytes_and_official_metrics_exact": True,
        "same_scores_classes_order_count": True, "inference_without_annotations": True,
        "inputs_sources_checkpoints_unchanged": True, "no_testdev": True,
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
