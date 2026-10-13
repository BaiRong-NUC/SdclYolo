"""Official cached-action seed repeat, with aggregate gates fixed before training."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.io import loadmat
import torch

from sdcl.config import project_path
from sdcl.evaluation.candidate_series import CACHE, checked_cache, transfer_inputs
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.evaluation.testdev_transfer import ORIGINAL
from sdcl.postprocess.localization import ActionScorer, apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.ranking_seed_confirmation import ARMS, REGISTRATION, RUN, recipe
from sdcl.training.regret_probe import checked_inputs


POLICY = {
    "uniform_mean_AP_vs_calibration_min_pp": .10,
    "uniform_mean_AP_vs_absolute_min_exclusive_pp": 0.,
    "uniform_positive_vs_absolute_seeds_min": 2,
    "uniform_mean_AR500_vs_calibration_min_pp": -.50,
    "regret_mean_AP_vs_uniform_min_exclusive_pp": 0.,
    "regret_positive_vs_uniform_seeds_min": 2,
    "regret_mean_AR500_vs_calibration_min_pp": -.50,
}


def seed_decision(metrics):
    if set(metrics) != {"0", "1", "2"}:
        raise ValueError("Report all three seeds; do not select seeds.")
    for arms in metrics.values():
        if set(arms) != {"calibration", *ARMS}:
            raise ValueError("Incomplete matched arms.")
        for values in arms.values():
            if set(values) != set(METRICS) or not all(
                np.isfinite(v) and 0 <= v <= 100 for v in values.values()
            ):
                raise ValueError("Invalid official metric vector.")
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for metric in METRICS:
            values = np.array([metrics[str(s)][arm][metric] for s in range(3)])
            summary[arm][metric] = {
                "mean": float(values.mean()), "std_head_seed_ddof1": float(values.std(ddof=1)),
                "min": float(values.min()), "max": float(values.max()),
            }
    pairs = {}
    for arm, control in (("uniform", "absolute"), ("regret", "uniform"),
                         ("uniform", "calibration"), ("regret", "calibration")):
        values = np.array([metrics[str(s)][arm]["AP"] - metrics[str(s)][control]["AP"]
                           for s in range(3)])
        ars = np.array([metrics[str(s)][arm]["AR500"] - metrics[str(s)][control]["AR500"]
                        for s in range(3)])
        pairs[f"{arm}_vs_{control}"] = {
            "AP_per_seed_pp": values.tolist(), "AP_mean_pp": float(values.mean()),
            "AP_std_paired_head_seed_ddof1_pp": float(values.std(ddof=1)),
            "AP_positive_seeds": int((values > 0).sum()),
            "AR500_per_seed_pp": ars.tolist(), "AR500_mean_pp": float(ars.mean()),
        }
    task, total = pairs["uniform_vs_absolute"], pairs["uniform_vs_calibration"]
    if task["AP_mean_pp"] <= 0 or task["AP_positive_seeds"] < 2:
        ranking = "NO_REPEATABLE_MATCHED_ADVANTAGE"
    elif total["AP_mean_pp"] <= 0:
        ranking = "NO_AVERAGE_CONTROL_BENEFIT"
    elif total["AR500_mean_pp"] < -.50:
        ranking = "AP_RECALL_TRADEOFF_EXCEEDS_BUDGET"
    elif total["AP_mean_pp"] < .10:
        ranking = "SMALL_POSITIVE_SIGNAL"
    else:
        ranking = "PROMISING_HEAD_SEED_REPEAT"
    extra, regret_total = pairs["regret_vs_uniform"], pairs["regret_vs_calibration"]
    weighting = (
        "POSITIVE_EXTRA_HEAD_SEED_SIGNAL"
        if extra["AP_mean_pp"] > 0 and extra["AP_positive_seeds"] >= 2
        and regret_total["AR500_mean_pp"] >= -.50
        else "NO_REPEATABLE_EXTRA_ADVANTAGE_WITHIN_BUDGET"
    )
    return {
        "ranking_task": ranking, "regret_weight": weighting,
        "registered_policy": POLICY, "paired_comparisons": pairs, "summary": summary,
        "scope": "Same already-viewed test-dev; three head seeds, one frozen detector/calibrator. "
                 "No seed selection, no AP confidence interval or independent-data claim.",
    }


def seed0_metrics(cache):
    result = {}
    for candidate in ("relative", "ranking"):
        previous = project_path(f"output/analysis/candidate-{candidate}-fixed-20261013")
        plan = json.loads((previous / "plan.json").read_text())
        if (plan["cache_plan_sha256"] != file_hash(cache / "plan.json")
                or not all(json.loads((previous / "verification.json").read_text()).values())):
            raise ValueError("Seed0 must come from the verified shared cache.")
        metrics = json.loads((previous / "official_metrics.json").read_text())
        for arm in ("calibration", *ARMS):
            if arm not in metrics:
                continue
            mat = dict(zip(METRICS, loadmat(previous / f"{arm}.mat")["values"].ravel().tolist()))
            if mat != metrics[arm] or (arm in result and result[arm] != mat):
                raise ValueError("Seed0 MAT/report inconsistency.")
            result[arm] = mat
    return result


def evaluate(cache=CACHE, run=RUN, output=None, workers=2):
    if not 1 <= workers <= 3:
        raise ValueError("Use one to three official workers.")
    torch.set_num_threads(1)
    fixed = recipe()
    run = project_path(run)
    training = json.loads((run / "plan.json").read_text())
    expected = json.loads(json.dumps(fixed.__dict__))
    if (training["recipe"] != expected or not training["full_training_data"]
            or not all(json.loads((run / "verification.json").read_text()).values())
            or file_hash(project_path(REGISTRATION)) != training["registration_sha256"]
            or file_hash(run / "source.zip") != training["source_archive_sha256"]):
        raise ValueError("Seed-training provenance mismatch.")
    verify_dependencies(training["source_hashes"], run / "source.zip")
    checked_inputs(training["cache_path"], training)
    cache, cache_plan, records = checked_cache(cache)
    transfer, _ = transfer_inputs()
    if file_hash(transfer / "official_metrics.json") != cache_plan["prior_official_metrics_sha256"]:
        raise ValueError("Pinned official calibration report changed.")
    runtime = verify_runtime_files(ORIGINAL)
    _, mean, std, _ = frozen_heads()
    models, checkpoints = {}, {}
    for seed in fixed.seeds:
        path = run / f"seed{seed}" / "last.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["epoch"] != fixed.epochs or saved["seed"] != seed
                or any(saved[k] != training[k] for k in
                       ("recipe", "source_hashes", "source_archive_sha256",
                        "cache_sha256", "registration_sha256"))
                or not np.array_equal(saved["mean"].numpy(), mean)
                or not np.array_equal(saved["std"].numpy(), std)):
            raise ValueError("Only matched final heads with frozen normalization are eligible.")
        checkpoints[str(path)] = file_hash(path)
        for arm in ARMS:
            name = f"{arm}_seed{seed}"
            models[name] = ActionScorer().eval()
            models[name].load_state_dict(saved["models"][arm])
    seed0 = seed0_metrics(cache)
    controls_hashes = {
        str(p): file_hash(p)
        for candidate in ("relative", "ranking")
        for p in project_path(f"output/analysis/candidate-{candidate}-fixed-20261013").glob("*.mat")
    }
    output = project_path(output or f"output/analysis/ranking-seeds-{datetime.now():%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "plan.json", {
        "recipe": training["recipe"], "policy": POLICY, "registration_sha256": training["registration_sha256"],
        "training_plan_sha256": file_hash(run / "plan.json"), "checkpoints_sha256": checkpoints,
        "seed0_MAT_sha256": controls_hashes, "seed0_metrics": seed0,
        "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "baseline_metrics_sha256": cache_plan["prior_official_metrics_sha256"],
        "scope": "All1610 test-dev; cached prediction-only features, unchanged detector/calibrator.",
    })
    shutil.copyfile(Path(__file__), output / "evaluation_source.py")
    for directory in ("toolkit", "compat"):
        shutil.copytree(transfer / directory, output / directory)
    shutil.copyfile(transfer / "visdrone_helpers.oct", output / "visdrone_helpers.oct")
    for name in models:
        (output / "detections" / name).mkdir(parents=True)
    chunks = json.loads((cache / "chunks.json").read_text())
    timings = []
    with torch.inference_mode(), (output / "predictions.jsonl").open("x") as stream:
        for index, record in enumerate(records, 1):
            stem = record["stem"]
            with np.load(cache / chunks[stem]["file"], allow_pickle=False) as state:
                x = torch.from_numpy(state["normalized"])
                c, boxes, valid = state["calibration"], state["boxes"], state["valid"]
            row, values = {"image": stem}, {}
            for name, model in models.items():
                tick = perf_counter()
                selected = select_actions(model(x).numpy(), valid, "absolute")
                variant = apply_actions(c, boxes, selected)
                row[name + "_cached_head_selection_ms"] = (perf_counter() - tick) * 1000
                values[name] = variant.tolist()
                write_detection(output / "detections" / name / f"{stem}.txt", variant, record["size"])
            stream.write(json.dumps({"image": stem, "arms": values}) + "\n")
            timings.append(row)
            if index == 1 or index % 200 == 0 or index == len(records):
                print(f"Seed confirmation cached inference: {index}/{len(records)}", flush=True)
    write_csv(output / "timings.csv", timings)
    root, dataset = matlab_string(output), matlab_string(transfer / "dataset")
    lib = matlab_string(output / "visdrone_helpers.oct")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    names = ",".join(f"'{name}'" for name in models)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}'; dataset='{dataset}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}}; assert(any(strcmp(arm,{{{names}}})));
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==1610);
[gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(root,'detections',arm),fullfile(dataset,'images'),1610,names);
[a,b,c,d,e,f,g]=calcAccuracy(1610,gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('SEED_CONFIRMATION_COMPLETED %s\\n',arm);
""", encoding="ascii")
    inputs = {p.relative_to(output).as_posix(): file_hash(p)
              for p in output.rglob("*") if p.is_file()}
    save_json(output / "input_manifest.json", inputs)

    def run_arm(name):
        with (output / f"{name}.log").open("x") as log:
            done = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), name],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT,
            )
        if done.returncode:
            raise RuntimeError(f"Official seed evaluation failed: {name}.log")
        values = loadmat(output / f"{name}.mat")["values"].ravel()
        if len(values) != len(METRICS):
            raise ValueError("Incomplete seed official metric vector.")
        print(f"Seed official evaluation completed: {name}", flush=True)
        return name, dict(zip(METRICS, values.tolist()))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(run_arm, models))
    metrics = {"0": seed0}
    for seed in fixed.seeds:
        metrics[str(seed)] = {"calibration": seed0["calibration"],
                             **{a: results[f"{a}_seed{seed}"] for a in ARMS}}
    decision = seed_decision(metrics)
    save_json(output / "official_metrics.json", metrics)
    save_json(output / "decision.json", decision)
    checked_cache(cache)
    checked_inputs(training["cache_path"], training)
    transfer_inputs()
    verify_runtime_files(ORIGINAL)
    verify_dependencies(training["source_hashes"], run / "source.zip")
    if (any(file_hash(p) != h for p, h in {**checkpoints, **controls_hashes}.items())
            or file_hash(run / "plan.json") != json.loads((output / "plan.json").read_text())[
                "training_plan_sha256"]
            or any(file_hash(output / n) != h for n, h in inputs.items())):
        raise ValueError("Seed evidence changed during evaluation.")
    if seed0_metrics(cache) != seed0:
        raise ValueError("Seed0 reuse changed.")
    save_json(output / "verification.json", {
        "all1610testdev": True, "both_new_seeds_all_matched_arms": True,
        "checkpoint_train_cache_source_match": True, "seed0_MAT_reuse_verified": True,
        "same_scores_classes_order_count": True, "no_test_labels_in_predictions": True,
        "runtime_and_official_inputs_verified": True, "no_seed_or_checkpoint_selection": True,
        "no_detector_training_or_inference": True,
    })
    return {"output": str(output), "official_metrics": metrics, "decision": decision}
