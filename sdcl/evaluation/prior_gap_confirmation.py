"""Official matched confirmation of the adapted ordinary IoU-gap ranking control."""

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
from sdcl.evaluation.ranking_seed_confirmation import seed0_metrics
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.evaluation.testdev_transfer import ORIGINAL
from sdcl.postprocess.localization import ActionScorer, apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.prior_gap_confirmation import REGISTRATION, RUN, checked_repeat, recipe
from sdcl.training.regret_probe import checked_inputs


POLICY = {
    "mean_AP_vs_uniform_min_exclusive_pp": 0., "positive_vs_uniform_seeds_min": 2,
    "mean_AP_vs_calibration_min_pp": .10, "mean_AR500_vs_calibration_min_pp": -.50,
}


def decide(prior, controls):
    if set(prior) != {"0", "1", "2"} or set(controls) != set(prior):
        raise ValueError("All three matched seeds are required.")
    for values in [*prior.values(), *(v for arms in controls.values() for v in arms.values())]:
        if set(values) != set(METRICS) or not all(
            np.isfinite(v) and 0 <= v <= 100 for v in values.values()
        ):
            raise ValueError("Invalid official prior-control metric vector.")
    comparisons = {}
    for arm in ("calibration", "absolute", "uniform", "regret"):
        ap = np.array([prior[str(s)]["AP"] - controls[str(s)][arm]["AP"] for s in range(3)])
        ar = np.array([prior[str(s)]["AR500"] - controls[str(s)][arm]["AR500"] for s in range(3)])
        comparisons[arm] = {
            "AP_per_seed_pp": ap.tolist(), "AP_mean_pp": float(ap.mean()),
            "AP_std_paired_head_seed_ddof1_pp": float(ap.std(ddof=1)),
            "AP_positive_seeds": int((ap > 0).sum()),
            "AR500_per_seed_pp": ar.tolist(), "AR500_mean_pp": float(ar.mean()),
        }
    matched, total = comparisons["uniform"], comparisons["calibration"]
    if matched["AP_mean_pp"] <= 0 or matched["AP_positive_seeds"] < 2:
        decision = "NO_REPEATABLE_ADVANTAGE_OVER_UNIFORM"
    elif total["AP_mean_pp"] <= 0:
        decision = "NO_AVERAGE_CONTROL_BENEFIT"
    elif total["AR500_mean_pp"] < -.50:
        decision = "AP_RECALL_TRADEOFF_EXCEEDS_BUDGET"
    elif total["AP_mean_pp"] < .10:
        decision = "SMALL_POSITIVE_SIGNAL"
    else:
        decision = "POSITIVE_ORDINARY_PRIOR_WORK_CONTROL"
    return {
        "decision": decision, "registered_policy": POLICY, "comparisons": comparisons,
        "summary": {
            k: {"mean": float(np.mean([prior[str(s)][k] for s in range(3)])),
                "std_head_seed_ddof1": float(np.std([prior[str(s)][k] for s in range(3)], ddof=1))}
            for k in METRICS
        },
        "scope": "Existing positive ranking-loss shape adapted to our fixed action task. "
                 "Not RankDetNet reproduction, novel loss, independent data or AP significance proof.",
    }


def evaluate(cache=CACHE, run=RUN, output=None, workers=2):
    if not 1 <= workers <= 3:
        raise ValueError("Use one to three official workers.")
    torch.set_num_threads(1)
    fixed, repeat = recipe(), checked_repeat()
    run = project_path(run)
    training = json.loads((run / "plan.json").read_text())
    if (training["recipe"] != fixed or not training["full_training_data"]
            or not all(json.loads((run / "verification.json").read_text()).values())
            or file_hash(project_path(REGISTRATION)) != training["registration_sha256"]
            or file_hash(run / "source.zip") != training["source_archive_sha256"]
            or any(file_hash(repeat / n) != h for n, h in training["repeat_inputs_sha256"].items())):
        raise ValueError("Prior-gap training/registration/repeat mismatch.")
    verify_dependencies(training["source_hashes"], run / "source.zip")
    checked_inputs(training["cache_path"], training)
    cache, cache_plan, records = checked_cache(cache)
    transfer, _ = transfer_inputs()
    if file_hash(transfer / "official_metrics.json") != cache_plan["prior_official_metrics_sha256"]:
        raise ValueError("Frozen ordinary calibration report changed.")
    runtime = verify_runtime_files(ORIGINAL)
    _, mean, std, _ = frozen_heads()
    controls = json.loads((repeat / "official_metrics.json").read_text())
    if controls["0"] != seed0_metrics(cache):
        raise ValueError("Original seed0 reuse changed.")
    control_hashes = {}
    for seed in (1, 2):
        for arm in ("absolute", "uniform", "regret"):
            path = repeat / f"{arm}_seed{seed}.mat"
            mat = dict(zip(METRICS, loadmat(path)["values"].ravel().tolist()))
            if mat != controls[str(seed)][arm]:
                raise ValueError("Matched repeat official MAT/report mismatch.")
            control_hashes[str(path)] = file_hash(path)
    models, checkpoints = {}, {}
    for seed in fixed["seeds"]:
        path = run / f"seed{seed}" / "last.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["seed"] != seed or saved["epoch"] != fixed["epochs"]
                or any(saved[k] != training[k] for k in ("recipe", "source_hashes", "cache_sha256",
                    "source_archive_sha256", "registration_sha256", "repeat_inputs_sha256"))
                or not np.array_equal(saved["mean"].numpy(), mean)
                or not np.array_equal(saved["std"].numpy(), std)):
            raise ValueError("Only frozen-normalization final prior-gap heads are eligible.")
        name = f"prior_gap_seed{seed}"
        models[name] = ActionScorer().eval()
        models[name].load_state_dict(saved["model"])
        checkpoints[str(path)] = file_hash(path)
    output = project_path(output or f"output/analysis/prior-gap-{datetime.now():%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "recipe": fixed, "policy": POLICY, "registration_sha256": training["registration_sha256"],
        "training_plan_sha256": file_hash(run / "plan.json"), "checkpoints_sha256": checkpoints,
        "matched_controls_MAT_sha256": control_hashes,
        "repeat_inputs_sha256": training["repeat_inputs_sha256"],
        "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "scope": "All1610 cached test-dev, prediction-only coordinate features. Ordinary control.",
    }
    save_json(output / "plan.json", plan)
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
                print(f"Prior-gap cached inference: {index}/{len(records)}", flush=True)
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
fprintf('PRIOR_GAP_COMPLETED %s\\n',arm);
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
            raise RuntimeError(f"Official prior-gap evaluation failed: {name}.log")
        values = loadmat(output / f"{name}.mat")["values"].ravel()
        if len(values) != len(METRICS):
            raise ValueError("Incomplete prior-gap official metric vector.")
        print(f"Prior-gap official evaluation completed: {name}", flush=True)
        return name, dict(zip(METRICS, values.tolist()))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(run_arm, models))
    metrics = {str(s): results[f"prior_gap_seed{s}"] for s in fixed["seeds"]}
    decision = decide(metrics, controls)
    save_json(output / "official_metrics.json", metrics)
    save_json(output / "decision.json", decision)
    checked_cache(cache)
    checked_inputs(training["cache_path"], training)
    transfer_inputs()
    checked_repeat()
    verify_runtime_files(ORIGINAL)
    verify_dependencies(training["source_hashes"], run / "source.zip")
    if (any(file_hash(p) != h for p, h in {**checkpoints, **control_hashes}.items())
            or any(file_hash(repeat / n) != h for n, h in training["repeat_inputs_sha256"].items())
            or file_hash(run / "plan.json") != plan["training_plan_sha256"]
            or any(file_hash(output / n) != h for n, h in inputs.items())):
        raise ValueError("Prior-gap official evidence changed.")
    save_json(output / "verification.json", {
        "all1610testdev_all_three_seeds": True, "matched_controls_MAT_verified": True,
        "same_scores_classes_order_count": True, "no_test_labels_in_predictions": True,
        "train_cache_checkpoint_source_match": True, "official_inputs_runtime_verified": True,
        "no_seed_or_checkpoint_selection": True, "no_detector_training_or_inference": True,
    })
    return {"output": str(output), "metrics": metrics, "decision": decision}
