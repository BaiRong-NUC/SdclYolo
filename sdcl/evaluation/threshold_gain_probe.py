"""Confirm registered threshold heads on prediction-only cached test-dev actions."""

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
import yaml

from sdcl.config import project_path
from sdcl.evaluation.candidate_series import CACHE, checked_cache, confirm_decision, transfer_inputs
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, write_csv
from sdcl.evaluation.testdev_transfer import ORIGINAL, OUTPUT as TRANSFER
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.threshold_gain_probe import ARMS, RUN, ThresholdScorer, recipe


def evaluate(cache=CACHE, checkpoint=f"{RUN}/last.pt", output=None, workers=2):
    if not 1 <= workers <= 3:
        raise ValueError("Use one to three official evaluator workers.")
    torch.set_num_threads(1)
    fixed = recipe()
    checkpoint = project_path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != fixed.epochs or not saved["full_training_data"]:
        raise ValueError("Only the registered full-data final epoch20 is eligible.")
    from dataclasses import asdict
    if saved["recipe"] != asdict(fixed):
        raise ValueError("Threshold checkpoint recipe changed.")
    archive = checkpoint.parent / "source.zip"
    if file_hash(archive) != saved["source_archive_sha256"]:
        raise ValueError("Threshold training archive changed.")
    verify_dependencies(saved["source_hashes"], archive)
    if file_hash(project_path(saved["cache_path"]) / "manifest.json") != saved["cache_sha256"]:
        raise ValueError("Training cache manifest changed.")
    if file_hash(project_path("docs/960_threshold_gain_probe_registration_20261013.md")) != saved["registration_sha256"]:
        raise ValueError("Threshold pre-result registration changed.")
    cache, plan, records = checked_cache(cache)
    _, mean, std, _ = frozen_heads()
    if not np.array_equal(saved["mean"].numpy(), mean) or not np.array_equal(saved["std"].numpy(), std):
        raise ValueError("Threshold heads no longer match the normalized action cache.")
    runtime = verify_runtime_files(ORIGINAL)
    transfer, _ = transfer_inputs()
    models = {}
    for arm in ARMS:
        models[arm] = ThresholdScorer().eval()
        models[arm].load_state_dict(saved["models"][arm])
    output = project_path(output or f"output/analysis/threshold-gain-probe-{datetime.now():%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True, exist_ok=False)
    local_plan = {
        "registered_recipe": saved["recipe"], "registered_policy": plan["policy"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "registration_sha256": saved["registration_sha256"], "arms": ("calibration", *ARMS),
        "scope": "Registered before coordinate confirmations; one seed, shared public test-dev. "
                 "Same-GT threshold labels are not full AP supervision.",
    }
    save_json(output / "plan.json", local_plan)
    shutil.copyfile(Path(__file__), output / "evaluation_source.py")
    for name in ("toolkit", "compat"):
        shutil.copytree(transfer / name, output / name)
    shutil.copyfile(transfer / "visdrone_helpers.oct", output / "visdrone_helpers.oct")
    for arm in ARMS:
        (output / "detections" / arm).mkdir(parents=True)
    chunks = json.loads((cache / "chunks.json").read_text())
    predictions, timings = {a: {} for a in ("calibration", *ARMS)}, []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            stem = record["stem"]
            with np.load(cache / chunks[stem]["file"], allow_pickle=False) as state:
                x = torch.from_numpy(state["normalized"])
                c, boxes, valid = state["calibration"], state["boxes"], state["valid"]
            predictions["calibration"][stem] = c
            row = {"image": stem}
            for arm in ARMS:
                tick = perf_counter()
                utility = models[arm](x).numpy().mean(-1)
                variant = "absolute" if arm == "threshold_absolute" else "gain"
                selected = select_actions(utility, valid, variant)
                values = apply_actions(c, boxes, selected)
                row[arm + "_cached_head_selection_ms"] = (perf_counter() - tick) * 1000
                predictions[arm][stem] = values
                write_detection(output / "detections" / arm / f"{stem}.txt", values, record["size"])
            timings.append(row)
            if index == 1 or index % 200 == 0 or index == len(records):
                print(f"Threshold test-dev cached inference: {index}/{len(records)}", flush=True)
    write_csv(output / "timings.csv", timings)
    with (output / "predictions.jsonl").open("x") as stream:
        for record in records:
            stem = record["stem"]
            stream.write(json.dumps({"image": stem, "arms": {
                a: mapping[stem].tolist() for a, mapping in predictions.items()
            }}) + "\n")
    root, dataset = matlab_string(output), matlab_string(transfer / "dataset")
    lib = matlab_string(output / "visdrone_helpers.oct")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    names = ",".join(f"'{a}'" for a in ARMS)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}'; dataset='{dataset}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}}; assert(any(strcmp(arm,{{{names}}})));
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==1610);
[gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(root,'detections',arm),fullfile(dataset,'images'),1610,names);
[a,b,c,d,e,f,g]=calcAccuracy(1610,gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('THRESHOLD_PROBE_COMPLETED %s\\n',arm);
""", encoding="ascii")
    inputs = {p.relative_to(output).as_posix(): file_hash(p)
              for p in output.rglob("*") if p.is_file()}
    save_json(output / "input_manifest.json", inputs)

    def run_arm(arm):
        with (output / f"{arm}.log").open("x") as log:
            done = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), arm],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT
            )
        if done.returncode:
            raise RuntimeError(f"Threshold official evaluation failed: {arm}.log")
        values = loadmat(output / f"{arm}.mat")["values"].ravel()
        if len(values) != len(METRICS):
            raise ValueError("Incomplete threshold official metrics.")
        print(f"Threshold official evaluation completed: {arm}", flush=True)
        return arm, dict(zip(METRICS, values.tolist()))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        metrics = dict(pool.map(run_arm, ARMS))
    metrics["calibration"] = json.loads((transfer / "official_metrics.json").read_text())["calibration"]
    decision = confirm_decision("relative", {
        "calibration": metrics["calibration"], "absolute": metrics["threshold_absolute"],
        "gain": metrics["threshold_gain"],
    })
    decision.update(candidate="threshold_supervision",
                    primary="official AP: threshold_gain minus threshold_absolute")
    comparisons = {}
    for candidate in ("relative", "ranking"):
        previous = project_path(f"output/analysis/candidate-{candidate}-fixed-20261013")
        prior_plan = json.loads((previous / "plan.json").read_text())
        if (prior_plan["cache_plan_sha256"] != local_plan["cache_plan_sha256"]
                or not all(json.loads((previous / "verification.json").read_text()).values())):
            raise ValueError("Prior candidate comparison is not verified on this cache.")
        prior_metrics = json.loads((previous / "official_metrics.json").read_text())
        for arm in ("gain", "uniform", "regret"):
            if arm in prior_metrics:
                comparisons[arm] = {
                    key + "_pp": metrics["threshold_gain"][key] - prior_metrics[arm][key]
                    for key in METRICS
                }
    if decision["decision"] == "PROMISING_POINT_ESTIMATE" and comparisons["regret"]["AP_pp"] <= 0:
        decision["decision"] = "POSITIVE_BUT_BELOW_FROZEN_RANKING"
    decision["existing_frozen_controls"] = comparisons
    save_json(output / "official_metrics.json", metrics)
    save_json(output / "decision.json", decision)
    names = yaml.safe_load(project_path("data/visdrone/dataset.yaml").read_text())["names"]
    names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    sizes = {}
    for arm in predictions:
        print(f"Threshold secondary reference640 metrics: {arm}", flush=True)
        sizes[arm] = coco_size_metrics(records, predictions[arm], names, reference_size=640)
    save_json(output / "size_diagnostic.json", {
        "scope": "COCO-style/IoF-ignore reference640, not official VisDrone size AP.", "metrics": sizes
    })
    checked_cache(cache)
    transfer_inputs()
    verify_runtime_files(ORIGINAL)
    verify_dependencies(saved["source_hashes"], archive)
    if file_hash(checkpoint) != local_plan["checkpoint_sha256"]:
        raise ValueError("Threshold checkpoint changed during evaluation.")
    if any(file_hash(output / name) != digest for name, digest in inputs.items()):
        raise ValueError("Threshold official inputs changed.")
    save_json(output / "verification.json", {
        "all1610testdev": True, "source_cache_checkpoint_match": True,
        "same_scores_classes_order_count": True, "no_test_labels_in_predictions": True,
        "runtime_verified": True, "official_inputs_unchanged": True,
        "no_detector_training_or_inference": True, "no_checkpoint_or_threshold_selection": True,
    })
    return {"output": str(output), "metrics": metrics, "decision": decision}
