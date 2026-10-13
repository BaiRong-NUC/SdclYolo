"""Fixed three-head coordinate agreement; prediction-only ordinary ensemble control."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.io import loadmat
import torch

from sdcl.config import project_path
from sdcl.evaluation.candidate_series import CACHE, checked_cache, sources, transfer_inputs
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.official_runtime import HELPERS, octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.ranking_seed_confirmation import seed0_metrics
from sdcl.evaluation.size_analysis import file_hash
from sdcl.evaluation.testdev_transfer import ORIGINAL
from sdcl.postprocess.rescue import save_json
from sdcl.training.prior_gap_confirmation import checked_repeat
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/960_coordinate_consensus_registration_20261013.md"
ARMS = ("uniform_consensus", "regret_consensus")
POLICY = {
    "AP_vs_calibration_min_pp": .10, "AP_vs_seed0_regret_min_pp": -.10,
    "AR500_vs_seed0_regret_min_exclusive_pp": 0.,
    "AP_vs_uniform_consensus_min_exclusive_pp": 0.,
}


def consensus(c, predictions):
    if len(predictions) != 3:
        raise ValueError("Use all three fixed heads.")
    values = [np.asarray(v).reshape(-1, 6) for v in predictions]
    if any(v.shape != c.shape or not np.array_equal(v[:, 4:], c[:, 4:]) for v in values):
        raise ValueError("Head predictions changed frozen scores/classes/order/count.")
    agreement = (values[0][:, :4] == values[1][:, :4]).all(1)
    agreement &= (values[0][:, :4] == values[2][:, :4]).all(1)
    result = c.copy()
    result[agreement, :4] = values[0][agreement, :4]
    if (not np.isfinite(result).all() or np.any(result[:, 2:4] <= result[:, :2])):
        raise ValueError("Invalid consensus coordinates.")
    return result


def decide(metrics, original):
    if set(metrics) != set(ARMS) or set(original) != {"calibration", "regret"}:
        raise ValueError("Missing fixed consensus controls.")
    for m in [*metrics.values(), *original.values()]:
        if set(m) != set(METRICS) or not all(np.isfinite(v) and 0 <= v <= 100 for v in m.values()):
            raise ValueError("Invalid official consensus vector.")
    method = metrics["regret_consensus"]
    ap = method["AP"] - original["calibration"]["AP"]
    retention = method["AP"] - original["regret"]["AP"]
    ar = method["AR500"] - original["regret"]["AR500"]
    matched = method["AP"] - metrics["uniform_consensus"]["AP"]
    if retention < -.10 or ar <= 0:
        verdict = "NO_BUDGETED_AP_RECALL_TRADEOFF"
    elif ap <= 0:
        verdict = "NO_CONTROL_BENEFIT"
    elif ap < .10:
        verdict = "SMALL_POSITIVE_SIGNAL"
    elif matched <= 0:
        verdict = "ORDINARY_CONSENSUS_TRADEOFF_ONLY"
    else:
        verdict = "PROMISING_CONSENSUS_TRADEOFF"
    return {
        "decision": verdict, "registered_policy": POLICY,
        "AP_vs_calibration_pp": ap, "AP_vs_seed0_regret_pp": retention,
        "AR500_vs_seed0_regret_pp": ar, "AP_vs_uniform_consensus_pp": matched,
        "AR500_vs_calibration_pp": method["AR500"] - original["calibration"]["AR500"],
        "scope": "Standard agreement/retention ensemble, no new training or novelty claim. "
                 "Already-viewed same test-dev, not independent-data or significance evidence.",
    }


def evaluate(cache=CACHE, output=None, workers=2):
    if not 1 <= workers <= 3:
        raise ValueError("Use one to three official workers.")
    torch.set_num_threads(1)
    cache, cache_plan, records = checked_cache(cache)
    repeat = checked_repeat()
    transfer, _ = transfer_inputs()
    runtime = verify_runtime_files(ORIGINAL)
    if file_hash(transfer / "official_metrics.json") != cache_plan["prior_official_metrics_sha256"]:
        raise ValueError("Frozen official control changed.")
    controls = seed0_metrics(cache)
    captures = (cache / "predictions.jsonl", repeat / "predictions.jsonl")
    references = [*captures, project_path(REGISTRATION), Path(__file__).resolve(),
                  repeat / "official_metrics.json", repeat / "verification.json"]
    hashes = {str(p): file_hash(p) for p in references}
    output = project_path(output or f"output/analysis/coordinate-consensus-{datetime.now():%Y%m%d-%H%M%S-%f}")
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = sources()
    source_hashes[REGISTRATION] = file_hash(project_path(REGISTRATION))
    archive_source(output, source_hashes)
    save_json(output / "plan.json", {
        "policy": POLICY, "references_sha256": hashes,
        "source_hashes": source_hashes, "source_archive_sha256": file_hash(output / "source.zip"),
        "cache_plan_sha256": file_hash(cache / "plan.json"),
        "cache_input_manifest_sha256": file_hash(cache / "input_manifest.json"),
        "heads": [0, 1, 2], "agreement": "Exact coordinate unanimity, otherwise keep.",
        "scope": "Two matched existing-head ensembles; no labels in prediction or threshold selection.",
    })
    shutil.copyfile(Path(__file__), output / "evaluation_source.py")
    for directory in ("toolkit", "compat"):
        shutil.copytree(transfer / directory, output / directory)
    shutil.copyfile(transfer / "visdrone_helpers.oct", output / "visdrone_helpers.oct")
    for arm in ARMS:
        (output / "detections" / arm).mkdir(parents=True)
    changed = dict.fromkeys(ARMS, 0)
    with captures[0].open() as first, captures[1].open() as second, (output / "predictions.jsonl").open("x") as stream:
        for index, (record, a, b) in enumerate(zip(records, first, second, strict=True), 1):
            a, b = json.loads(a), json.loads(b)
            if record["stem"] != a["image"] or a["image"] != b["image"]:
                raise ValueError("Consensus capture images/order mismatch.")
            c = np.asarray(a["arms"]["calibration"]).reshape(-1, 6)
            predictions = {}
            for arm, name in zip(ARMS, ("uniform", "regret")):
                prediction = consensus(c, [a["arms"][name], b["arms"][name + "_seed1"],
                                           b["arms"][name + "_seed2"]])
                changed[arm] += int((prediction[:, :4] != c[:, :4]).any(1).sum())
                predictions[arm] = prediction.tolist()
                write_detection(output / "detections" / arm / f"{record['stem']}.txt",
                                prediction, record["size"])
            stream.write(json.dumps({"image": record["stem"], "arms": predictions}) + "\n")
            if index == 1 or index % 400 == 0 or index == len(records):
                print(f"Coordinate consensus generation: {index}/{len(records)}", flush=True)
    save_json(output / "coordinate_changes.json", {
        "changed_boxes": changed, "scope": "Prediction-only frequency, not correct localization.",
    })
    root, dataset = matlab_string(output), matlab_string(transfer / "dataset")
    lib = matlab_string(output / "visdrone_helpers.oct")
    setup = "\n".join(f"autoload('{name}','{lib}');" for name in HELPERS)
    names = ",".join(f"'{name}'" for name in ARMS)
    script = output / "run_official.m"
    script.write_text(f"""root='{root}'; dataset='{dataset}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
args=argv(); arm=args{{1}}; assert(any(strcmp(arm,{{{names}}})));
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==1610);
[gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(root,'detections',arm),fullfile(dataset,'images'),1610,names);
[a,b,c,d,e,f,g]=calcAccuracy(1610,gt,dt); values=[a,b,c,d,e,f,g];
save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
fprintf('CONSENSUS_COMPLETED %s\\n',arm);
""", encoding="ascii")
    inputs = {p.relative_to(output).as_posix(): file_hash(p)
              for p in output.rglob("*") if p.is_file()}
    save_json(output / "input_manifest.json", inputs)

    def run_arm(arm):
        with (output / f"{arm}.log").open("x") as log:
            done = subprocess.run(
                [runtime["octave_path"], "--no-gui", "--quiet", "--no-history", str(script), arm],
                env=octave_env(runtime["octave_path"]), stdout=log, stderr=subprocess.STDOUT,
            )
        if done.returncode:
            raise RuntimeError(f"Official consensus evaluation failed: {arm}.log")
        values = loadmat(output / f"{arm}.mat")["values"].ravel()
        if len(values) != len(METRICS):
            raise ValueError("Incomplete official consensus metrics.")
        print(f"Consensus official evaluation completed: {arm}", flush=True)
        return arm, dict(zip(METRICS, values.tolist()))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        metrics = dict(pool.map(run_arm, ARMS))
    original = {a: controls[a] for a in ("calibration", "regret")}
    decision = decide(metrics, original)
    save_json(output / "official_metrics.json", {"controls": original, "methods": metrics})
    save_json(output / "decision.json", decision)
    checked_cache(cache)
    checked_repeat()
    transfer_inputs()
    verify_runtime_files(ORIGINAL)
    verify_dependencies(source_hashes, output / "source.zip")
    if (seed0_metrics(cache) != controls
            or file_hash(output / "source.zip") != json.loads((output / "plan.json").read_text())[
                "source_archive_sha256"]
            or any(file_hash(p) != h for p, h in hashes.items())
            or any(file_hash(output / n) != h for n, h in inputs.items())):
        raise ValueError("Consensus inputs/source/control changed.")
    save_json(output / "verification.json", {
        "all1610testdev": True, "all_three_fixed_heads": True,
        "same_scores_classes_order_count": True, "no_annotations_in_prediction": True,
        "input_source_control_hashes_match": True, "official_inputs_runtime_verified": True,
        "no_training_detector_inference_or_retuning": True,
    })
    return {"output": str(output), "metrics": metrics, "decision": decision}
