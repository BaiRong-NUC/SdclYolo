"""Evaluation-only IoF-ignore size diagnostics, preserving all raw predictions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import yaml

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, load_records, record_hash


def size_worker(job):
    arm, records, predictions, names = job
    torch.set_num_threads(1)
    print(f"Post-prediction IoF-ignore size diagnostic started: {arm}", flush=True)
    metrics = coco_size_metrics(records, predictions, names, reference_size=640)
    print(f"Post-prediction IoF-ignore size diagnostic completed: {arm}", flush=True)
    return arm, metrics


def supplement(cache, threshold_capture, output, workers=2):
    if not 1 <= workers <= 3:
        raise ValueError("Use one to three diagnostic workers.")
    torch.set_num_threads(1)
    cache, threshold_capture, output = map(project_path, (cache, threshold_capture, output))
    if output.exists():
        raise FileExistsError("Do not overwrite diagnostic evidence.")
    dataset = project_path("data/visdrone")
    records = load_records(dataset, "test")
    if len(records) != 1610:
        raise ValueError("This supplement requires all1610test-dev images.")
    expected = {r["stem"]: r for r in records}
    names = yaml.safe_load((dataset / "dataset.yaml").read_text())["names"]
    names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    captures = (cache / "predictions.jsonl", threshold_capture)
    hashes = {str(p): file_hash(p) for p in captures}
    arms = ("calibration", "absolute", "gain", "gain_always", "uniform", "regret",
            "threshold_absolute", "threshold_gain")
    predictions = {a: {} for a in arms}
    for source, allowed in zip(captures, (arms[:6], arms[6:])):
        with source.open() as stream:
            for line in stream:
                item = json.loads(line)
                stem = item["image"]
                if stem not in expected:
                    raise ValueError("Unexpected captured image.")
                for arm in allowed:
                    if stem in predictions[arm]:
                        raise ValueError("Duplicate captured image.")
                    predictions[arm][stem] = np.asarray(item["arms"][arm]).reshape(-1, 6)
    if any(set(mapping) != set(expected) for mapping in predictions.values()):
        raise ValueError("Incomplete diagnostic predictions.")
    removed = dict.fromkeys(arms, 0)
    for r in records:
        # Annotation information is used only after the immutable predictions exist.
        ignores = json.loads((dataset / "metadata/test" / f"{r['stem']}.json").read_text())[
            "ignore_boxes_xyxy"
        ]
        ignored = torch.tensor(ignores, dtype=torch.float64).reshape(-1, 4)
        for arm in arms:
            values = predictions[arm][r["stem"]]
            mask = prediction_ignore_mask(torch.from_numpy(values[:, :4]), ignored, .5).numpy()
            predictions[arm][r["stem"]] = values[~mask]
            removed[arm] += int(mask.sum())
    output.mkdir(parents=True)
    dependencies = (Path(__file__), project_path("sdcl/data/geometry.py"),
                    project_path("sdcl/evaluation/size_analysis.py"))
    source_hashes = {str(p.resolve()): file_hash(p) for p in dependencies}
    shutil.copyfile(__file__, output / "analysis_source.py")
    plan = {
        "images": 1610, "reference_size": 640, "captures_sha256": hashes,
        "records_sha256": record_hash(records), "source_hashes": source_hashes,
        "removed_predictions": removed,
        "scope": "COCO-style reference640; GT ignore boxes used ONLY for post-prediction IoF>0.5 "
                 "evaluation masking. NOT official VisDrone tiny/small AP. No prediction, training "
                 "or registered investment-decision changes.",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    jobs = [(a, records, predictions[a], names) for a in arms]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        metrics = dict(pool.map(size_worker, jobs))
    if (any(file_hash(p) != h for p, h in hashes.items())
            or any(file_hash(p) != h for p, h in source_hashes.items())
            or record_hash(load_records(dataset, "test")) != plan["records_sha256"]):
        raise ValueError("Supplement inputs/source changed.")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    verification = {
        "all1610images": True, "raw_prediction_captures_unchanged": True,
        "same_evaluation_mask_rule_for_all_arms": True,
        "labels_only_after_prediction_generation": True, "registered_decisions_unchanged": True,
        "records_source_unchanged": True,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return {"output": str(output), "removed_predictions": removed, "verification": verification}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="output/candidate_series/testdev_actions_20261013")
    parser.add_argument("--threshold-capture",
                        default="output/analysis/threshold-gain-probe-fixed-20261013/predictions.jsonl")
    parser.add_argument("--output",
                        default="output/analysis/candidate-size-postignore-fixed-20261013")
    parser.add_argument("--workers", type=int, default=2)
    print(json.dumps(supplement(**vars(parser.parse_args())), indent=2))
