"""Evaluate the registered fixed retrieval rule on existing diagnostic pools."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold,
)
from sdcl.postprocess.censored_retrieval import retrieve
from sdcl.postprocess.rescue import save_json, soft_nms_with_origins
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate22_censored_retrieval_registration_20261014.json"
OUTPUT = "output/analysis/censored-retrieval-fixed-20261014"
PRIOR = "output/analysis/native-censoring-fixed-20261014"
BASE = "output/analysis/class-choice-pilot-fixed-20261014"
GROUPS = ("all", "tiny", "small_all")
ARMS = ("uniform", "contrast", "native_rank", "ordinary_agreement",
        "uncapped_native_rank", "ordinary_dense_softnms")


def score_image(predictions, extra, gt, classes, areas):
    ious = box_iou_matrix(gt, predictions[:, :4])
    matched, used = match_at_threshold(ious, classes, predictions, .25, .5)
    ten = [match_at_threshold(ious, classes, predictions, .25, float(t))[0] >= 0
           for t in np.linspace(.5, .95, 10)]
    extra_fp = extra & ~used
    same = classes[:, None] == predictions[None, :, 5]
    duplicates = extra_fp & ((ious >= .5) & same).any(0)
    return {
        "groups": {g: {
            "gt": int(group_mask(areas, g).sum()),
            "tp50": int(((matched >= 0) & group_mask(areas, g)).sum()),
            "tp_ten": int(sum((m & group_mask(areas, g)).sum() for m in ten)),
        } for g in GROUPS},
        "fp50": int(((predictions[:, 4] >= .25) & ~used).sum()),
        "additions": int(extra.sum()), "added_tp50": int((extra & used).sum()),
        "added_duplicate_fp50": int(duplicates.sum()),
        "added_other_fp50": int((extra_fp & ~duplicates).sum()),
        "matched": matched,
    }


def decide(metrics, policy):
    base, local = metrics["uniform"], metrics["contrast"]
    values = {
        "tiny_recall_vs_uniform_pp": local["tiny_recall_percent"] - base["tiny_recall_percent"],
        "small_all_recall_vs_uniform_pp": local["small_all_recall_percent"] -
        base["small_all_recall_percent"],
        "small_all_recall_vs_native_rank_pp": local["small_all_recall_percent"] -
        metrics["native_rank"]["small_all_recall_percent"],
        "small_all_recall_vs_ordinary_agreement_pp": local["small_all_recall_percent"] -
        metrics["ordinary_agreement"]["small_all_recall_percent"],
        "all_fppi_vs_uniform": local["all_fppi"] - base["all_fppi"],
        "added_match_precision": local["added_tp50"] / max(local["additions"], 1),
    }
    checks = {
        "tiny_gain": values["tiny_recall_vs_uniform_pp"] >=
        policy["tiny_recall_vs_uniform_min_pp"],
        "small_gain": values["small_all_recall_vs_uniform_pp"] >=
        policy["small_all_recall_vs_uniform_min_pp"],
        "native_rank_gain": values["small_all_recall_vs_native_rank_pp"] >
        policy["small_all_recall_vs_native_rank_min_exclusive_pp"],
        "agreement_gain": values["small_all_recall_vs_ordinary_agreement_pp"] >
        policy["small_all_recall_vs_ordinary_agreement_min_exclusive_pp"],
        "fp_budget": values["all_fppi_vs_uniform"] <=
        policy["all_fppi_vs_uniform_max_increase"],
        "added_precision": values["added_match_precision"] >= policy["added_match_precision_min"],
    }
    return {"decision": "SMALL_RETRIEVAL_LEARNABILITY_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_SUPPRESSOR_CONTRAST_RETRIEVAL_RULE",
            "checks": checks, "values": values,
            "scope": "Reused train-head-holdout diagnostic after cached ignore filtering; not official AP."}


def evaluate(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior, base = project_path(PRIOR), project_path(BASE)
    record, plan = verify_record(
        prior, "docs/experiments/candidate21_native_censoring_result_20261014.json")
    _, base_plan = verify_record(
        base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    if record["decision"]["decision"] != "CENSORING_RETRIEVAL_PILOT_ALLOWED":
        raise ValueError("Native-censoring screen did not pass.")
    stems = plan["images"]
    if stems != base_plan["image_selection"]["holdout"][:64]:
        raise ValueError("Fixed64 image list changed.")
    snapshots = {r["stem"]: r for r in base_plan["records"]["holdout"]}
    raw_path = base / "raw_views.jsonl"
    if file_hash(raw_path) != plan["prior_raw_views_sha256"]:
        raise ValueError("Frozen raw windows changed.")
    windows = {r["image"]: r["windows"] for r in
               (json.loads(line) for line in raw_path.read_text().splitlines())}
    frames = json.loads((prior / "frames.json").read_text())
    baseline_frames = json.loads((base / "frames.json").read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    paths = (REGISTRATION, "sdcl/postprocess/censored_retrieval.py",
             "sdcl/evaluation/censored_retrieval.py", "scripts/evaluation/diagnose_censored_retrieval.py",
             "sdcl/postprocess/rescue.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "candidate21_result_sha256": file_hash(project_path(
            "docs/experiments/candidate21_native_censoring_result_20261014.json")),
        "images": stems,
    })
    totals = {arm: {"groups": {g: dict.fromkeys(("gt", "tp50", "tp_ten"), 0) for g in GROUPS},
                    **dict.fromkeys(("fp50", "additions", "added_tp50", "added_duplicate_fp50",
                                     "added_other_fp50", "recovered50", "lost50"), 0)}
              for arm in ARMS}
    outputs, timings = {}, dict.fromkeys(("retrieval", "ordinary_dense_softnms"), 0.)
    for index, stem in enumerate(stems, 1):
        paths_and_hashes = ((prior / frames[stem]["file"], frames[stem]["sha256"]),
                            (base / baseline_frames[stem]["file"], baseline_frames[stem]["sha256"]))
        if any(file_hash(p) != h for p, h in paths_and_hashes):
            raise ValueError("Frozen pool or baseline changed.")
        with np.load(paths_and_hashes[0][0], allow_pickle=False) as saved:
            pools = {k: saved[k].copy() for k in saved.files}
        with np.load(paths_and_hashes[1][0], allow_pickle=False) as saved:
            baseline = saved["baseline"].copy()
        if windows[stem][0][:2] != [0, 0]:
            raise ValueError("Original fullimage window changed.")
        size = windows[stem][0][2:]
        stage = perf_counter()
        arms, inventory = retrieve(
            baseline, pools["dense"], pools["dense_view"], pools["uncapped"], pools["uncapped_view"])
        timings["retrieval"] += perf_counter() - stage
        stage = perf_counter()
        ordinary, _, _ = soft_nms_with_origins([pools["dense"]], size, [])
        timings["ordinary_dense_softnms"] += perf_counter() - stage
        arms["ordinary_dense_softnms"] = (ordinary, np.zeros(len(ordinary), bool))
        arms["uniform"] = (baseline, np.zeros(len(baseline), bool))
        # Candidate construction is complete before object labels are read.
        metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        if file_hash(metadata_path) != snapshots[stem]["metadata_sha256"]:
            raise ValueError("Frozen diagnostic annotations changed.")
        meta = json.loads(metadata_path.read_text())
        gt = np.asarray([o["box_xyxy"] for o in meta["valid_objects"]], float).reshape(-1, 4)
        classes = np.asarray([o["class_id"] for o in meta["valid_objects"]], int)
        areas = box_areas(gt) * (640 / max(size)) ** 2
        scored = {arm: score_image(*arms[arm], gt, classes, areas) for arm in ARMS}
        for arm, values in scored.items():
            recovered = (values["matched"] >= 0) & (scored["uniform"]["matched"] < 0)
            lost = (values["matched"] < 0) & (scored["uniform"]["matched"] >= 0)
            for group, counts in values["groups"].items():
                for key, value in counts.items():
                    totals[arm]["groups"][group][key] += value
            for key in ("fp50", "additions", "added_tp50", "added_duplicate_fp50", "added_other_fp50"):
                totals[arm][key] += values[key]
            totals[arm]["recovered50"] += int(recovered.sum())
            totals[arm]["lost50"] += int(lost.sum())
        target = output / "predictions" / f"{stem}.npz"
        np.savez_compressed(target, **{
            k: v for arm, (predictions, extra) in arms.items()
            for k, v in ((arm, predictions), (arm + "_is_extra", extra))})
        outputs[stem] = {"file": target.relative_to(output).as_posix(),
                         "sha256": file_hash(target), "inventory": inventory}
        if index == 1 or index % 16 == 0:
            print(f"Censored retrieval: {index}/64, {perf_counter()-started:.1f}s", flush=True)
    metrics = {}
    for arm, counts in totals.items():
        metrics[arm] = {k: v for k, v in counts.items() if k != "groups"}
        metrics[arm].update({
            **{g + "_recall_percent": c["tp50"] / max(c["gt"], 1) * 100
               for g, c in counts["groups"].items()},
            **{g + "_ten_threshold_recall_percent": c["tp_ten"] / max(c["gt"] * 10, 1) * 100
               for g, c in counts["groups"].items()},
            "all_fppi": counts["fp50"] / len(stems), "counts": counts["groups"],
        })
    verdict = decide(metrics, registration["policy"])
    save_json(output / "frames.json", outputs)
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": len(stems), "detector_forwards": 0, "fitting_steps": 0,
        "no_val_no_testdev": True, "no_threshold_search": True,
        "cache_has_diagnostic_ignore_filter": True,
        "prediction_construction_before_object_annotations": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "stage_seconds": timings, "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
