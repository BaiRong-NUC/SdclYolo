"""Bounded contextual class-choice test, matched against ordinary evidence."""

from copy import deepcopy
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.class_conflict_preflight import owned_oracle
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold, write_csv,
)
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.postprocess.class_choice import (
    CONTEXT_DIM, ClassChoiceScorer, choose_labels, make_features,
)
from sdcl.postprocess.class_conflict import apply_labels, class_evidence, vote_labels
from sdcl.postprocess.localization import (
    action_menu, apply_actions, calibrated_candidates, frozen_control, load_calibrator,
    normalized_actions, select_actions,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.rescue import archive_source, checked_data, save_checkpoint


REGISTRATION = "docs/experiments/candidate12_class_choice_registration_20261014.json"
OUTPUT = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("evidence_only", "observation_context")
GROUPS = ("all", "tiny", "small_all")


def pilot_decision(metrics, policy):
    primary = metrics["observation_context"]
    baseline, matched = metrics["uniform"], metrics["evidence_only"]
    values = {
        "small_all_recall_vs_uniform_pp":
            primary["small_all_recall_percent"] - baseline["small_all_recall_percent"],
        "small_all_recall_vs_evidence_only_pp":
            primary["small_all_recall_percent"] - matched["small_all_recall_percent"],
        "all_fppi_vs_uniform": primary["all_fppi"] - baseline["all_fppi"],
    }
    checks = {
        "baseline_gain": values["small_all_recall_vs_uniform_pp"] >=
        policy["context_small_all_recall_vs_uniform_min_pp"],
        "matched_context_gain": values["small_all_recall_vs_evidence_only_pp"] >
        policy["context_small_all_recall_vs_evidence_only_min_exclusive_pp"],
        "fp_budget": values["all_fppi_vs_uniform"] <=
        policy["context_all_fppi_vs_uniform_max_increase"],
    }
    return {"decision": "REPEAT_SMALL_CLASS_HEADS" if all(checks.values()) else
            "STOP_THIS_CONTEXT_CLASS_CHOICE_PILOT",
            "values": values, "checks": checks,
            "scope": "Training-image small-head holdout, not AP/AR or novelty."}


def prepare(output, registration, device):
    from ultralytics import YOLO
    from sdcl.training.trainer import check_version

    check_version()
    gate_path = project_path("output/analysis/class-conflict-headroom-fixed-20261014")
    gate_record = json.loads(project_path(
        "docs/experiments/candidate11_class_conflict_result_20261014.json").read_text())
    for name, digest in gate_record["artifacts_sha256"].items():
        if file_hash(gate_path / name) != digest:
            raise ValueError("Class-choice headroom evidence changed.")
    if gate_record["decision"]["decision"] != "SMALL_CLASS_CHOICE_PILOT_WORTH_TESTING":
        raise ValueError("No registered class-choice headroom; do not train.")
    dataset, checkpoint, all_records = checked_data()
    lookup = {r["stem"]: r for r in all_records}
    prior = project_path("output/analysis/visual-preflight-fixed-20261014")
    prior_plan = json.loads((prior / "plan.json").read_text())
    verify_dependencies(prior_plan["source_hashes"], prior / "source.zip")
    selected = prior_plan["image_selection"]
    records = {g: [lookup[s] for s in stems] for g, stems in selected.items()}
    if {r["stem"].split("_")[0] for r in records["train"]} & {
        r["stem"].split("_")[0] for r in records["holdout"]}:
        raise ValueError("Head training/holdout scene overlap.")
    heads, mean, std, provenance = frozen_heads()
    calibrator = load_calibrator(frozen_control())
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    paths = (REGISTRATION, "sdcl/postprocess/class_choice.py",
             "sdcl/training/class_choice_preflight.py",
             "scripts/training/run_class_choice_preflight.py",
             "sdcl/postprocess/class_conflict.py",
             "sdcl/evaluation/class_conflict_preflight.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "headroom_result_record_sha256": file_hash(project_path(
            "docs/experiments/candidate11_class_conflict_result_20261014.json")),
        "prior_image_selection_plan_sha256": file_hash(prior / "plan.json"),
        "image_selection": selected, "heads": provenance,
        "detector_sha256": file_hash(checkpoint),
        "records": {g: [{k: r[k] for k in ("stem", "image_sha256", "metadata_sha256")}
                        for r in rows] for g, rows in records.items()},
    })
    detector = YOLO(str(checkpoint))
    kwargs = dict(imgsz=960, conf=.001, iou=.7, max_det=500, quantize=16,
                  rect=False, batch=1, device=device, verbose=False,
                  save=False, augment=False, agnostic_nms=False)
    def predict(image):
        boxes = detector.predict(source=image, **kwargs)[0].boxes
        return torch.cat((boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]), 1).float().cpu().numpy()
    cv2.setNumThreads(1)
    predict(np.zeros((960, 960, 3), np.uint8))
    if not detector.predictor.model.fp16:
        raise ValueError("Frozen view policy requires effective FP16.")
    fitting, frames, forwards = [], {}, 0
    started = perf_counter()
    with torch.inference_mode(), (output / "raw_views.jsonl").open("x") as capture:
        for group, chosen in records.items():
            for index, record in enumerate(chosen, 1):
                stem = record["stem"]
                image = cv2.imread(record["image"])
                if image is None or [image.shape[1], image.shape[0]] != record["size"]:
                    raise ValueError("Training image dimensions changed.")
                windows = [[0, 0, *record["size"]], *tile_windows(*record["size"])]
                views = [predict(image)]
                for window in windows[1:]:
                    x1, y1, x2, y2 = window
                    views.append(map_tile_predictions(predict(image[y1:y2, x1:x2].copy()), window))
                forwards += 5
                capture.write(json.dumps({"image": stem, "windows": windows,
                                          "views": [v.tolist() for v in views]}) + "\n")
                c, origins, scores, base = calibrated_candidates(
                    views, windows, record["size"], [], calibrator)
                boxes, valid_box, geometry = action_menu(
                    c, origins, scores, base, views, record["size"], [])
                normalized = normalized_actions(geometry, mean, std)
                actions = select_actions(heads["uniform"](torch.from_numpy(
                    normalized)).numpy(), valid_box, "absolute")
                baseline = apply_actions(c, boxes, actions)
                valid, score, raw_score, context = class_evidence(
                    baseline, views, windows, record["size"])
                features = make_features(baseline, score, raw_score, context, record["size"])
                metadata_path = dataset / "metadata/train" / f"{stem}.json"
                if file_hash(metadata_path) != record["metadata_sha256"]:
                    raise ValueError("Training metadata changed.")
                metadata = json.loads(metadata_path.read_text())
                ignored = prediction_ignore_mask(torch.from_numpy(baseline[:, :4]),
                    torch.tensor(metadata["ignore_boxes_xyxy"], dtype=torch.float64).reshape(-1, 4))
                keep = ~ignored.numpy()
                baseline, valid, score, features = (
                    values[keep] for values in (baseline, valid, score, features))
                target, _ = owned_oracle(record, baseline, valid)
                voting = vote_labels(baseline, valid, score)
                selected_rows = (valid.sum(1) > 1) & (baseline[:, 4] >= .05)
                if group == "train":
                    fitting.append((features[selected_rows], valid[selected_rows], target[selected_rows],
                                    baseline[selected_rows, 4].astype(np.float32)))
                path = output / "frames" / f"{stem}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, features=features, valid=valid, targets=target,
                                        baseline=baseline, vote_labels=voting)
                frames[stem] = {"file": path.relative_to(output).as_posix(),
                                "sha256": file_hash(path), "group": group}
                if index == 1 or index % 64 == 0 or index == len(chosen):
                    print(f"Class-choice frames {group}: {index}/{len(chosen)}, "
                          f"{perf_counter() - started:.1f}s.", flush=True)
    fit = tuple(np.concatenate([r[i] for r in fitting]) for i in range(4))
    if len(fit[0]) < 100:
        raise ValueError("Too few conflicting fitting anchors for a bounded class pilot.")
    moments = fit[0][fit[1]].astype(np.float64)
    feature_mean = moments.mean(0).astype(np.float32)
    feature_std = np.maximum(moments.std(0), .05).astype(np.float32)
    save_json(output / "normalization.json", {"mean": feature_mean.tolist(), "std": feature_std.tolist()})
    with (output / "training_features.npz").open("xb") as stream:
        np.savez_compressed(stream, features=fit[0], valid=fit[1], targets=fit[2], weights=fit[3])
    save_json(output / "frames.json", frames)
    del detector
    torch.cuda.empty_cache()
    return fit, records, frames, {
        "detector_view_forwards_excluding_warmup": forwards,
        "preparation_seconds_excluding_input_checks": perf_counter() - started,
        "fitting_conflict_rows": len(fit[0]),
    }


def evaluate_heads(models, records, frames, output, mean, std, gpu):
    controls = ("uniform", "weighted_vote", *ARMS)
    totals = {a: {g: {"gt": 0, "tp": 0} for g in GROUPS} for a in controls}
    fp, changes = dict.fromkeys(controls, 0), dict.fromkeys(controls, 0)
    with torch.inference_mode():
        for record in records:
            with np.load(output / frames[record["stem"]]["file"], allow_pickle=False) as chunk:
                baseline, valid = chunk["baseline"], chunk["valid"]
                inputs = np.clip((chunk["features"] - mean) / std, -6, 6).astype(np.float32)
                voting = chunk["vote_labels"]
            predictions = {"uniform": baseline, "weighted_vote": apply_labels(baseline, voting)}
            x = torch.from_numpy(inputs).to(gpu)
            for arm, model in models.items():
                arm_x = x.clone()
                if arm == "evidence_only":
                    arm_x[:, :, -CONTEXT_DIM:] = 0
                labels = choose_labels(model(arm_x).cpu().numpy(), baseline, valid)
                predictions[arm] = apply_labels(baseline, labels)
            gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in record["objects"]], int)
            areas = box_areas(gt) * (640 / max(record["size"])) ** 2
            ious = box_iou_matrix(gt, baseline[:, :4])
            for arm, values in predictions.items():
                matches, _ = match_at_threshold(ious, classes, values, .25)
                fp[arm] += int((values[:, 4] >= .25).sum() - (matches >= 0).sum())
                changes[arm] += int((values[:, 5] != baseline[:, 5]).sum())
                for group in GROUPS:
                    mask = group_mask(areas, group)
                    totals[arm][group]["gt"] += int(mask.sum())
                    totals[arm][group]["tp"] += int((mask & (matches >= 0)).sum())
    return {
        a: {**{g + "_recall_percent": v["tp"] / max(v["gt"], 1) * 100
               for g, v in groups.items()}, "all_fppi": fp[a] / len(records),
            "changed_labels": changes[a], "gt_tp": groups}
        for a, groups in totals.items()
    }


def fit_seed(output, fit, records, frames, recipe, mean, std, gpu, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = deepcopy(ClassChoiceScorer().state_dict())
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = ClassChoiceScorer().to(gpu)
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(),
            lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    x = torch.from_numpy(np.clip((fit[0] - mean) / std, -6, 6).astype(np.float32)).to(gpu)
    blind = x.clone()
    blind[:, :, -CONTEXT_DIM:] = 0
    inputs = {"evidence_only": blind, "observation_context": x}
    valid = torch.from_numpy(fit[1]).to(gpu)
    target = torch.from_numpy(fit[2]).long().to(gpu)
    weight = torch.from_numpy(fit[3]).to(gpu)
    logs = []
    for epoch in range(recipe["epochs"]):
        order = torch.randperm(len(x),
            generator=torch.Generator().manual_seed(seed + epoch)).to(gpu)
        sums, total = dict.fromkeys(ARMS, 0.), 0.
        for start in range(0, len(order), recipe["batch"]):
            selected = order[start:start + recipe["batch"]]
            denominator = float(weight[selected].sum())
            total += denominator
            for arm in ARMS:
                optimizers[arm].zero_grad(set_to_none=True)
                result = models[arm](inputs[arm][selected]).masked_fill(~valid[selected], -torch.inf)
                per_row = F.cross_entropy(result, target[selected], reduction="none")
                loss = (per_row * weight[selected]).sum() / weight[selected].sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite class-choice loss.")
                loss.backward()
                optimizers[arm].step()
                sums[arm] += float(loss.detach()) * denominator
        row = {"seed": seed, "epoch": epoch + 1, **{a: sums[a] / total for a in ARMS}}
        logs.append(row)
        print(json.dumps(row), flush=True)
    directory = output / f"seed{seed}"
    directory.mkdir()
    save_checkpoint(directory / "last.pt", {
        "seed": seed, "epoch": recipe["epochs"], "recipe": recipe,
        "models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()}
                   for a, m in models.items()},
        "plan_sha256": file_hash(output / "plan.json"),
        "training_feature_sha256": file_hash(output / "training_features.npz"),
        "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
    })
    write_csv(directory / "losses.csv", logs)
    return evaluate_heads(models, records["holdout"], frames, output, mean, std, gpu)


def run(device="0", output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    if registration["arms"] != list(ARMS) or registration["recipe"]["epochs"] != 8:
        raise ValueError("Registered class-choice recipe changed.")
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output, gpu = project_path(output), torch.device(f"cuda:{device}")
    fit, records, frames, preparation = prepare(output, registration, device)
    normalization = json.loads((output / "normalization.json").read_text())
    mean, std = (np.array(normalization[k], np.float32) for k in ("mean", "std"))
    metrics, decisions = {}, {}
    for seed in registration["recipe"]["seeds"]:
        if seed and decisions["0"]["decision"] != "REPEAT_SMALL_CLASS_HEADS":
            break
        metrics[str(seed)] = fit_seed(
            output, fit, records, frames, registration["recipe"], mean, std, gpu, seed)
        decisions[str(seed)] = pilot_decision(metrics[str(seed)], registration["policy"])
        save_json(output / f"seed{seed}" / "metrics.json", metrics[str(seed)])
        save_json(output / f"seed{seed}" / "decision.json", decisions[str(seed)])
    verdict = {
        "decision": decisions["0"]["decision"], "per_seed": decisions,
        "head_seeds_completed": list(metrics), "detector_training_ran": False,
        "official_evaluation_ran": False,
    }
    if len(metrics) == 3:
        values = [d["values"] for d in decisions.values()]
        baseline_gain = float(np.mean([v["small_all_recall_vs_uniform_pp"] for v in values]))
        matched_gain = float(np.mean([v["small_all_recall_vs_evidence_only_pp"] for v in values]))
        fp = float(np.mean([v["all_fppi_vs_uniform"] for v in values]))
        positive = sum(v["small_all_recall_vs_evidence_only_pp"] > 0 for v in values)
        passed = (baseline_gain >= registration["policy"][
            "repeat_mean_small_all_recall_vs_uniform_min_pp"] and matched_gain > 0 and
            positive >= registration["policy"]["repeat_positive_vs_evidence_only_seeds_min"] and
            fp <= registration["policy"]["context_all_fppi_vs_uniform_max_increase"])
        verdict.update(
            decision="CONTEXT_CLASS_CHOICE_WORTH_OFFICIAL_SCREEN" if passed else
            "NO_REPEATABLE_CONTEXT_CLASS_ADVANTAGE",
            mean_small_all_recall_vs_uniform_pp=baseline_gain,
            mean_small_all_recall_vs_evidence_only_pp=matched_gain,
            mean_all_fppi_increase=fp, positive_vs_evidence_only_seeds=positive,
        )
    plan = json.loads((output / "plan.json").read_text())
    if not all(file_hash(project_path(p)) == h for p, h in plan["source_hashes"].items()):
        raise ValueError("Class-choice source changed.")
    if any(file_hash(output / c["file"]) != c["sha256"] for c in frames.values()):
        raise ValueError("Prepared class-choice frame changed.")
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        **preparation, "source_unchanged": True, "scene_overlap": 0,
        "fitting_images": len(records["train"]), "head_holdout_images": len(records["holdout"]),
        "nominal_parameters_per_head": sum(p.numel() for p in ClassChoiceScorer().parameters()),
        "total_seconds": perf_counter() - started,
        "scope": "Small-head train-scene holdout, same historically trained detector; not AP/AR.",
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics,
            "elapsed_seconds": perf_counter() - started}
