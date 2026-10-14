"""Small matched intervention heads; expand only after pre-registered gates."""

from copy import deepcopy
import json
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.detection_soft_evidence import native_view, posterior_evidence
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold, write_csv,
)
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.class_conflict import apply_labels, class_evidence
from sdcl.postprocess.marginal_class_intervention import (
    LOCAL_DIM, InterventionScorer, choose, features, marginal_targets,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.rescue import archive_source, checked_data, save_checkpoint


REGISTRATION = "docs/experiments/candidate16_marginal_class_intervention_registration_20261014.json"
OUTPUT = "output/analysis/marginal-class-intervention-fixed-20261014"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
SOFT = "output/analysis/detection-soft-evidence-fixed-20261014"
ARMS = ("class_ce", "marginal_blind", "marginal_local")
GROUPS = ("all", "tiny", "small_all")


def decision(metrics, policy):
    local = metrics["marginal_local"]
    base = metrics["uniform"]
    gain = local["small_all_recall_percent"] - base["small_all_recall_percent"]
    ce = local["small_all_recall_percent"] - metrics["class_ce"]["small_all_recall_percent"]
    blind = local["small_all_recall_percent"] - metrics["marginal_blind"]["small_all_recall_percent"]
    fp = local["all_fppi"] - base["all_fppi"]
    checks = {
        "baseline_gain": gain >= policy["local_small_all_recall_vs_uniform_min_pp"],
        "matched_class_objective_gain": ce > policy["local_small_all_recall_vs_class_ce_min_exclusive_pp"],
        "matched_local_features_gain": blind > policy["local_small_all_recall_vs_marginal_blind_min_exclusive_pp"],
        "fp_budget": fp <= policy["local_all_fppi_vs_uniform_max_increase"],
    }
    return {"decision": "REPEAT_TINY_INTERVENTION_HEADS" if all(checks.values()) else
            "STOP_THIS_MARGINAL_INTERVENTION_PILOT", "checks": checks,
            "values": {"small_all_vs_uniform_pp": gain, "small_all_vs_class_ce_pp": ce,
                       "small_all_vs_blind_pp": blind, "all_fppi_increase": fp}}


def verify_record(path, record_path):
    record = json.loads(project_path(record_path).read_text())
    for name, digest in record["artifacts_sha256"].items():
        if file_hash(path / name) != digest:
            raise ValueError("Pinned historical result/input changed.")
    plan = json.loads((path / "plan.json").read_text())
    verify_dependencies(plan["source_hashes"], path / "source.zip")
    return record, plan


def prepare(output, registration, device):
    prior, soft = project_path(PRIOR), project_path(SOFT)
    _, prior_plan = verify_record(prior, "docs/experiments/candidate12_class_choice_result_20261014.json")
    gate, _ = verify_record(soft, "docs/experiments/candidate15_detection_soft_evidence_result_20261014.json")
    if gate["decision"]["decision"] != "TINY_SOFT_INTERVENTION_PILOT_ALLOWED":
        raise ValueError("Real detection soft-evidence screen has not passed.")
    _, checkpoint, all_records = checked_data()
    lookup = {r["stem"]: r for r in all_records}
    selected = {
        "train": prior_plan["image_selection"]["train"][:128],
        "holdout": prior_plan["image_selection"]["holdout"],
    }
    records = {g: [lookup[s] for s in stems] for g, stems in selected.items()}
    if (any(len(rows) != 128 for rows in records.values()) or
            {s.split("_")[0] for s in selected["train"]} &
            {s.split("_")[0] for s in selected["holdout"]}):
        raise ValueError("Fixed intervention scene split changed.")
    raw = {}
    with (prior / "raw_views.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            raw[item["image"]] = item
    frames = json.loads((prior / "frames.json").read_text())
    saved_soft = json.loads((soft / "frames.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    paths = (REGISTRATION, "sdcl/postprocess/marginal_class_intervention.py",
             "sdcl/training/marginal_class_intervention.py",
             "scripts/training/run_marginal_class_intervention.py",
             "sdcl/evaluation/detection_soft_evidence.py",
             "sdcl/training/class_margin_probe.py", "sdcl/postprocess/class_conflict.py",
             "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "candidate12_plan_sha256": file_hash(prior / "plan.json"),
        "candidate15_plan_sha256": file_hash(soft / "plan.json"),
        "detector_sha256": file_hash(checkpoint), "image_selection": selected,
    })
    extractor = FrozenViews(device)
    fitting, manifest = [], {}
    started = perf_counter()
    try:
        with torch.inference_mode():
            for group, rows in records.items():
                for index, record in enumerate(rows, 1):
                    stem = record["stem"]
                    with np.load(prior / frames[stem]["file"], allow_pickle=False) as c:
                        baseline = c["baseline"].copy()
                        labels = c["targets"].copy()
                    if group == "holdout":
                        with np.load(soft / saved_soft[stem]["file"], allow_pickle=False) as c:
                            valid, hard, posterior = c["soft_valid"], c["hard_score"], c["soft_score"]
                            if not np.array_equal(baseline, c["baseline"]):
                                raise ValueError("Saved holdout baseline differs.")
                        targets = np.zeros(valid.shape, np.float32)
                    else:
                        if file_hash(record["image"]) != record["image_sha256"]:
                            raise ValueError("Training image changed.")
                        image = cv2.imread(record["image"])
                        if image is None or list(image.shape[1::-1]) != record["size"]:
                            raise ValueError("Invalid image.")
                        windows = raw[stem]["windows"]
                        views, scores = [], []
                        for view, (x1, y1, x2, y2) in enumerate(windows):
                            boxes, score = native_view(extractor, image[y1:y2, x1:x2].copy())
                            if view:
                                boxes = map_tile_predictions(boxes, windows[view])
                            expected = np.asarray(raw[stem]["views"][view], np.float32).reshape(-1, 6)
                            if not np.array_equal(boxes, expected):
                                raise ValueError("Frozen native fitting replay differs.")
                            views.append(boxes)
                            scores.append(score)
                        hard_valid, hard, _, _ = class_evidence(baseline, views, windows, record["size"])
                        valid, posterior = posterior_evidence(baseline, views, scores, hard_valid)
                        gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
                        classes = np.array([o["class_id"] for o in record["objects"]], int)
                        if file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) != record["metadata_sha256"]:
                            raise ValueError("Fitting labels changed.")
                        targets = marginal_targets(box_iou_matrix(gt, baseline[:, :4]),
                                                   classes, baseline, valid)
                        # Ownership stays fixed; expose newly supported GT labels from saved native audit.
                        geometry = baseline.copy()
                        geometry[:, 5] = 0
                        owned, _ = match_at_threshold(box_iou_matrix(gt, baseline[:, :4]),
                                                     np.zeros(len(classes), int), geometry, .001)
                        labels = baseline[:, 5].astype(int)
                        for gt_row in np.flatnonzero(owned >= 0):
                            dt_row = owned[gt_row]
                            if valid[dt_row, classes[gt_row]]:
                                labels[dt_row] = classes[gt_row]
                    x = features(baseline, hard, posterior, record["size"])
                    mask = (baseline[:, 4] >= .25) & (valid.sum(1) > 1)
                    if group == "train":
                        fitting.append((x[mask], valid[mask], labels[mask],
                                        targets[mask], baseline[mask, 4], baseline[mask, 5].astype(int)))
                    target = output / "frames" / f"{stem}.npz"
                    np.savez_compressed(target, baseline=baseline, valid=valid, features=x)
                    manifest[stem] = {"group": group, "file": target.relative_to(output).as_posix(),
                                      "sha256": file_hash(target)}
                    if index == 1 or index % 32 == 0:
                        print(f"Intervention prep {group}: {index}/128; {perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    fit = tuple(np.concatenate([chunk[i] for chunk in fitting]) for i in range(6))
    supported = fit[0][fit[1]]
    mean, std = supported.mean(0), np.maximum(supported.std(0), .05)
    save_json(output / "normalization.json", {"mean": mean.tolist(), "std": std.tolist()})
    save_json(output / "frames.json", manifest)
    np.savez_compressed(output / "training.npz", x=fit[0], valid=fit[1], label=fit[2],
                        gain=fit[3], weight=fit[4], original=fit[5])
    active_gain = fit[3][fit[1]]
    preparation = {"fitting_conflict_rows": len(fit[0]), "positive_action_targets": int((active_gain > 0).sum()),
                   "negative_action_targets": int((active_gain < 0).sum()),
                   "zero_action_targets": int((active_gain == 0).sum()),
                   "detector_view_forwards_excluding_warmup": 640,
                   "preparation_seconds": perf_counter() - started}
    return fit, records, manifest, mean, std, preparation


def evaluate_heads(models, records, frames, output, mean, std):
    arms = ("uniform", *ARMS)
    totals = {a: {g: {"gt": 0, "tp": 0} for g in GROUPS} for a in arms}
    fp, changes = dict.fromkeys(arms, 0), dict.fromkeys(arms, 0)
    with torch.inference_mode():
        for record in records:
            with np.load(output / frames[record["stem"]]["file"], allow_pickle=False) as c:
                baseline, valid = c["baseline"], c["valid"]
                x = torch.from_numpy(np.clip((c["features"] - mean) / std, -6, 6).astype(np.float32))
            predictions = {"uniform": baseline}
            for arm, model in models.items():
                inputs = x.clone()
                if arm == "marginal_blind":
                    inputs[:, :, -LOCAL_DIM:] = 0
                predictions[arm] = apply_labels(baseline, choose(model(inputs).numpy(), baseline, valid, arm))
            gt = np.array([o["box_xyxy"] for o in record["objects"]]).reshape(-1, 4)
            cls = np.array([o["class_id"] for o in record["objects"]], int)
            iou = box_iou_matrix(gt, baseline[:, :4])
            areas = box_areas(gt) * (640 / max(record["size"])) ** 2
            for arm, values in predictions.items():
                matched, _ = match_at_threshold(iou, cls, values, .25)
                fp[arm] += int((values[:, 4] >= .25).sum() - (matched >= 0).sum())
                changes[arm] += int((values[:, 5] != baseline[:, 5]).sum())
                for group in GROUPS:
                    mask = group_mask(areas, group)
                    totals[arm][group]["gt"] += int(mask.sum())
                    totals[arm][group]["tp"] += int((mask & (matched >= 0)).sum())
    return {
        a: {**{g + "_recall_percent": v["tp"] / max(v["gt"], 1) * 100
               for g, v in groups.items()}, "all_fppi": fp[a] / len(records),
            "changed_labels_at025": changes[a], "gt_tp": groups}
        for a, groups in totals.items()
    }


def fit_seed(fit, records, frames, output, mean, std, recipe, seed):
    torch.manual_seed(seed)
    initial = deepcopy(InterventionScorer().state_dict())
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = InterventionScorer()
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(),
            lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    x = torch.from_numpy(np.clip((fit[0] - mean) / std, -6, 6).astype(np.float32))
    valid, label, gain, weight, original = (
        torch.from_numpy(v) for v in fit[1:])
    blind = x.clone()
    blind[:, :, -LOCAL_DIM:] = 0
    inputs = {"class_ce": x, "marginal_blind": blind, "marginal_local": x}
    logs = []
    for epoch in range(recipe["epochs"]):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))
        sums = dict.fromkeys(ARMS, 0.)
        batches = 0
        for start in range(0, len(x), recipe["batch"]):
            selected = order[start:start + recipe["batch"]]
            for arm, model in models.items():
                optimizers[arm].zero_grad(set_to_none=True)
                value = model(inputs[arm][selected])
                if arm == "class_ce":
                    per_row = F.cross_entropy(value.masked_fill(~valid[selected], -torch.inf), label[selected])
                    loss = (per_row * weight[selected]).sum() / weight[selected].sum()
                else:
                    keep = value.gather(1, original[selected, None])
                    predicted = value - keep
                    target = gain[selected]
                    per_action = F.smooth_l1_loss(predicted, target, reduction="none")
                    scale = torch.where(target != 0, recipe["nonzero_gain_loss_weight"], 1.)
                    weighted = scale * weight[selected, None] * valid[selected]
                    loss = (per_action * weighted).sum() / weighted.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite intervention loss.")
                loss.backward()
                optimizers[arm].step()
                sums[arm] += float(loss.detach())
            batches += 1
        row = {"seed": seed, "epoch": epoch + 1, **{a: sums[a] / batches for a in ARMS}}
        logs.append(row)
        print(json.dumps(row), flush=True)
    directory = output / f"seed{seed}"
    directory.mkdir()
    save_checkpoint(directory / "last.pt", {
        "seed": seed, "epoch": recipe["epochs"], "recipe": recipe,
        "plan_sha256": file_hash(output / "plan.json"),
        "training_sha256": file_hash(output / "training.npz"),
        "models": {a: m.state_dict() for a, m in models.items()},
        "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
    })
    write_csv(directory / "losses.csv", logs)
    for model in models.values():
        model.eval()
    return evaluate_heads(models, records["holdout"], frames, output, mean, std)


def run(device="0", output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = project_path(output)
    fit, records, frames, mean, std, preparation = prepare(output, registration, device)
    metrics, decisions = {}, {}
    for seed in registration["recipe"]["seeds"]:
        if seed and decisions["0"]["decision"] != "REPEAT_TINY_INTERVENTION_HEADS":
            break
        metrics[str(seed)] = fit_seed(fit, records, frames, output, mean, std, registration["recipe"], seed)
        decisions[str(seed)] = decision(metrics[str(seed)], registration["policy"])
        save_json(output / f"seed{seed}/metrics.json", metrics[str(seed)])
        save_json(output / f"seed{seed}/decision.json", decisions[str(seed)])
    verdict = {"decision": decisions["0"]["decision"], "per_seed": decisions,
               "head_seeds_completed": list(metrics)}
    if len(metrics) == 3:
        values = [v["values"] for v in decisions.values()]
        mean_gain = float(np.mean([v["small_all_vs_uniform_pp"] for v in values]))
        mean_ce = float(np.mean([v["small_all_vs_class_ce_pp"] for v in values]))
        mean_blind = float(np.mean([v["small_all_vs_blind_pp"] for v in values]))
        mean_fp = float(np.mean([v["all_fppi_increase"] for v in values]))
        positive = sum(v["small_all_vs_class_ce_pp"] > 0 and v["small_all_vs_blind_pp"] > 0 for v in values)
        passed = (mean_gain >= registration["policy"]["local_small_all_recall_vs_uniform_min_pp"]
                  and mean_ce > 0 and mean_blind > 0
                  and mean_fp <= registration["policy"]["local_all_fppi_vs_uniform_max_increase"]
                  and positive >= registration["policy"]["repeat_positive_vs_both_matched_seeds_min"])
        verdict.update(decision="INTERVENTION_WORTH_OFFICIAL_SCREEN" if passed else "NO_REPEATABLE_INTERVENTION_ADVANTAGE",
                       mean_small_all_vs_uniform_pp=mean_gain, mean_small_all_vs_ce_pp=mean_ce,
                       mean_small_all_vs_blind_pp=mean_blind, mean_all_fppi_increase=mean_fp,
                       positive_vs_both_seeds=positive)
    plan = json.loads((output / "plan.json").read_text())
    if (not all(file_hash(project_path(p)) == h for p, h in plan["source_hashes"].items())
            or any(file_hash(output / f["file"]) != f["sha256"] for f in frames.values())):
        raise ValueError("Intervention source/frame changed during pilot.")
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        **preparation, "source_unchanged": True, "fitting_images": 128,
        "head_holdout_images": 128, "scene_overlap": 0, "no_detector_training": True,
        "no_official_evaluation": True, "nominal_parameters_per_head":
            sum(p.numel() for p in InterventionScorer().parameters()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
