"""Matched cached pilot of stable pair ordering versus count-controlled thinning."""

from copy import deepcopy
import hashlib
from itertools import combinations
import json
import os
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.action_order_sensitivity import OUTPUT as PRIOR
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, group_mask, match_at_threshold, write_csv,
)
from sdcl.postprocess.localization import ActionScorer, apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source, save_checkpoint


REGISTRATION = "docs/experiments/candidate19_stable_order_pilot_registration_20261014.json"
OUTPUT = "output/analysis/stable-order-pilot-fixed-20261014"
ARMS = ("nominal_order", "random_thinned_order", "stable_order")
PAIRS = np.array(list(combinations(range(6), 2)))
THRESHOLDS = np.linspace(.5, .95, 10)
GROUPS = ("all", "tiny", "small_all")


def pair_masks(ious, valid, eligible, seed):
    delta = ious[:, :, PAIRS[:, 0]] - ious[:, :, PAIRS[:, 1]]
    signs = np.sign(delta[:, 0])
    nominal = (valid[:, PAIRS[:, 0]] & valid[:, PAIRS[:, 1]] &
               (np.abs(delta[:, 0]) > .01) & eligible[:, None])
    stable = nominal & ((delta * signs[:, None]).min(1) > .01)
    thinned = np.zeros_like(stable)
    generator = np.random.default_rng(seed)
    for row in range(len(nominal)):
        count = int(stable[row].sum())
        if count:
            thinned[row, generator.choice(np.flatnonzero(nominal[row]), count, replace=False)] = True
    if not np.array_equal(stable.sum(1), thinned.sum(1)):
        raise ValueError("Per-anchor count control differs.")
    return signs.astype(np.float32), {
        "nominal_order": nominal, "random_thinned_order": thinned, "stable_order": stable,
    }


def objective(outputs, signs, masks, valid, owned, active):
    pairs = torch.as_tensor(PAIRS, device=outputs.device)
    difference = outputs[:, pairs[:, 0]] - outputs[:, pairs[:, 1]]
    pair_loss = F.softplus(-signs * difference / .05)
    positive = (pair_loss * masks).sum(1) / masks.sum(1).clamp_min(1)
    negative_loss = F.softplus((outputs[:, 1:] - outputs[:, :1]) / .05)
    negative = (negative_loss * valid[:, 1:]).sum(1) / valid[:, 1:].sum(1).clamp_min(1)
    values = torch.where(owned, positive, negative)
    return (values * active).sum() / active.sum().clamp_min(1)


def decide(metrics, policy):
    local, base = metrics["stable_order"], metrics["uniform"]
    key = "small_all_ten_threshold_recall_percent"
    values = {
        "vs_uniform_pp": local[key] - base[key],
        "vs_nominal_pp": local[key] - metrics["nominal_order"][key],
        "vs_thinned_pp": local[key] - metrics["random_thinned_order"][key],
        "all_fppi_increase": local["all_fppi_at50"] - base["all_fppi_at50"],
    }
    checks = {
        "baseline_gain": values["vs_uniform_pp"] >=
        policy["small_all_ten_threshold_recall_vs_uniform_min_pp"],
        "ordinary_objective_gain": values["vs_nominal_pp"] >
        policy["small_all_ten_threshold_recall_vs_nominal_min_exclusive_pp"],
        "count_controlled_gain": values["vs_thinned_pp"] >
        policy["small_all_ten_threshold_recall_vs_thinned_min_exclusive_pp"],
        "fp_budget": values["all_fppi_increase"] <=
        policy["all_fppi_at50_vs_uniform_max_increase"],
    }
    return {"decision": "REPEAT_MATCHED_STABLE_ORDER_HEADS" if all(checks.values()) else
            "STOP_THIS_STABLE_ORDER_PILOT", "checks": checks, "values": values,
            "scope": "Cached tiny-head diagnostic, not official AP/AR or innovation confirmation."}


def load_inputs(output, registration):
    prior = project_path(PRIOR)
    gate, plan = verify_record(
        prior, "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")
    if gate["decision"]["decision"] != "MATCHED_STABLE_ORDER_HEAD_PILOT_ALLOWED":
        raise ValueError("No permission from the registered nuisance diagnostic.")
    frames = json.loads((prior / "frames.json").read_text())
    heads, _, _, provenance = frozen_heads()
    if provenance != plan["heads"]:
        raise ValueError("Frozen head provenance differs.")
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/training/stable_order_pilot.py",
             "scripts/training/run_stable_order_pilot.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")),
        "prior_plan_sha256": file_hash(prior / "plan.json"),
        "prior_frames_manifest_sha256": file_hash(prior / "frames.json"),
        "image_selection": plan["image_selection"], "heads": provenance,
    })
    fitting, holdout = [], []
    for group, stems in plan["image_selection"].items():
        for stem in stems:
            item = frames[stem]
            path = prior / item["file"]
            metadata = project_path(f"data/visdrone/metadata/train/{stem}.json")
            if file_hash(path) != item["sha256"] or file_hash(metadata) != item["metadata_sha256"]:
                raise ValueError("Pinned candidate18 frame/metadata changed.")
            with np.load(path, allow_pickle=False) as saved:
                frame = {k: saved[k] for k in saved.files}
            frame["stem"] = stem
            frame["metadata"] = json.loads(metadata.read_text())
            (fitting if group == "train" else holdout).append(frame)
    return fitting, holdout, heads["uniform"], hashes


def evaluate_heads(models, frames, gpu):
    totals = {a: {g: {"gt": 0, "tp50": 0, "tp_thresholds": 0} for g in GROUPS}
              for a in ("uniform", *ARMS)}
    fp, moves, owned_values = {}, {}, {}
    for arm in totals:
        fp[arm], moves[arm], owned_values[arm] = 0, 0, []
    with torch.inference_mode():
        for frame in frames:
            c, boxes, valid, uniform = (frame[k] for k in ("candidates", "boxes", "valid", "uniform"))
            x = torch.from_numpy(frame["features"]).to(gpu)
            selected = {"uniform": uniform}
            for arm, model in models.items():
                selected[arm] = select_actions(model(x).cpu().numpy(), valid, "absolute")
            gt = np.asarray([o["box_xyxy"] for o in frame["metadata"]["valid_objects"]]).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in frame["metadata"]["valid_objects"]], int)
            areas = box_areas(gt) * (640 / max(frame["metadata"]["original_size"])) ** 2
            for arm, actions in selected.items():
                predictions = apply_actions(c, boxes, actions)
                iou = box_iou_matrix(gt, predictions[:, :4])
                matches = [match_at_threshold(iou, classes, predictions, .25, t)[0] >= 0
                           for t in THRESHOLDS]
                fp[arm] += int((predictions[:, 4] >= .25).sum() - matches[0].sum())
                moves[arm] += int((actions != uniform).sum())
                for group in GROUPS:
                    include = group_mask(areas, group)
                    totals[arm][group]["gt"] += int(include.sum())
                    totals[arm][group]["tp50"] += int((include & matches[0]).sum())
                    totals[arm][group]["tp_thresholds"] += int(
                        np.asarray(matches)[:, include].sum())
                owned_mask = frame["eligible"] & group_mask(frame["areas"], "small_all")
                row = np.flatnonzero(owned_mask)
                target = frame["ious"][row, 0, actions[row]]
                base = frame["ious"][row, 0, uniform[row]]
                owned_values[arm].append(np.column_stack((target, base)))
    metrics = {}
    for arm, groups in totals.items():
        owned = np.concatenate(owned_values[arm])
        metrics[arm] = {
            **{g + "_ten_threshold_recall_percent": v["tp_thresholds"] / max(v["gt"] * 10, 1) * 100
               for g, v in groups.items()},
            **{g + "_recall_at50_percent": v["tp50"] / max(v["gt"], 1) * 100
               for g, v in groups.items()},
            "all_fppi_at50": fp[arm] / len(frames), "moves_vs_uniform": moves[arm],
            "small_all_owned_mean_iou_percent": float(owned[:, 0].mean() * 100),
            "small_all_owned_harm_vs_uniform_percent": float(
                (owned[:, 0] < owned[:, 1] - .01).mean() * 100),
            "gt_tp": groups,
        }
    return metrics


def fit_seed(output, fitting, holdout, frozen, recipe, seed, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fitting_arrays = {k: [] for k in ("features", "valid", "signs", "owned", "active", *ARMS)}
    for frame in fitting:
        hashed = int.from_bytes(hashlib.sha256(
            f"stable-order:{seed}:{frame['stem']}".encode()).digest()[:8], "little")
        signs, masks = pair_masks(frame["ious"], frame["valid"], frame["eligible"], hashed)
        owned = frame["owner"] >= 0
        active = ((frame["candidates"][:, 4] >= .05) & (frame["valid"].sum(1) >= 2) &
                  (~owned | frame["eligible"]))
        for key, value in {
                "features": frame["features"], "valid": frame["valid"], "signs": signs,
                "owned": owned, "active": active, **masks}.items():
            fitting_arrays[key].append(value)
    values = {k: np.concatenate(v) for k, v in fitting_arrays.items()}
    counts = {a: {"pairs": int(values[a].sum()),
                  "anchors_with_pairs": int(values[a].any(1).sum())} for a in ARMS}
    if not np.array_equal(values["stable_order"].sum(1), values["random_thinned_order"].sum(1)):
        raise ValueError("Training masks lose the matched per-anchor count control.")
    rows = values["active"]
    tensors = {k: torch.from_numpy(v[rows]).to(gpu) for k, v in values.items()}
    initial = deepcopy(frozen.state_dict())
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = ActionScorer().to(gpu)
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(
            models[arm].parameters(), lr=recipe["learning_rate"],
            weight_decay=recipe["weight_decay"])
    losses = []
    for epoch in range(recipe["epochs"]):
        order = torch.randperm(len(tensors["active"]),
            generator=torch.Generator().manual_seed(seed * 100 + epoch)).to(gpu)
        sums = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), recipe["batch"]):
            index = order[start:start + recipe["batch"]]
            for arm, model in models.items():
                optimizers[arm].zero_grad(set_to_none=True)
                loss = objective(model(tensors["features"][index]), tensors["signs"][index],
                    tensors[arm][index], tensors["valid"][index], tensors["owned"][index],
                    tensors["active"][index])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; do not tune the recipe.")
                loss.backward()
                optimizers[arm].step()
                sums[arm] += float(loss.detach()) * len(index)
        row = {"seed": seed, "epoch": epoch + 1,
               **{k: v / max(len(order), 1) for k, v in sums.items()}}
        print(json.dumps(row), flush=True)
        losses.append(row)
    seed_path = output / f"seed{seed}"
    seed_path.mkdir()
    save_checkpoint(seed_path / "last.pt", {
        "seed": seed, "epoch": recipe["epochs"], "recipe": recipe,
        "models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()}
                   for a, m in models.items()}, "plan_sha256": file_hash(output / "plan.json"),
    })
    write_csv(seed_path / "losses.csv", losses)
    save_json(seed_path / "comparison_counts.json", {
        "counts": counts, "active_anchors": int(rows.sum()),
        "active_unowned": int((rows & ~values["owned"]).sum()),
        "stable_random_count_per_anchor_exact": True,
    })
    metrics = evaluate_heads(models, holdout, gpu)
    save_json(seed_path / "metrics.json", metrics)
    return metrics


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    fitting, holdout, frozen, hashes = load_inputs(output, registration)
    gpu = torch.device(f"cuda:{device}")
    per_seed, verdict = {}, None
    for seed in registration["recipe"]["seeds"]:
        metrics = fit_seed(output, fitting, holdout, frozen, registration["recipe"], seed, gpu)
        check = decide(metrics, registration["policy"])
        per_seed[str(seed)] = {"metrics": metrics, "decision": check}
        save_json(output / f"seed{seed}/decision.json", check)
        if seed == 0 and check["decision"] == "STOP_THIS_STABLE_ORDER_PILOT":
            verdict = check
            break
    if verdict is None:
        comparisons = [r["decision"]["values"] for r in per_seed.values()]
        gain = float(np.mean([c["vs_uniform_pp"] for c in comparisons]))
        positives = sum(c["vs_nominal_pp"] > 0 and c["vs_thinned_pp"] > 0 for c in comparisons)
        checks = {
            "mean_baseline_gain": gain >= registration["policy"][
                "repeat_mean_small_all_ten_threshold_recall_vs_uniform_min_pp"],
            "positive_matched_seeds": positives >= registration["policy"][
                "repeat_positive_vs_both_matched_seeds_min"],
            "all_seed_fp_budget": all(c["all_fppi_increase"] <= registration["policy"][
                "all_fppi_at50_vs_uniform_max_increase"] for c in comparisons),
        }
        verdict = {"decision": "BOUNDED_OFFICIAL_AUDIT_ALLOWED" if all(checks.values()) else
                   "STOP_THIS_STABLE_ORDER_REPEAT", "checks": checks,
                   "values": {"mean_vs_uniform_pp": gain, "positive_matched_seeds": positives}}
    save_json(output / "seed_results.json", per_seed)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "fitting_images": len(fitting), "head_holdout_images": len(holdout),
        "head_parameters_each": sum(p.numel() for p in frozen.parameters()),
        "seeds_completed": list(per_seed), "epochs_each": registration["recipe"]["epochs"],
        "detector_forwards": 0, "same_pretrained_initial_weights": True,
        "no_testdev": True, "final_epoch_only": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "seed0": per_seed["0"]}
