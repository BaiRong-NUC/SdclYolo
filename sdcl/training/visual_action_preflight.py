"""Bounded train-scene-held-out pilot for observable visual coordinate evidence."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.postprocess.localization import ActionScorer, normalized_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.postprocess.visual_action_features import VISUAL_DIM, recover_cached_boxes, visual_features
from sdcl.training.localization import DATA, regression_loss
from sdcl.training.regret_probe import checked_inputs, ranking_loss
from sdcl.training.rescue import archive_source, save_checkpoint


REGISTRATION = "docs/experiments/candidate9_visual_preflight_registration_20261014.json"
OUTPUT = "output/analysis/visual-preflight-fixed-20261014"
ARMS = ("geometry_absolute", "geometry_ranking", "visual_absolute", "visual_ranking")
THRESHOLDS = np.linspace(.5, .95, 10)


class PilotScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(68 + VISUAL_DIM, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def pilot_metrics(outputs, valid, targets, weights):
    selected = select_actions(outputs, valid, "absolute")
    chosen = targets[np.arange(len(targets)), selected]
    keep = targets[:, 0]
    owned = keep > 0
    denominator = weights[owned].sum()
    negative_denominator = weights[~owned].sum()
    if denominator <= 0:
        raise ValueError("No owned anchors in pilot holdout.")
    def average(values):
        return float(np.sum(values[owned] * weights[owned]) / denominator)
    hits = (chosen[:, None] >= THRESHOLDS).mean(1)
    oracle = np.max(np.where(valid, targets, -1.), axis=1)
    return {
        "owned_mean_iou_percent": average(chosen) * 100,
        "owned_ten_threshold_percent": average(hits) * 100,
        "owned_harm_rate_percent": average(chosen < keep - .01) * 100,
        "owned_move_rate_percent": average(selected != 0) * 100,
        "unowned_move_rate_percent": float(
            ((selected[~owned] != 0) * weights[~owned]).sum() /
            max(negative_denominator, 1e-12) * 100),
        "owned_oracle_regret_iou_pp": average(oracle - chosen) * 100,
    }


def pilot_decision(metrics, policy):
    comparison = {
        f"{candidate}_vs_{control}": {
            metric + "_pp": metrics[candidate][metric] - metrics[control][metric]
            for metric in metrics[candidate]
        }
        for candidate, control in (
            ("visual_ranking", "geometry_ranking"),
            ("visual_absolute", "geometry_absolute"),
            ("visual_ranking", "visual_absolute"),
            ("visual_ranking", "frozen_uniform"),
        )
    }
    primary = comparison["visual_ranking_vs_geometry_ranking"]
    checks = {
        "threshold_gain": primary["owned_ten_threshold_percent_pp"] >=
        policy["visual_ranking_vs_geometry_ranking_owned_ten_threshold_min_pp"],
        "mean_iou_gain": primary["owned_mean_iou_percent_pp"] >
        policy["visual_ranking_vs_geometry_ranking_owned_mean_iou_min_exclusive_pp"],
        "harm_budget": primary["owned_harm_rate_percent_pp"] <=
        policy["visual_ranking_vs_geometry_ranking_harm_rate_max_increase_pp"],
    }
    return {
        "decision": "REPEAT_SMALL_HEADS" if all(checks.values()) else
        "STOP_THIS_VISUAL_FEATURE_PILOT",
        "checks": checks, "comparisons": comparison,
        "scope": "Sampled train-scene holdout only; not detection AP or novelty.",
    }


def prepare(output, registration):
    started = perf_counter()
    data = project_path(DATA)
    manifest, original = checked_inputs(data)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/visual_action_features.py",
             "sdcl/training/visual_action_preflight.py",
             "scripts/training/run_visual_action_preflight.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    chosen, arrays, inputs = {}, {}, {}
    for group, holdout in (("train", False), ("holdout", True)):
        ordered = sorted(
            (s for s, c in manifest["chunks"].items() if c["holdout"] == holdout),
            key=lambda s: hashlib.sha256(("visual-preflight-v1:" + s).encode()).digest())
        count = registration["sample"][group + "_images"]
        if len(ordered) < count:
            raise ValueError("Insufficient scene-separated cached images.")
        chosen[group] = ordered[:count]
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "training_cache_sha256": file_hash(data / "manifest.json"),
        "historical_dependencies": original["source_hashes"],
        "geometry_normalization_checkpoint": "output/runs/localization_gain_960_seed0/last.pt",
        "geometry_normalization_checkpoint_sha256":
            file_hash(project_path("output/runs/localization_gain_960_seed0/last.pt")),
        "image_selection": chosen,
    })
    cv2.setNumThreads(1)
    for group, stems in chosen.items():
        rows = []
        for index, stem in enumerate(stems, 1):
            image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
            metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            image = cv2.imread(str(image_path))
            if image is None or [image.shape[1], image.shape[0]] != metadata["original_size"]:
                raise ValueError("Training image size mismatch.")
            chunk = data / manifest["chunks"][stem]["file"]
            with np.load(chunk, allow_pickle=False) as saved:
                features, valid = saved["features"], saved["valid"]
                targets, weights = saved["targets"], saved["weights"]
            boxes = recover_cached_boxes(features, metadata["original_size"])
            visual = visual_features(image, boxes)
            rows.append((
                normalized_actions(features, original["mean"].numpy(), original["std"].numpy()),
                visual, valid, targets, weights,
                np.full(len(weights), stem.split("_", 1)[0]),
            ))
            inputs[stem] = {"image_sha256": file_hash(image_path),
                            "metadata_sha256": file_hash(metadata_path),
                            "chunk_sha256": file_hash(chunk)}
            if index == 1 or index % 64 == 0 or index == len(stems):
                print(f"Visual cache {group}: {index}/{len(stems)}, "
                      f"{perf_counter() - started:.1f}s.", flush=True)
        arrays[group] = tuple(np.concatenate([r[i] for r in rows]) for i in range(6))
    if set(arrays["train"][5]) & set(arrays["holdout"][5]):
        raise ValueError("Pilot train/holdout scene overlap.")
    fitting = arrays["train"][1][arrays["train"][2]].astype(np.float64)
    mean = fitting.mean(0).astype(np.float32)
    std = np.maximum(fitting.std(0), .01).astype(np.float32)
    for group, row in arrays.items():
        geometry, raw_visual, valid, targets, weights, scenes = row
        visual = np.clip((raw_visual - mean) / std, -6, 6).astype(np.float32)
        with (output / f"{group}_features.npz").open("xb") as stream:
            np.savez_compressed(stream, geometry=geometry, visual=visual,
                                valid=valid, targets=targets, weights=weights, scenes=scenes)
    save_json(output / "input_manifest.json", inputs)
    save_json(output / "visual_normalization.json", {"mean": mean.tolist(), "std": std.tolist()})
    return arrays, perf_counter() - started


def load_prepared(output):
    arrays = {}
    for group in ("train", "holdout"):
        with np.load(output / f"{group}_features.npz", allow_pickle=False) as saved:
            arrays[group] = {k: saved[k] for k in saved.files}
    return arrays


def fit_seed(output, arrays, recipe, seed, gpu, frozen):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = deepcopy(PilotScorer().state_dict())
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = PilotScorer().to(gpu)
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(),
            lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    fitting, held = arrays["train"], arrays["holdout"]
    geometry = torch.from_numpy(fitting["geometry"]).to(gpu)
    visual = torch.from_numpy(fitting["visual"]).to(gpu)
    x = {
        "geometry": torch.cat((geometry, torch.zeros_like(visual)), dim=2),
        "visual": torch.cat((geometry, visual), dim=2),
    }
    valid, target, weight = (torch.from_numpy(fitting[k]).to(gpu)
                             for k in ("valid", "targets", "weights"))
    logs = []
    for epoch in range(recipe["epochs"]):
        order = torch.randperm(len(geometry),
            generator=torch.Generator().manual_seed(seed + epoch)).to(gpu)
        sums = dict.fromkeys(ARMS, 0.)
        total = 0.
        for start in range(0, len(order), recipe["batch"]):
            selected = order[start:start + recipe["batch"]]
            denominator = float(weight[selected].sum())
            total += denominator
            for arm in ARMS:
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                inputs = x[arm.split("_")[0]][selected]
                outputs = models[arm](inputs)
                loss = (ranking_loss(outputs, target[selected], valid[selected],
                                     weight[selected], "uniform") if arm.endswith("ranking") else
                        regression_loss(outputs, target[selected], valid[selected],
                                        weight[selected], "absolute"))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite pilot loss; do not tune after failure.")
                loss.backward()
                optimizer.step()
                sums[arm] += float(loss.detach()) * denominator
        row = {"seed": seed, "epoch": epoch + 1,
               **{a: sums[a] / total for a in ARMS}}
        logs.append(row)
        print(json.dumps(row), flush=True)
    directory = output / f"seed{seed}"
    directory.mkdir()
    save_checkpoint(directory / "last.pt", {
        "seed": seed, "epoch": recipe["epochs"], "recipe": recipe,
        "models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()}
                   for a, m in models.items()},
        "pilot_plan_sha256": file_hash(output / "plan.json"),
        "train_feature_sha256": file_hash(output / "train_features.npz"),
    })
    write_csv(directory / "losses.csv", logs)
    outputs = {a: [] for a in (*ARMS, "frozen_uniform")}
    with torch.inference_mode():
        for start in range(0, len(held["weights"]), recipe["batch"]):
            g = torch.from_numpy(held["geometry"][start:start + recipe["batch"]]).to(gpu)
            v = torch.from_numpy(held["visual"][start:start + recipe["batch"]]).to(gpu)
            for arm in ARMS:
                extra = v if arm.startswith("visual") else torch.zeros_like(v)
                outputs[arm].append(models[arm](torch.cat((g, extra), dim=2)).cpu().numpy())
            outputs["frozen_uniform"].append(frozen(g).cpu().numpy())
    return {a: pilot_metrics(np.concatenate(values), held["valid"], held["targets"],
                             held["weights"]) for a, values in outputs.items()}


def run(device="0", output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text(encoding="utf-8"))
    if registration["arms"] != list(ARMS) or registration["recipe"]["epochs"] != 8:
        raise ValueError("Pilot registration changed.")
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = project_path(output)
    _, preparation_seconds = prepare(output, registration)
    arrays = load_prepared(output)
    gpu = torch.device(f"cuda:{device}")
    checkpoint_path = project_path("output/runs/regret_objective_probe_960_seed0/last.pt")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frozen = ActionScorer().to(gpu).eval()
    frozen.load_state_dict(saved["models"]["uniform"])
    results, decisions = {}, {}
    for seed in registration["recipe"]["seeds"]:
        if seed and decisions["0"]["decision"] != "REPEAT_SMALL_HEADS":
            break
        results[str(seed)] = fit_seed(output, arrays, registration["recipe"], seed, gpu, frozen)
        decisions[str(seed)] = pilot_decision(results[str(seed)], registration["policy"])
        save_json(output / f"seed{seed}" / "metrics.json", results[str(seed)])
        save_json(output / f"seed{seed}" / "decision.json", decisions[str(seed)])
    verdict = {
        "decision": decisions["0"]["decision"], "per_seed": decisions,
        "head_seeds_completed": list(results), "official_evaluation_ran": False,
        "detector_inference_ran": False, "detector_training_ran": False,
    }
    if len(results) == 3:
        deltas = [d["comparisons"]["visual_ranking_vs_geometry_ranking"] for d in decisions.values()]
        gain = float(np.mean([d["owned_ten_threshold_percent_pp"] for d in deltas]))
        harm = float(np.mean([d["owned_harm_rate_percent_pp"] for d in deltas]))
        iou = float(np.mean([d["owned_mean_iou_percent_pp"] for d in deltas]))
        positive = sum(d["owned_ten_threshold_percent_pp"] > 0 for d in deltas)
        passed = (gain >= registration["policy"]["repeat_mean_threshold_min_pp"] and
                  positive >= registration["policy"]["repeat_positive_seeds_min"] and
                  iou > 0 and harm <= .5)
        verdict.update(
            decision="VISUAL_SIGNAL_WORTH_LARGER_HEAD_SCREEN" if passed else
            "NO_REPEATABLE_PILOT_ADVANTAGE",
            mean_owned_threshold_gain_pp=gain, mean_owned_iou_gain_pp=iou,
            mean_harm_increase_pp=harm, threshold_positive_seeds=positive,
        )
    plan = json.loads((output / "plan.json").read_text())
    if not all(file_hash(project_path(p)) == h for p, h in plan["source_hashes"].items()):
        raise ValueError("Pilot source/registration changed during run.")
    save_json(output / "metrics.json", results)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "train_scene_count": len(set(arrays["train"]["scenes"])),
        "holdout_scene_count": len(set(arrays["holdout"]["scenes"])),
        "train_rows": len(arrays["train"]["weights"]),
        "holdout_rows": len(arrays["holdout"]["weights"]),
        "scene_overlap": 0, "source_unchanged": True,
        "selected_image_count": sum(len(v) for v in plan["image_selection"].values()),
        "nominal_parameters_per_head": sum(p.numel() for p in PilotScorer().parameters()),
        "fixed_uniform_checkpoint_sha256": file_hash(checkpoint_path),
        "preparation_seconds": preparation_seconds,
        "total_seconds": perf_counter() - started,
        "scope": "Sampled fixed-reference training holdout, not AP/AR or independent-data evidence.",
    })
    return {"output": str(output), "decision": verdict, "metrics": results,
            "elapsed_seconds": perf_counter() - started}
