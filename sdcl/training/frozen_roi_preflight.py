"""Cost-bounded local CNN evidence pilot with matched ordinary visual controls."""

from copy import deepcopy
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.postprocess.frozen_roi_features import (
    CHANNELS, VISUAL_DIM, fixed_projection, image_tensor, sample_context,
)
from sdcl.postprocess.localization import ActionScorer
from sdcl.postprocess.rescue import save_json
from sdcl.postprocess.visual_action_features import recover_cached_boxes
from sdcl.training.localization import DATA, regression_loss
from sdcl.training.regret_probe import ranking_loss
from sdcl.training.rescue import archive_source, save_checkpoint
from sdcl.training.visual_action_preflight import ARMS, load_prepared, pilot_decision, pilot_metrics


REGISTRATION = "docs/experiments/candidate10_frozen_roi_preflight_registration_20261014.json"
OUTPUT = "output/analysis/frozen-roi-preflight-fixed-20261014"
PRIOR = "output/analysis/visual-preflight-fixed-20261014"


class LocalScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(68 + VISUAL_DIM, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def prepare(output, registration, gpu):
    from ultralytics import YOLO

    prior = project_path(PRIOR)
    prior_plan = json.loads((prior / "plan.json").read_text())
    result_record = json.loads(project_path(
        "docs/experiments/candidate9_visual_preflight_result_20261014.json").read_text())
    for name, digest in result_record["artifacts_sha256"].items():
        if file_hash(prior / name) != digest:
            raise ValueError("Prior sample/cache or results changed.")
    verify_dependencies(prior_plan["source_hashes"], prior / "source.zip")
    verify_dependencies(prior_plan["historical_dependencies"],
                        project_path("output/runs/localization_gain_960_seed0/source.zip"))
    detector_path = project_path(registration["visual_input"]["detector"])
    if file_hash(detector_path) != registration["visual_input"]["detector_sha256"]:
        raise ValueError("Frozen detector checkpoint changed.")
    data = project_path(DATA)
    if file_hash(data / "manifest.json") != prior_plan["training_cache_sha256"]:
        raise ValueError("Historical training menu changed.")
    manifest = json.loads((data / "manifest.json").read_text())
    inputs = json.loads((prior / "input_manifest.json").read_text())
    arrays = load_prepared(prior)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/frozen_roi_features.py",
             "sdcl/training/frozen_roi_preflight.py",
             "scripts/training/run_frozen_roi_preflight.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "prior_plan_sha256": file_hash(prior / "plan.json"),
        "prior_result_record_sha256": file_hash(project_path(
            "docs/experiments/candidate9_visual_preflight_result_20261014.json")),
        "image_selection": prior_plan["image_selection"],
        "frozen_uniform_checkpoint_sha256":
            file_hash(project_path("output/runs/regret_objective_probe_960_seed0/last.pt")),
    })
    detector = YOLO(str(detector_path)).model.to(gpu).half().eval()
    detector.requires_grad_(False)
    head = detector.model[-1]
    if type(head).__name__ != "Detect" or head.stride.tolist() != [8., 16., 32.]:
        raise ValueError("Frozen P3 feature policy requires the registered three-level head.")
    captured = {}
    def capture(_module, args):
        captured["p3"] = args[0][0].detach()
    hook = head.register_forward_pre_hook(capture)
    projection = fixed_projection(gpu)
    cv2.setNumThreads(1)
    tick, forwards = perf_counter(), 0
    with torch.inference_mode():
        for group, stems in prior_plan["image_selection"].items():
            values = []
            for index, stem in enumerate(stems, 1):
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                chunk_path = data / manifest["chunks"][stem]["file"]
                expected = inputs[stem]
                if (file_hash(image_path) != expected["image_sha256"] or
                        file_hash(metadata_path) != expected["metadata_sha256"] or
                        file_hash(chunk_path) != expected["chunk_sha256"]):
                    raise ValueError("Selected image or cached coordinates changed.")
                image = cv2.imread(str(image_path))
                size = json.loads(metadata_path.read_text())["original_size"]
                if image is None or [image.shape[1], image.shape[0]] != size:
                    raise ValueError("Original training image size mismatch.")
                with np.load(chunk_path, allow_pickle=False) as chunk:
                    boxes = recover_cached_boxes(chunk["features"], size)
                detector(image_tensor(image, gpu))
                forwards += 1
                p3 = captured.pop("p3")
                if tuple(p3.shape[1:]) != (CHANNELS, 120, 120):
                    raise ValueError("Wrong frozen local feature geometry.")
                projected = F.conv2d(p3.float(), projection)
                values.append(sample_context(projected, boxes, size))
                if index == 1 or index % 64 == 0 or index == len(stems):
                    print(f"Frozen local cache {group}: {index}/{len(stems)}, "
                          f"{perf_counter() - tick:.1f}s.", flush=True)
            arrays[group]["visual"] = np.concatenate(values)
    hook.remove()
    del detector
    torch.cuda.empty_cache()
    train = arrays["train"]
    # Stream visual moments to avoid a second full float64 action tensor.
    total = np.zeros(VISUAL_DIM, np.float64)
    square, count = total.copy(), 0
    for start in range(0, len(train["weights"]), 1024):
        batch = train["visual"][start:start + 1024][train["valid"][start:start + 1024]]
        batch = batch.astype(np.float64)
        count += len(batch)
        total += batch.sum(0)
        square += (batch * batch).sum(0)
    mean = (total / count).astype(np.float32)
    std = np.maximum(np.sqrt(np.maximum(square / count - (total / count) ** 2, 0)),
                     .01).astype(np.float32)
    for group, row in arrays.items():
        row["visual"] = np.clip((row["visual"] - mean) / std, -6, 6).astype(np.float32)
        with (output / f"{group}_features.npz").open("xb") as stream:
            np.savez_compressed(stream, **row)
    save_json(output / "visual_normalization.json", {"mean": mean.tolist(), "std": std.tolist()})
    return arrays, {"detector_forwards": forwards, "feature_extraction_seconds": perf_counter() - tick}


def fit_seed(output, arrays, recipe, seed, gpu, frozen):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = deepcopy(LocalScorer().state_dict())
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = LocalScorer().to(gpu)
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(),
            lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    fit, held = arrays["train"], arrays["holdout"]
    g, v = (torch.from_numpy(fit[k]).to(gpu) for k in ("geometry", "visual"))
    inputs = {"geometry": torch.cat((g, torch.zeros_like(v)), 2),
              "visual": torch.cat((g, v), 2)}
    del g, v
    valid, targets, weights = (torch.from_numpy(fit[k]).to(gpu)
                               for k in ("valid", "targets", "weights"))
    logs = []
    for epoch in range(recipe["epochs"]):
        order = torch.randperm(len(weights),
            generator=torch.Generator().manual_seed(seed + epoch)).to(gpu)
        sums, total = dict.fromkeys(ARMS, 0.), 0.
        for start in range(0, len(order), recipe["batch"]):
            selected = order[start:start + recipe["batch"]]
            denominator = float(weights[selected].sum())
            total += denominator
            for arm in ARMS:
                optimizers[arm].zero_grad(set_to_none=True)
                result = models[arm](inputs[arm.split("_")[0]][selected])
                loss = (ranking_loss(result, targets[selected], valid[selected],
                                     weights[selected], "uniform") if arm.endswith("ranking") else
                        regression_loss(result, targets[selected], valid[selected],
                                        weights[selected], "absolute"))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite local pilot loss.")
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
                outputs[arm].append(models[arm](torch.cat((g, extra), 2)).cpu().numpy())
            outputs["frozen_uniform"].append(frozen(g).cpu().numpy())
    outputs = {a: np.concatenate(v) for a, v in outputs.items()}
    metrics = {a: pilot_metrics(v, held["valid"], held["targets"], held["weights"])
               for a, v in outputs.items()}
    utilities = np.where(held["valid"], outputs["geometry_ranking"], -np.inf)
    ordered = np.sort(utilities, axis=1)
    ambiguous = (held["valid"].sum(1) >= 2) & (ordered[:, -1] - ordered[:, -2] <= .05)
    secondary = {"rows": int(ambiguous.sum()), "definition": "Geometry top-two gap<=0.05.",
                 "metrics": {a: pilot_metrics(v[ambiguous], held["valid"][ambiguous],
                                              held["targets"][ambiguous], held["weights"][ambiguous])
                             for a, v in outputs.items()}}
    save_json(directory / "ambiguity_subset.json", secondary)
    return metrics


def run(device="0", output=OUTPUT):
    started = perf_counter()
    registration = json.loads(project_path(REGISTRATION).read_text())
    if registration["arms"] != list(ARMS) or registration["recipe"]["epochs"] != 8:
        raise ValueError("Fixed local feature registration changed.")
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu, output = torch.device(f"cuda:{device}"), project_path(output)
    arrays, extraction = prepare(output, registration, gpu)
    checkpoint = project_path("output/runs/regret_objective_probe_960_seed0/last.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frozen = ActionScorer().to(gpu).eval()
    frozen.load_state_dict(saved["models"]["uniform"])
    metrics, decisions = {}, {}
    for seed in registration["recipe"]["seeds"]:
        if seed and decisions["0"]["decision"] != "REPEAT_SMALL_HEADS":
            break
        metrics[str(seed)] = fit_seed(output, arrays, registration["recipe"], seed, gpu, frozen)
        decisions[str(seed)] = pilot_decision(metrics[str(seed)], registration["policy"])
        save_json(output / f"seed{seed}" / "metrics.json", metrics[str(seed)])
        save_json(output / f"seed{seed}" / "decision.json", decisions[str(seed)])
    verdict = {"decision": decisions["0"]["decision"], "per_seed": decisions,
               "head_seeds_completed": list(metrics), "official_evaluation_ran": False,
               "detector_training_ran": False}
    if len(metrics) == 3:
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
        raise ValueError("Local feature pilot source changed.")
    if set(arrays["train"]["scenes"]) & set(arrays["holdout"]["scenes"]):
        raise ValueError("Head train/holdout scene overlap.")
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        **extraction, "source_unchanged": True, "scene_overlap": 0,
        "selected_train_images": 256, "selected_holdout_images": 128,
        "train_rows": len(arrays["train"]["weights"]),
        "holdout_rows": len(arrays["holdout"]["weights"]),
        "nominal_parameters_per_head": sum(p.numel() for p in LocalScorer().parameters()),
        "total_seconds": perf_counter() - started,
        "scope": "Head-held-out training scenes; frozen detector trained on these images historically.",
    })
    return {"output": str(output), "decision": verdict,
            "elapsed_seconds": perf_counter() - started}
