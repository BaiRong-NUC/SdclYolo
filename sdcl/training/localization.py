"""One fixed, resumable pair of localization heads on train-only geometry."""

from dataclasses import asdict
import hashlib
import json
import os
from time import perf_counter
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash, record_hash, write_csv
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.postprocess.localization import (
    ACTIONS, CONTROL_SHA256, VARIANTS, ActionScorer, LocalizationRecipe, action_feature_names, action_menu,
    action_targets, calibrated_candidates, check_sources, frozen_control, load_calibrator,
    load_recipe, normalized_actions, sample_anchors, source_hashes,
)
from sdcl.postprocess.rescue import save_json, scene_holdout
from sdcl.runtime.environment import environment_info
from sdcl.runtime.files import atomic_replace
from sdcl.training.rescue import archive_source, checked_data, save_checkpoint


DATA = "output/localization/train_actions_seed0"
RUN = "output/runs/localization_gain_960_seed0"


def verify_archive(directory, hashes, digest):
    path = directory / "source.zip"
    if file_hash(path) != digest:
        raise ValueError("Localization source archive changed.")
    with zipfile.ZipFile(path) as archive:
        if (set(archive.namelist()) != set(hashes)
                or any(hashlib.sha256(archive.read(name)).hexdigest() != value
                       for name, value in hashes.items())):
            raise ValueError("Localization archived source mismatch.")


def prepare_localization(output=DATA, device="0", limit=None):
    import cv2
    from ultralytics import YOLO
    from sdcl.training.trainer import check_version

    recipe = load_recipe()
    control = frozen_control()
    check_version()
    if not str(device).isdigit() or (limit is not None and not 2 <= limit <= 6471):
        raise ValueError("Use a CUDA index; train-only smoke limit must be in [2, 6471].")
    dataset, checkpoint, all_records = checked_data()
    records = all_records
    if limit is not None:
        chosen = np.random.default_rng(recipe.seed).choice(len(records), limit, replace=False)
        records = [records[index] for index in sorted(chosen)]
    output = project_path(output)
    manifest_path = output / "manifest.json"
    expected = {
        "recipe": asdict(recipe), "split": "train", "control_sha256": CONTROL_SHA256,
        "all_train_records_sha256": record_hash(all_records),
        "selected_records_sha256": record_hash(records),
        "full_training_data": limit is None, "images": len(records),
        "source_hashes": source_hashes(), "feature_names": action_feature_names(),
        "holdout_rule": "SHA256(scene prefix) modulo 10 == 0; training scenes only.",
        "targets": "Same baseline-owned GT for every action; unmatched/duplicate anchors zero.",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["settings"] != expected:
            raise ValueError("Action cache source/recipe/data mismatch; refusing reuse.")
        verify_archive(output, expected["source_hashes"], manifest["source_archive_sha256"])
        for chunk in manifest["chunks"].values():
            if file_hash(output / chunk["file"]) != chunk["sha256"]:
                raise ValueError("Action cache chunk changed.")
    else:
        if output.exists():
            raise FileExistsError(f"Directory exists without a matching manifest: {output}")
        (output / "chunks").mkdir(parents=True)
        archive_source(output, expected["source_hashes"])
        manifest = {"settings": expected, "complete": False, "chunks": {},
                    "source_archive_sha256": file_hash(output / "source.zip")}
        save_json(manifest_path, manifest)
    if manifest["complete"]:
        if set(manifest["chunks"]) != {r["stem"] for r in records}:
            raise ValueError("Completed cache has the wrong image set.")
        return {"output": str(output), "images": len(records), "complete": True}
    model = YOLO(str(checkpoint))
    calibrator = load_calibrator(control)
    kwargs = {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 500,
              "quantize": 16, "rect": False, "batch": 1, "device": device,
              "verbose": False, "save": False, "augment": False, "agnostic_nms": False}

    def predict(image):
        boxes = model.predict(source=image, **kwargs)[0].boxes
        return torch.cat((boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]), 1).float().cpu().numpy()

    predict(np.zeros((960, 960, 3), dtype=np.uint8))
    if not model.predictor.model.fp16:
        raise ValueError("Frozen view policy requires effective FP16.")
    started = perf_counter()
    for index, record in enumerate(records, 1):
        if record["stem"] in manifest["chunks"]:
            continue
        image = cv2.imread(record["image"])
        if image is None or record["size"] != [image.shape[1], image.shape[0]]:
            raise ValueError(f"Invalid training image: {record['stem']}")
        ignores = json.loads(
            (dataset / "metadata/train" / f"{record['stem']}.json").read_text(encoding="utf-8")
        )["ignore_boxes_xyxy"]
        windows = [[0, 0, *record["size"]], *tile_windows(*record["size"])]
        views = [predict(image)]
        for window in windows[1:]:
            x1, y1, x2, y2 = window
            views.append(map_tile_predictions(predict(image[y1:y2, x1:x2].copy()), window))
        candidates, origins, raw_scores, base = calibrated_candidates(
            views, windows, record["size"], ignores, calibrator)
        boxes, valid, features = action_menu(
            candidates, origins, raw_scores, base, views, record["size"], ignores)
        targets, owned = action_targets(record, candidates, boxes)
        selected, weights = sample_anchors(owned, record["stem"])
        target = output / "chunks" / f"{record['stem']}.npz"
        partial = target.with_name(target.name + ".partial")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, features=features[selected], valid=valid[selected],
                                targets=targets[selected], weights=weights)
        atomic_replace(partial, target)
        manifest["chunks"][record["stem"]] = {
            "file": target.relative_to(output).as_posix(), "sha256": file_hash(target),
            "holdout": scene_holdout(record["stem"]), "rows": len(selected),
            "owned_gt": int(owned.sum()), "total_anchors": len(candidates),
            "peer_anchors": int(valid[:, 1:5].any(1).sum()),
        }
        save_json(manifest_path, manifest)
        if index == 1 or index % 50 == 0 or index == len(records):
            print(f"Prepared localization actions: {index}/{len(records)}, "
                  f"{perf_counter() - started:.1f}s.", flush=True)
    check_sources(expected["source_hashes"])
    _, _, current = checked_data()
    if record_hash(current) != expected["all_train_records_sha256"]:
        raise ValueError("Training data changed during action preparation.")
    if set(manifest["chunks"]) != {r["stem"] for r in records}:
        raise ValueError("Action cache image set mismatch.")
    manifest["complete"] = True
    save_json(manifest_path, manifest)
    return {"output": str(output), "images": len(records), "complete": True,
            "full_training_data": expected["full_training_data"],
            "rows": sum(chunk["rows"] for chunk in manifest["chunks"].values())}


def verify_cache(path, saved=None):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    if (not manifest["complete"] or settings["split"] != "train"
            or settings["recipe"] != asdict(load_recipe())
            or settings["control_sha256"] != CONTROL_SHA256
            or settings["feature_names"] != action_feature_names()
            or len(manifest["chunks"]) != settings["images"]
            or (settings["full_training_data"] and settings["images"] != 6471)):
        raise ValueError("Incomplete or incompatible action cache.")
    check_sources(settings["source_hashes"])
    verify_archive(path, settings["source_hashes"], manifest["source_archive_sha256"])
    if saved is not None and (
            file_hash(path / "manifest.json") != saved["cache_sha256"]
            or saved["source_hashes"] != settings["source_hashes"]
            or saved["full_training_data"] != settings["full_training_data"]
            or saved["control_sha256"] != CONTROL_SHA256):
        raise ValueError("Checkpoint/action cache provenance mismatch.")
    for chunk in manifest["chunks"].values():
        if file_hash(path / chunk["file"]) != chunk["sha256"]:
            raise ValueError("Action chunk content changed.")
    _, _, records = checked_data()
    if record_hash(records) != settings["all_train_records_sha256"]:
        raise ValueError("Training record hash changed.")
    selected = [r for r in records if r["stem"] in manifest["chunks"]]
    if (record_hash(selected) != settings["selected_records_sha256"]
            or any(chunk["holdout"] != scene_holdout(stem)
                   for stem, chunk in manifest["chunks"].items())):
        raise ValueError("Action cache split/selection mismatch.")
    return manifest


def read_action_cache(path, manifest):
    groups = {"train": [], "holdout": []}
    keys = ("features", "valid", "targets", "weights")
    for chunk in manifest["chunks"].values():
        with np.load(path / chunk["file"], allow_pickle=False) as saved:
            arrays = tuple(saved[key] for key in keys)
        x, valid, target, weight = arrays
        n = chunk["rows"]
        if (x.shape != (n, ACTIONS, len(action_feature_names()))
                or valid.shape != (n, ACTIONS) or valid.dtype != np.bool_
                or target.shape != valid.shape or weight.shape != (n,)
                or not valid[:, 0].all() or not np.isfinite(x).all()
                or not np.isfinite(target).all() or not np.isfinite(weight).all()
                or np.any(weight <= 0) or np.any((target < 0) | (target > 1))):
            raise ValueError("Malformed localization training chunk.")
        groups["holdout" if chunk["holdout"] else "train"].append(arrays)
    result = {}
    for group, chunks in groups.items():
        result[group] = tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(4)) if chunks else (
            np.empty((0, ACTIONS, len(action_feature_names())), np.float32),
            np.empty((0, ACTIONS), bool), np.empty((0, ACTIONS), np.float32),
            np.empty(0, np.float32))
    if not len(result["train"][0]):
        raise ValueError("No fitting anchors after scene holdout.")
    return result


def regression_loss(output, targets, valid, weights, variant):
    if variant == "gain":
        targets = targets - targets[:, :1]
    elif variant != "absolute":
        raise ValueError("Unknown training variant.")
    losses = F.smooth_l1_loss(output, targets, reduction="none", beta=LocalizationRecipe().smooth_l1_beta)
    per_anchor = (losses * valid).sum(1) / valid.sum(1).clamp_min(1)
    return (per_anchor * weights).sum() / weights.sum()


def fitting_statistics(features, valid):
    # Stream moments to avoid an additional full copy of the action cache.
    count = 0
    total = np.zeros(features.shape[-1], dtype=np.float64)
    square = total.copy()
    for start in range(0, len(features), 4096):
        x = features[start:start + 4096][valid[start:start + 4096]].astype(np.float64)
        count += len(x)
        total += x.sum(0)
        square += (x * x).sum(0)
    if not count:
        raise ValueError("No valid training actions.")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 0))
    return mean.astype(np.float32), np.maximum(std, 0.05).astype(np.float32)


def train_localization(data=DATA, output=RUN, device="0", resume=False):
    recipe = load_recipe()
    frozen_control()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA device index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True)
    data, output = project_path(data), project_path(output)
    path = output / "last.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False) if resume else None
    manifest = verify_cache(data, saved)
    if saved is not None:
        if saved["recipe"] != asdict(recipe) or not 1 <= saved["epoch"] <= recipe.epochs:
            raise ValueError("Resume recipe/epoch mismatch.")
        verify_archive(output, saved["source_hashes"], saved["source_archive_sha256"])
    elif output.exists():
        raise FileExistsError(f"Run exists; use --resume: {output}")
    else:
        output.mkdir(parents=True)
        archive_source(output, manifest["settings"]["source_hashes"])
    arrays = read_action_cache(data, manifest)
    features, valid, targets, weights = arrays["train"]
    mean, std = fitting_statistics(features, valid) if saved is None else (
        saved["mean"].numpy(), saved["std"].numpy())
    gpu = torch.device(f"cuda:{device}")
    x = torch.from_numpy(normalized_actions(features, mean, std)).to(gpu)
    v, y, w = (torch.from_numpy(a).to(gpu) for a in (valid, targets, weights))
    initial = ActionScorer().state_dict()
    models, optimizers = {}, {}
    for variant in VARIANTS:
        models[variant] = ActionScorer().to(gpu)
        models[variant].load_state_dict(initial)
        optimizers[variant] = torch.optim.AdamW(
            models[variant].parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        if saved is not None:
            models[variant].load_state_dict(saved["models"][variant])
            optimizers[variant].load_state_dict(saved["optimizers"][variant])
    from sdcl.evaluation.localization_screen import GATES
    plan = {
        "recipe": asdict(recipe), "source_hashes": source_hashes(),
        "source_archive_sha256": file_hash(output / "source.zip"),
        "control_sha256": CONTROL_SHA256, "cache_sha256": file_hash(data / "manifest.json"),
        "cache_path": str(data), "full_training_data": manifest["settings"]["full_training_data"],
        "rows": {key: len(value[0]) for key, value in arrays.items()},
        "selection": "Epoch20 only; holdout monitors loss; no validation fitting or selection.",
        "fixed_gates": GATES, "environment": environment_info(),
    }
    save_json(output / "plan.json", plan)
    logs = saved["logs"] if saved is not None else []
    start = saved["epoch"] if saved is not None else 0
    for epoch in range(start, recipe.epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(recipe.seed + epoch)).to(gpu)
        sums, total = dict.fromkeys(VARIANTS, 0.0), 0.0
        for offset in range(0, len(order), recipe.batch):
            chosen = order[offset:offset + recipe.batch]
            denominator = float(w[chosen].sum())
            total += denominator
            for variant in VARIANTS:
                model, optimizer = models[variant], optimizers[variant]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = regression_loss(model(x[chosen]), y[chosen], v[chosen], w[chosen], variant)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite localization loss.")
                loss.backward()
                optimizer.step()
                sums[variant] += float(loss.detach()) * denominator
        row = {"epoch": epoch + 1}
        for variant in VARIANTS:
            row[f"{variant}_train_loss"] = sums[variant] / total
            row[f"{variant}_holdout_loss"] = None
            hx, hv, hy, hw = arrays["holdout"]
            if len(hx):
                numerator, denominator = 0.0, 0.0
                models[variant].eval()
                with torch.inference_mode():
                    for offset in range(0, len(hx), recipe.batch):
                        stop = offset + recipe.batch
                        bx = torch.from_numpy(normalized_actions(hx[offset:stop], mean, std)).to(gpu)
                        bv, by, bw = (torch.from_numpy(a[offset:stop]).to(gpu) for a in (hv, hy, hw))
                        loss = regression_loss(models[variant](bx), by, bv, bw, variant)
                        denominator += float(bw.sum())
                        numerator += float(loss) * float(bw.sum())
                row[f"{variant}_holdout_loss"] = numerator / denominator
        logs.append(row)
        save_checkpoint(path, {
            **plan, "epoch": epoch + 1, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
            "models": {name: {k: value.detach().cpu() for k, value in model.state_dict().items()}
                       for name, model in models.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "logs": logs,
        })
        write_csv(output / "losses.csv", logs)
        print(json.dumps(row), flush=True)
    check_sources(plan["source_hashes"])
    return {"output": str(output), "checkpoint": str(path), "epoch": recipe.epochs,
            "full_training_data": plan["full_training_data"], "models": list(VARIANTS)}
