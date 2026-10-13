"""Resumable train-only feature preparation and one fixed three-model run."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.runtime.environment import environment_info
from sdcl.runtime.files import atomic_replace
from sdcl.postprocess.rescue import (
    VARIANTS, RescueRecipe, RescueScorer, candidate_features, check_sources,
    feature_layout, load_recipe, normalized_features, rescue_labels, sample_candidates,
    save_json, scene_holdout, soft_nms_with_origins, source_hashes, variant_labels,
)
from sdcl.evaluation.size_analysis import file_hash, load_records, record_hash, write_csv
from sdcl.evaluation.slicing_screen import (
    BASELINE, BASELINE_SHA256, DATA_YAML_SHA256, EXPECTED_IMAGES, RECORDS_SHA256,
    tile_windows,
)


DATA = "output/rescue/train_features_seed0"
RUN = "output/runs/rescue_softnms_960_seed0"


def checked_data():
    dataset = project_path("data/visdrone")
    checkpoint = project_path(BASELINE)
    if file_hash(checkpoint) != BASELINE_SHA256 or file_hash(dataset / "dataset.yaml") != DATA_YAML_SHA256:
        raise ValueError("Frozen detector/data YAML mismatch.")
    validation = load_records(dataset, "val")
    if len(validation) != EXPECTED_IMAGES or record_hash(validation) != RECORDS_SHA256:
        raise ValueError("Frozen validation data changed.")
    training = load_records(dataset, "train")
    if len(training) != 6471:
        raise ValueError("Frozen screen requires all 6471 training images.")
    if {r["stem"] for r in training} & {r["stem"] for r in validation}:
        raise ValueError("Training/validation stem overlap.")
    return dataset, checkpoint, training


def archive_source(output, hashes):
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(project_path(name), name)


def prepare_rescue(output=DATA, device="0", limit=None):
    import cv2
    from ultralytics import YOLO
    from sdcl.evaluation.fusion_screen import soft_merge_predictions
    from sdcl.training.trainer import check_version

    check_version()
    recipe = load_recipe()
    if not str(device).isdigit() or (limit is not None and not 2 <= limit <= 6471):
        raise ValueError("Use a CUDA device index and smoke limit >=2.")
    dataset, checkpoint, all_records = checked_data()
    records = all_records
    if limit is not None:
        chosen = np.random.default_rng(recipe.seed).choice(len(records), limit, replace=False)
        records = [records[index] for index in sorted(chosen)]
    output = project_path(output)
    manifest_path = output / "manifest.json"
    expected = {
        "version": recipe.version, "recipe": asdict(recipe), "split": "train",
        "checkpoint_sha256": BASELINE_SHA256, "data_yaml_sha256": DATA_YAML_SHA256,
        "all_train_records_sha256": record_hash(all_records),
        "selected_records_sha256": record_hash(records),
        "full_training_data": limit is None, "images": len(records),
        "source_hashes": source_hashes(), "feature_names": feature_layout()[0],
        "holdout_rule": "SHA256(scene prefix) modulo 10 == 0; scene grouping within train only.",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["settings"] != expected:
            raise ValueError("Feature cache recipe/source/data mismatch; refusing to reuse.")
        for stem, chunk in manifest["chunks"].items():
            if file_hash(output / chunk["file"]) != chunk["sha256"]:
                raise ValueError(f"Corrupt feature chunk: {stem}")
    else:
        if output.exists():
            raise FileExistsError(f"Directory exists without a valid manifest: {output}")
        output.mkdir(parents=True)
        (output / "chunks").mkdir()
        manifest = {"settings": expected, "complete": False, "chunks": {},
                    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
        save_json(manifest_path, manifest)
        archive_source(output, expected["source_hashes"])
    if manifest["complete"]:
        print(f"Full matching feature cache already prepared: {output}", flush=True)
        return {"output": str(output), "images": len(records), "complete": True,
                "full_training_data": expected["full_training_data"]}
    model = YOLO(str(checkpoint))
    kwargs = {"imgsz": 960, "conf": 0.001, "iou": 0.7, "max_det": 500,
              "quantize": 16, "rect": False, "batch": 1, "device": device,
              "verbose": False, "save": False, "augment": False, "agnostic_nms": False}

    def predict(image):
        boxes = model.predict(source=image, **kwargs)[0].boxes
        return torch.cat((boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]), 1).float().cpu().numpy()

    predict(np.zeros((960, 960, 3), dtype=np.uint8))
    if not model.predictor.model.fp16:
        raise ValueError("Frozen feature preparation requires FP16.")
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
        for x1, y1, x2, y2 in windows[1:]:
            from sdcl.evaluation.slicing_screen import map_tile_predictions
            views.append(map_tile_predictions(predict(image[y1:y2, x1:x2].copy()), (x1, y1, x2, y2)))
        candidates, origins, raw_scores = soft_nms_with_origins(views, record["size"], ignores)
        reference = soft_merge_predictions(views, record["size"], ignores)[0]
        if not np.array_equal(candidates, reference):
            raise AssertionError("Origin-tracking Soft-NMS differs from the strong control.")
        labels = rescue_labels(record, candidates, origins, views, windows, ignores)
        selected, weights = sample_candidates(labels, record["stem"])
        features = candidate_features(candidates[selected], origins[selected], raw_scores[selected],
                                      views, windows, record["size"], ignores)
        relative = f"chunks/{record['stem']}.npz"
        target = output / relative
        partial = target.with_name(target.name + ".partial")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, features=features, labels=labels[selected], weights=weights)
        atomic_replace(partial, target)
        manifest["chunks"][record["stem"]] = {
            "file": relative, "sha256": file_hash(target),
            "holdout": scene_holdout(record["stem"]), "rows": len(selected),
            "full_state_counts": np.bincount(labels, minlength=4).tolist(),
            "sampled_state_counts": np.bincount(labels[selected], minlength=4).tolist(),
        }
        save_json(manifest_path, manifest)
        if index == 1 or index % 50 == 0 or index == len(records):
            print(f"Prepared train-only features: {index}/{len(records)}, "
                  f"{perf_counter() - started:.1f}s.", flush=True)
    check_sources(expected["source_hashes"])
    if record_hash(load_records(dataset, "train")) != expected["all_train_records_sha256"]:
        raise ValueError("Training data changed during preparation.")
    manifest["complete"] = True
    manifest["state_names"] = ["false_positive", "supported", "observable_rescue", "sole_observer"]
    save_json(manifest_path, manifest)
    return {"output": str(output), "images": len(records), "complete": True,
            "full_training_data": expected["full_training_data"],
            "sampled_rows": sum(chunk["rows"] for chunk in manifest["chunks"].values())}


def read_feature_cache(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["settings"]["split"] != "train":
        raise ValueError("Incomplete or non-training feature cache.")
    check_sources(manifest["settings"]["source_hashes"])
    groups = {"train": [], "holdout": []}
    for chunk in manifest["chunks"].values():
        file = path / chunk["file"]
        if file_hash(file) != chunk["sha256"]:
            raise ValueError("Feature-cache content hash mismatch.")
        with np.load(file, allow_pickle=False) as saved:
            groups["holdout" if chunk["holdout"] else "train"].append(
                (saved["features"], saved["labels"], saved["weights"]))
    result = {}
    for name, chunks in groups.items():
        if chunks:
            result[name] = tuple(np.concatenate([chunk[index] for chunk in chunks]) for index in range(3))
        else:
            result[name] = (np.empty((0, len(feature_layout()[0])), dtype=np.float32),
                            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
    if not len(result["train"][0]):
        raise ValueError("No training rows after scene holdout.")
    return manifest, result


def save_checkpoint(path, value):
    partial = path.with_name(path.name + ".partial")
    torch.save(value, partial)
    atomic_replace(partial, path)


def train_rescue(data=DATA, output=RUN, device="0", resume=False):
    recipe = load_recipe()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA device index.")
    data, output = project_path(data), project_path(output)
    manifest, arrays = read_feature_cache(data)
    _, _, records = checked_data()
    if record_hash(records) != manifest["settings"]["all_train_records_sha256"]:
        raise ValueError("Current training data differs from the prepared cache.")
    cache_hash = file_hash(data / "manifest.json")
    checkpoint_path = output / "last.pt"
    existing = None
    if resume:
        existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (existing["cache_sha256"] != cache_hash
                or existing["recipe"] != asdict(recipe)):
            raise ValueError("Resume checkpoint/cache/recipe mismatch.")
        check_sources(existing["source_hashes"])
    else:
        if output.exists():
            raise FileExistsError(f"Run exists; use --resume to continue: {output}")
        output.mkdir(parents=True)
        archive_source(output, manifest["settings"]["source_hashes"])
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True)
    # Small dense CUDA layers require this for deterministic cuBLAS operations.
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    features, labels, weights = arrays["train"]
    mean, std = features.mean(axis=0), np.maximum(features.std(axis=0), 0.05)
    if existing:
        mean, std = existing["mean"].numpy(), existing["std"].numpy()
    gpu = torch.device(f"cuda:{device}")
    base_model = RescueScorer()
    initial = base_model.state_dict()
    models, optimizers, training_inputs, targets = {}, {}, {}, {}
    for variant in VARIANTS:
        models[variant] = RescueScorer().to(gpu)
        models[variant].load_state_dict(initial)
        optimizers[variant] = torch.optim.AdamW(
            models[variant].parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        training_inputs[variant] = torch.from_numpy(
            normalized_features(features, mean, std, variant)).to(gpu)
        targets[variant] = torch.from_numpy(variant_labels(labels, variant)).to(gpu)
        if existing:
            models[variant].load_state_dict(existing["models"][variant])
            optimizers[variant].load_state_dict(existing["optimizers"][variant])
    importance = torch.from_numpy(weights).to(gpu)
    logs = existing["logs"] if existing else []
    start = existing["epoch"] if existing else 0
    plan = {
        "recipe": asdict(recipe), "cache_sha256": cache_hash, "source_hashes": source_hashes(),
        "data": str(data), "full_training_data": manifest["settings"]["full_training_data"],
        "variants": list(VARIANTS), "rows": {key: len(value[0]) for key, value in arrays.items()},
        "selection": "Last epoch 20 only. Train-scene holdout is monitoring only; val never loaded for fitting.",
        "detector_checkpoint_sha256": BASELINE_SHA256, "environment": environment_info(),
    }
    from sdcl.evaluation.rescue_screen import GATES
    plan["fixed_gates_pp"] = GATES
    save_json(output / "plan.json", plan)
    for epoch in range(start, recipe.epochs):
        generator = torch.Generator().manual_seed(recipe.seed + epoch)
        order = torch.randperm(len(features), generator=generator).to(gpu)
        sums = {variant: 0.0 for variant in VARIANTS}
        totals = 0.0
        for offset in range(0, len(order), recipe.batch):
            chosen = order[offset:offset + recipe.batch]
            weight = importance[chosen]
            denominator = weight.sum()
            totals += float(denominator)
            for variant in VARIANTS:
                models[variant].train()
                optimizer = optimizers[variant]
                optimizer.zero_grad(set_to_none=True)
                logits = models[variant](training_inputs[variant][chosen])
                losses = F.cross_entropy(logits, targets[variant][chosen], reduction="none")
                loss = (losses * weight).sum() / denominator
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite rescoring loss.")
                loss.backward()
                optimizer.step()
                sums[variant] += float(loss.detach()) * float(denominator)
        row = {"epoch": epoch + 1}
        for variant in VARIANTS:
            row[f"{variant}_train_loss"] = sums[variant] / totals
            hx, hy, hw = arrays["holdout"]
            row[f"{variant}_holdout_loss"] = None
            if len(hx):
                models[variant].eval()
                numerator, denominator = 0.0, 0.0
                with torch.inference_mode():
                    for offset in range(0, len(hx), recipe.batch):
                        stop = offset + recipe.batch
                        x = torch.from_numpy(normalized_features(hx[offset:stop], mean, std, variant)).to(gpu)
                        y = torch.from_numpy(variant_labels(hy[offset:stop], variant)).to(gpu)
                        w = torch.from_numpy(hw[offset:stop]).to(gpu)
                        loss = F.cross_entropy(models[variant](x), y, reduction="none")
                        numerator += float((loss * w).sum())
                        denominator += float(w.sum())
                row[f"{variant}_holdout_loss"] = numerator / denominator
        logs.append(row)
        save_checkpoint(checkpoint_path, {
            "epoch": epoch + 1, "recipe": asdict(recipe),
            "cache_sha256": cache_hash, "cache_path": str(data),
            "full_training_data": manifest["settings"]["full_training_data"],
            "source_hashes": plan["source_hashes"],
            "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
            "models": {variant: {key: value.detach().cpu()
                                for key, value in model.state_dict().items()}
                       for variant, model in models.items()},
            "optimizers": {variant: optimizer.state_dict() for variant, optimizer in optimizers.items()},
            "logs": logs,
        })
        write_csv(output / "losses.csv", logs)
        print(json.dumps(row), flush=True)
    check_sources(plan["source_hashes"])
    return {"output": str(output), "checkpoint": str(checkpoint_path),
            "epoch": recipe.epochs, "full_training_data": plan["full_training_data"],
            "models": list(VARIANTS)}
