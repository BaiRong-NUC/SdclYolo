"""Verify coverage repair on training images and dense AMP updates, without full training."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _bootstrap
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds

from sdcl.config import load_experiment, load_coverage_config, project_path
from sdcl.training.coverage import CoverageAssigner, CoverageConfig
from sdcl.runtime.environment import environment_info
from sdcl.evaluation.phase_screen import BASELINE
from sdcl.evaluation.size_analysis import file_hash
from sdcl.training.trainer import SDCLDetectionModel, make_dataset


def main(output, samples):
    destination = project_path(output)
    if destination.exists():
        raise FileExistsError(destination)
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    arguments, sdcl, _ = load_experiment("configs/experiments/coverage_960_100.yaml")
    config = load_coverage_config("configs/experiments/coverage_960_100.yaml")
    args = get_cfg(overrides=arguments)
    data_path = Path(args.data)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    init_seeds(0, deterministic=True)
    dataset = make_dataset(args, str(data_path.parent / "images/train"), data, "train", 8, 32, True)
    model = YOLO(str(project_path(BASELINE))).model.float().cuda().eval()
    model.args = args
    model.coverage_settings = config.to_dict()
    model.criterion = None
    criterion = model.init_criterion()
    criterion.epoch = 20
    indices = [round(i * (len(dataset) - 1) / (samples - 1)) for i in range(samples)]
    for n, index in enumerate(indices):
        batch = dataset.collate_fn([dataset[index]])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float() / 255
        with torch.no_grad():
            criterion.get_assigned_targets_and_loss(model(batch["img"])[1], batch)
        if n % 16 == 0:
            print("Training-image assignment probe:", n, flush=True)
    probe = dict(criterion.assigner.statistics)
    if probe.get("rescued_gt", 0) <= 0 or probe["donor_starvation"]:
        raise RuntimeError("Repair inactive or donor coverage violated.")
    del model, criterion, batch
    gc.collect()
    torch.cuda.empty_cache()
    init_seeds(0, deterministic=True)
    dense_indices = sorted(range(len(dataset)), key=lambda i: len(dataset.labels[i]["cls"]),
                           reverse=True)[:8]
    batch = dataset.collate_fn([dataset[i] for i in dense_indices])
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["img"] = batch["img"].float() / 255
    results = {}
    for name, settings in (("baseline", CoverageConfig()), ("coverage", config)):
        init_seeds(0, deterministic=True)
        pretrained = YOLO("yolo11s.pt").model
        model = SDCLDetectionModel(pretrained.yaml, nc=data["nc"], verbose=False)
        model.load(pretrained)
        del pretrained
        model.args = args
        model.sdcl_settings = sdcl.to_dict()
        model.sdcl_ignore_regions = True
        model.coverage_settings = settings.to_dict()
        model.sdcl_epoch = 20
        model = model.cuda().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.937)
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        rows = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                loss, items = model(batch)
                total = loss.sum()
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            finite = bool(torch.isfinite(total)) and all(
                bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if not finite:
                raise RuntimeError("Non-finite dense AMP loss or gradients.")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            rows.append({"step": step, "seconds": perf_counter() - start,
                         "loss": float(total.detach()), "finite": finite})
        assigner = model.criterion.assigner
        stats = dict(assigner.statistics) if isinstance(assigner, CoverageAssigner) else {}
        results[name] = {
            "steps": rows, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "mean_update_seconds_excluding_warmup": sum(row["seconds"] for row in rows[1:]) / 2,
            "assignment_statistics": stats,
        }
        if name == "coverage" and stats.get("donor_starvation", 0):
            raise RuntimeError("Dense-batch donor coverage invariant violated.")
        del model, optimizer, scaler, assigner, loss, total, items
        gc.collect()
        torch.cuda.empty_cache()
    result = {"gpu": torch.cuda.get_device_name(0), "imgsz": 960, "batch": 8,
              "source_hashes": environment_info()["sdcl_source_hashes"],
              "coverage_config": config.to_dict(),
              "baseline_checkpoint_sha256": file_hash(project_path(BASELINE)),
              "train_only_probe_images": samples, "train_probe_indices": indices,
              "assignment_probe": probe, "dense_gt": len(batch["cls"]),
              "source_gt_counts": [len(dataset.labels[i]["cls"]) for i in dense_indices],
              "runs": results,
              "update_time_ratio": results["coverage"]["mean_update_seconds_excluding_warmup"]
                                   / results["baseline"]["mean_update_seconds_excluding_warmup"],
              "scope": "Training images only; finite AMP and same-logit assignment coverage, "
                       "not AP, seed robustness, epoch-time guarantee, or novelty evidence."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/verification/coverage_20261012/preflight.json")
    parser.add_argument("--samples", type=int, default=64)
    main(**vars(parser.parse_args()))
