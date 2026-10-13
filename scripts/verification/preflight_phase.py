"""Dense real-data AMP probe; this does not train the 100-epoch candidate."""

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

from sdcl.config import SDCLConfig, load_experiment, load_phase_config, project_path
from sdcl.training.phase import PhaseConfig
from sdcl.training.trainer import SDCLDetectionModel, make_dataset


def main(output):
    destination = project_path(output)
    if destination.exists():
        raise FileExistsError(destination)
    arguments, _, _ = load_experiment("configs/experiments/phase_960_100.yaml")
    config = load_phase_config("configs/experiments/phase_960_100.yaml")
    args = get_cfg(overrides=arguments)
    data_yaml = Path(args.data)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_yaml.parent))
    dataset = make_dataset(args, str(data_yaml.parent / "images/train"), data, "train", 8, 32, True)
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset.labels[i]["cls"]),
                     reverse=True)[:8]
    init_seeds(0, deterministic=True)
    batch = dataset.collate_fn([dataset[i] for i in indices])
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["img"] = batch["img"].float() / 255
    results = {}
    for name, settings in (("baseline", PhaseConfig()), ("phase", config)):
        init_seeds(0, deterministic=True)
        pretrained = YOLO("yolo11s.pt").model
        model = SDCLDetectionModel(pretrained.yaml, nc=data["nc"], verbose=False)
        model.load(pretrained)
        del pretrained
        model.args = args
        model.sdcl_settings = SDCLConfig(enabled=False).to_dict()
        model.sdcl_ignore_regions = True
        model.phase_settings = settings.to_dict()
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
                raise RuntimeError("Non-finite dense-batch loss or gradients.")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            rows.append({"step": step, "seconds": perf_counter()-start, "loss": float(total.detach()),
                         "phase_loss": float(items.get("phase_loss", 0)),
                         "finite": finite})
        if name == "phase" and not (rows[0]["phase_loss"] > 0 and rows[1]["phase_loss"] == 0
                                     and rows[2]["phase_loss"] > 0):
            raise RuntimeError("Auxiliary interval scheduling is incorrect.")
        results[name] = {
            "steps": rows, "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "mean_update_seconds_excluding_warmup": sum(row["seconds"] for row in rows[1:])/2,
            "phase_statistics": getattr(model, "phase_statistics", {}),
        }
        del model, optimizer, scaler, loss, total, items
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "gpu": torch.cuda.get_device_name(0), "imgsz": 960, "batch": 8, "gt": len(batch["cls"]),
        "source_gt_counts": [len(dataset.labels[i]["cls"]) for i in indices], "runs": results,
        "update_time_ratio": results["phase"]["mean_update_seconds_excluding_warmup"] /
                             results["baseline"]["mean_update_seconds_excluding_warmup"],
        "scope": "Three AMP updates per method on one dense augmented batch; not a full run, "
                 "not a reliable epoch-time or general-memory bound.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/verification/phase_20261011/dense_amp.json")
    main(parser.parse_args().output)
