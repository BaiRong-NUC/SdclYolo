"""Matched ten-threshold supervision probe on the historical train-only cache."""

from dataclasses import asdict, dataclass
import json
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.postprocess.localization import normalized_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.localization import DATA, read_action_cache
from sdcl.training.regret_probe import checked_inputs
from sdcl.training.rescue import archive_source, save_checkpoint


CONFIG = "configs/experiments/threshold_gain_probe_960.yaml"
RUN = "output/runs/threshold_gain_probe_960_seed0"
ARMS = ("threshold_absolute", "threshold_gain")


@dataclass(frozen=True)
class Recipe:
    version: str = "2026-10-13-threshold-gain-probe-v1"
    seed: int = 0
    epochs: int = 20
    batch: int = 2048
    hidden: int = 64
    learning_rate: float = .002
    weight_decay: float = .0001
    smooth_l1_beta: float = .1
    selection_margin: float = .01
    threshold_start: float = .5
    threshold_step: float = .05
    threshold_count: int = 10


def recipe():
    fixed = Recipe()
    if yaml.safe_load(project_path(CONFIG).read_text()) != asdict(fixed):
        raise ValueError("Registered threshold recipe changed; no retuning.")
    return fixed


class ThresholdScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(68, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10)
        )

    def forward(self, features):
        return self.network(features)


def threshold_targets(ious, arm):
    if arm not in ARMS:
        raise ValueError("Unknown threshold supervision.")
    thresholds = torch.arange(10, device=ious.device, dtype=ious.dtype) * .05 + .5
    targets = (ious[..., None] >= thresholds).to(ious.dtype)
    return targets if arm == "threshold_absolute" else targets - targets[:, :1]


def threshold_loss(outputs, ious, valid, weights, arm):
    targets = threshold_targets(ious, arm)
    losses = F.smooth_l1_loss(outputs, targets, reduction="none", beta=.1).mean(-1)
    per_anchor = (losses * valid).sum(1) / valid.sum(1).clamp_min(1)
    return (per_anchor * weights).sum() / weights.sum()


def sources():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             project_path(CONFIG)]
    return {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}


def train(data=DATA, output=RUN, device="0"):
    fixed = recipe()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(fixed.seed)
    torch.cuda.manual_seed_all(fixed.seed)
    torch.use_deterministic_algorithms(True)
    data, output = project_path(data), project_path(output)
    if output.exists():
        raise FileExistsError("This registered one-shot probe must not overwrite/retrain a run.")
    manifest, original = checked_inputs(data)
    if not manifest["settings"]["full_training_data"]:
        raise ValueError("The threshold probe requires the original complete train cache.")
    arrays = read_action_cache(data, manifest)
    output.mkdir(parents=True)
    hashes = sources()
    archive_source(output, hashes)
    gpu = torch.device(f"cuda:{device}")
    mean, std = original["mean"].numpy(), original["std"].numpy()
    feature, valid, target, weight = arrays["train"]
    x = torch.from_numpy(normalized_actions(feature, mean, std)).to(gpu)
    v, y, w = (torch.from_numpy(a).to(gpu) for a in (valid, target, weight))
    initial = ThresholdScorer().state_dict()
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = ThresholdScorer().to(gpu)
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(
            models[arm].parameters(), lr=fixed.learning_rate, weight_decay=fixed.weight_decay
        )
    plan = {
        "recipe": asdict(fixed), "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "cache_path": str(data), "cache_sha256": file_hash(data / "manifest.json"),
        "full_training_data": True, "rows": {k: len(a[0]) for k, a in arrays.items()},
        "parameters_per_head": sum(p.numel() for p in models[ARMS[0]].parameters()),
        "selection": "Fixed epoch20, monitor train-scene holdout only; no val/test label fitting.",
        "registration_sha256": file_hash(project_path("docs/960_threshold_gain_probe_registration_20261013.md")),
        "normalization": "Exact frozen localization train-fitting mean/std.",
    }
    save_json(output / "plan.json", plan)
    logs = []
    for epoch in range(fixed.epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(epoch)).to(gpu)
        sums, total = dict.fromkeys(ARMS, 0.), 0.
        for start in range(0, len(order), fixed.batch):
            chosen = order[start:start + fixed.batch]
            denominator = float(w[chosen].sum())
            total += denominator
            for arm in ARMS:
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                loss = threshold_loss(models[arm](x[chosen]), y[chosen], v[chosen], w[chosen], arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite threshold loss.")
                loss.backward()
                optimizer.step()
                sums[arm] += float(loss.detach()) * denominator
        row = {"epoch": epoch + 1}
        hx, hv, hy, hw = arrays["holdout"]
        for arm in ARMS:
            row[arm + "_train_loss"] = sums[arm] / total
            numerator, denominator = 0., 0.
            with torch.inference_mode():
                for start in range(0, len(hx), fixed.batch):
                    stop = start + fixed.batch
                    bx = torch.from_numpy(normalized_actions(hx[start:stop], mean, std)).to(gpu)
                    bv, by, bw = (torch.from_numpy(a[start:stop]).to(gpu) for a in (hv, hy, hw))
                    loss = threshold_loss(models[arm](bx), by, bv, bw, arm)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite holdout threshold loss.")
                    numerator += float(loss) * float(bw.sum())
                    denominator += float(bw.sum())
            row[arm + "_holdout_loss"] = numerator / denominator if denominator else None
        logs.append(row)
        save_checkpoint(output / "last.pt", {
            **plan, "epoch": epoch + 1, "mean": original["mean"], "std": original["std"],
            "models": {a: {k: p.detach().cpu() for k, p in model.state_dict().items()}
                       for a, model in models.items()},
            "optimizers": {a: opt.state_dict() for a, opt in optimizers.items()}, "logs": logs,
        })
        write_csv(output / "losses.csv", logs)
        print(json.dumps(row), flush=True)
    verify_dependencies(hashes, output / "source.zip")
    checked_inputs(data)
    return {"output": str(output), "epoch": fixed.epochs,
            "parameters_per_head": plan["parameters_per_head"], "detector_training": False}
