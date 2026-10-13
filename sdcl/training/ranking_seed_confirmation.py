"""Matched absolute/ranking heads: repeat two seeds without detector training."""

from dataclasses import asdict, dataclass
import json
import os

import numpy as np
import torch
import yaml

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.postprocess.localization import ActionScorer, normalized_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.localization import DATA, read_action_cache, regression_loss
from sdcl.training.regret_probe import checked_inputs, ranking_loss
from sdcl.training.rescue import archive_source, save_checkpoint


CONFIG = "configs/experiments/ranking_seed_confirmation_960.yaml"
REGISTRATION = "docs/960_ranking_seed_confirmation_registration_20261013.md"
RUN = "output/runs/ranking_seed_confirmation_960_20261013"
ARMS = ("absolute", "uniform", "regret")


@dataclass(frozen=True)
class SeedRecipe:
    version: str = "2026-10-13-ranking-seed-confirmation-v1"
    seeds: tuple = (1, 2)
    epochs: int = 20
    batch: int = 2048
    learning_rate: float = .002
    weight_decay: float = .0001
    temperature: float = .05
    selection_margin: float = .01


def recipe():
    fixed = SeedRecipe()
    expected = json.loads(json.dumps(asdict(fixed)))
    if yaml.safe_load(project_path(CONFIG).read_text()) != expected:
        raise ValueError("Do not change this registered seed confirmation.")
    return fixed


def objective(outputs, targets, valid, weights, arm):
    if arm == "absolute":
        return regression_loss(outputs, targets, valid, weights, arm)
    return ranking_loss(outputs, targets, valid, weights, arm)


def matched_models(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = ActionScorer().state_dict()
    models = {arm: ActionScorer().to(device) for arm in ARMS}
    for model in models.values():
        model.load_state_dict(initial)
    return models


def sources():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             project_path(CONFIG), project_path(REGISTRATION)]
    return {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}


def train(data=DATA, output=RUN, device="0"):
    fixed = recipe()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, output = project_path(data), project_path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a registered run.")
    manifest, original = checked_inputs(data)
    if not manifest["settings"]["full_training_data"]:
        raise ValueError("Use the complete original train cache.")
    arrays = read_action_cache(data, manifest)
    output.mkdir(parents=True)
    hashes = sources()
    archive_source(output, hashes)
    mean, std = original["mean"].numpy(), original["std"].numpy()
    gpu = torch.device(f"cuda:{device}")
    features, valid, targets, weights = arrays["train"]
    x = torch.from_numpy(normalized_actions(features, mean, std)).to(gpu)
    v, y, w = (torch.from_numpy(a).to(gpu) for a in (valid, targets, weights))
    plan = {
        "recipe": json.loads(json.dumps(asdict(fixed))), "source_hashes": hashes,
        "source_archive": str(output / "source.zip"),
        "source_archive_sha256": file_hash(output / "source.zip"),
        "cache_path": str(data), "cache_sha256": file_hash(data / "manifest.json"),
        "registration_sha256": file_hash(project_path(REGISTRATION)),
        "full_training_data": True, "rows": {k: len(a[0]) for k, a in arrays.items()},
        "arms": ARMS, "selection": "Only epoch20; holdout is monitoring only.",
        "scope": "Two head-training seeds; frozen detector, normalization and action menu.",
    }
    save_json(output / "plan.json", plan)
    for seed in fixed.seeds:
        run = output / f"seed{seed}"
        run.mkdir()
        models = matched_models(seed, gpu)
        optimizers = {a: torch.optim.AdamW(model.parameters(), lr=fixed.learning_rate,
                                          weight_decay=fixed.weight_decay)
                      for a, model in models.items()}
        logs = []
        for epoch in range(fixed.epochs):
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch)).to(gpu)
            sums, total = dict.fromkeys(ARMS, 0.), 0.
            for start in range(0, len(order), fixed.batch):
                chosen = order[start:start + fixed.batch]
                denominator = float(w[chosen].sum())
                total += denominator
                for arm in ARMS:
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(models[arm](x[chosen]), y[chosen], v[chosen], w[chosen], arm)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite matched loss.")
                    loss.backward()
                    optimizer.step()
                    sums[arm] += float(loss.detach()) * denominator
            row = {"seed": seed, "epoch": epoch + 1}
            hx, hv, hy, hw = arrays["holdout"]
            for arm in ARMS:
                row[arm + "_train_loss"] = sums[arm] / total
                numerator, denominator = 0., 0.
                with torch.inference_mode():
                    for start in range(0, len(hx), fixed.batch):
                        stop = start + fixed.batch
                        bx = torch.from_numpy(normalized_actions(hx[start:stop], mean, std)).to(gpu)
                        bv, by, bw = (torch.from_numpy(a[start:stop]).to(gpu) for a in (hv, hy, hw))
                        loss = objective(models[arm](bx), by, bv, bw, arm)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite holdout loss.")
                        numerator += float(loss) * float(bw.sum())
                        denominator += float(bw.sum())
                row[arm + "_holdout_loss"] = numerator / denominator if denominator else None
            logs.append(row)
            save_checkpoint(run / "last.pt", {
                **plan, "seed": seed, "epoch": epoch + 1, "mean": original["mean"],
                "std": original["std"],
                "models": {a: {k: p.detach().cpu() for k, p in model.state_dict().items()}
                           for a, model in models.items()},
                "optimizers": {a: opt.state_dict() for a, opt in optimizers.items()}, "logs": logs,
            })
            write_csv(run / "losses.csv", logs)
            print(json.dumps(row), flush=True)
    verify_dependencies(hashes, output / "source.zip")
    checked_inputs(data)
    save_json(output / "verification.json", {
        "full_train_cache_unchanged": True, "source_unchanged": True,
        "matched_initialization_batches_capacity": True, "only_final_epoch": True,
        "no_detector_training_or_inference": True,
    })
    return {"output": str(output), "seeds": fixed.seeds, "epoch": fixed.epochs,
            "detector_training": False}
