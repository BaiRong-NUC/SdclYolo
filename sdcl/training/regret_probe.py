"""Matched ranking-objective learnability probe; reuse frozen geometry cache only."""

from dataclasses import asdict, dataclass
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import HISTORY, checked_history, verify_dependencies
from sdcl.evaluation.size_analysis import file_hash, record_hash, write_csv
from sdcl.postprocess.localization import ActionScorer, normalized_actions, source_hashes as old_hashes
from sdcl.postprocess.rescue import save_json, scene_holdout
from sdcl.training.localization import DATA, read_action_cache, verify_archive
from sdcl.training.rescue import archive_source, checked_data, save_checkpoint


RUN = "output/runs/regret_objective_probe_960_seed0"
HEADROOM = "output/analysis/headroom-fixed-20261013"
HEADROOM_PLAN_SHA256 = "3833934c3c542d2a2e6c0f9432f7b8d1bb53abda73c8f60881efd32b38019bde"
HEADROOM_DECISION_SHA256 = "f0c0c90401d672e6bec5afcad7de151a05fc8235817765a588202a54c24a2150"
ARMS = ("uniform", "regret")


@dataclass(frozen=True)
class ProbeRecipe:
    version: str = "2026-10-13-regret-objective-probe-v1"
    seed: int = 0
    epochs: int = 20
    batch: int = 2048
    learning_rate: float = .002
    weight_decay: float = .0001
    temperature: float = .05
    selection_margin: float = .01


def load_recipe():
    actual = yaml.safe_load(project_path("configs/experiments/regret_probe_960.yaml").read_text())
    if actual != asdict(ProbeRecipe()):
        raise ValueError("Do not retune this frozen objective probe.")
    return ProbeRecipe()


def source_hashes():
    return {**old_hashes(), "configs/experiments/regret_probe_960.yaml":
            file_hash(project_path("configs/experiments/regret_probe_960.yaml"))}


def check_sources(expected):
    if source_hashes() != expected:
        raise ValueError("Probe source/recipe changed; use its own saved source snapshot.")


def checked_inputs(data, saved=None):
    history, _, _ = checked_history()
    path = project_path(HEADROOM)
    if (file_hash(path / "plan.json") != HEADROOM_PLAN_SHA256
            or file_hash(path / "decision.json") != HEADROOM_DECISION_SHA256):
        raise ValueError("Pinned headroom diagnosis changed.")
    headroom = json.loads((path / "plan.json").read_text())
    verify_dependencies(headroom["source_hashes"], path / "source.zip")
    decision = json.loads((path / "decision.json").read_text())
    if decision["decision"] != "HEADROOM_ONLY" or not decision["full_validation"]:
        raise ValueError("No verified headroom; do not start an objective probe.")
    original = torch.load(history["checkpoint"], map_location="cpu", weights_only=False)
    if original["epoch"] != 20 or original["source_hashes"] != history["source_hashes"]:
        raise ValueError("Frozen original head metadata mismatch.")
    data = project_path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    settings = manifest["settings"]
    if (not manifest["complete"] or settings["split"] != "train"
            or settings["recipe"] != original["recipe"]
            or settings["source_hashes"] != original["source_hashes"]
            or settings["control_sha256"] != original["control_sha256"]
            or len(manifest["chunks"]) != settings["images"]):
        raise ValueError("Cache is not the frozen training-only geometry recipe.")
    verify_archive(data, settings["source_hashes"], manifest["source_archive_sha256"])
    digest = file_hash(data / "manifest.json")
    if settings["full_training_data"] and (settings["images"] != 6471
            or digest != history["training_cache_sha256"]):
        raise ValueError("Formal probe must reuse the exact complete historical cache.")
    if saved is not None and (saved["cache_sha256"] != digest
            or saved["full_training_data"] != settings["full_training_data"]):
        raise ValueError("Probe checkpoint/cache mismatch.")
    for stem, chunk in manifest["chunks"].items():
        if (file_hash(data / chunk["file"]) != chunk["sha256"]
                or chunk["holdout"] != scene_holdout(stem)):
            raise ValueError("Frozen chunk content or holdout assignment changed.")
    _, _, records = checked_data()
    selected = [r for r in records if r["stem"] in manifest["chunks"]]
    if (record_hash(records) != settings["all_train_records_sha256"]
            or record_hash(selected) != settings["selected_records_sha256"]):
        raise ValueError("Training/selected records changed.")
    return manifest, original


def ranking_loss(outputs, targets, valid, importance, arm):
    if arm not in ARMS:
        raise ValueError("Unknown ranking objective.")
    temperature = ProbeRecipe().temperature
    delta = targets[:, :, None] - targets[:, None, :]
    gap = outputs[:, :, None] - outputs[:, None, :]
    n = targets.shape[1]
    pairs = (valid[:, :, None] & valid[:, None, :] & (delta.abs() > 1e-6)
             & torch.triu(torch.ones(n, n, dtype=torch.bool, device=outputs.device), diagonal=1))
    pair_weight = pairs.to(outputs.dtype)
    if arm == "regret":
        pair_weight = pair_weight * delta.abs()
    loss = F.softplus(-delta.sign() * gap / temperature)
    positive = (loss * pair_weight).sum((1, 2)) / pair_weight.sum((1, 2)).clamp_min(1e-12)
    # Unowned/duplicate FP anchors have no IoU preference: both arms learn to retain keep.
    fp_loss = F.softplus((outputs[:, 1:] - outputs[:, :1]) / temperature)
    negative = (fp_loss * valid[:, 1:]).sum(1) / valid[:, 1:].sum(1).clamp_min(1)
    per_anchor = torch.where(targets[:, 0] > 0, positive, negative)
    return (per_anchor * importance).sum() / importance.sum()


def train_probe(data=DATA, output=RUN, device="0", resume=False):
    recipe = load_recipe()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True)
    data, output = project_path(data), project_path(output)
    path = output / "last.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False) if resume else None
    manifest, original = checked_inputs(data, saved)
    if saved is not None:
        check_sources(saved["source_hashes"])
        verify_archive(output, saved["source_hashes"], saved["source_archive_sha256"])
        if saved["recipe"] != asdict(recipe) or not 1 <= saved["epoch"] <= recipe.epochs:
            raise ValueError("Probe resume recipe/epoch mismatch.")
    elif output.exists():
        raise FileExistsError(f"Run exists; use --resume: {output}")
    else:
        output.mkdir(parents=True)
        archive_source(output, source_hashes())
    arrays = read_action_cache(data, manifest)
    mean, std = original["mean"].numpy(), original["std"].numpy()
    gpu = torch.device(f"cuda:{device}")
    feature, valid, target, weight = arrays["train"]
    x = torch.from_numpy(normalized_actions(feature, mean, std)).to(gpu)
    v, y, w = (torch.from_numpy(value).to(gpu) for value in (valid, target, weight))
    initial = ActionScorer().state_dict()
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm] = ActionScorer().to(gpu)
        models[arm].load_state_dict(initial)
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(),
            lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        if saved is not None:
            models[arm].load_state_dict(saved["models"][arm])
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
    from sdcl.evaluation.regret_probe import GATES
    plan = {"recipe": asdict(recipe), "source_hashes": source_hashes(),
        "source_archive_sha256": file_hash(output / "source.zip"),
        "cache_path": str(data), "cache_sha256": file_hash(data / "manifest.json"),
        "full_training_data": manifest["settings"]["full_training_data"],
        "headroom_plan_sha256": HEADROOM_PLAN_SHA256, "fixed_gates": GATES,
        "rows": {name: len(a[0]) for name, a in arrays.items()},
        "selection": "Epoch20 only; train-scene holdout loss monitors only. No val fitting/selection.",
        "scope": "Matched objective learnability probe; no new signal, detector inference or detector training. "
                 "Pairwise regret weighting is not asserted novel; prior localization STOP remains unchanged."}
    save_json(output / "plan.json", plan)
    logs = saved["logs"] if saved is not None else []
    for epoch in range(saved["epoch"] if saved is not None else 0, recipe.epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(recipe.seed + epoch)).to(gpu)
        sums, total = dict.fromkeys(ARMS, 0.), 0.
        for start in range(0, len(order), recipe.batch):
            chosen = order[start:start + recipe.batch]
            denominator = float(w[chosen].sum())
            total += denominator
            for arm in ARMS:
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                loss = ranking_loss(models[arm](x[chosen]), y[chosen], v[chosen], w[chosen], arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite ranking loss.")
                loss.backward()
                optimizer.step()
                sums[arm] += float(loss.detach()) * denominator
        row = {"epoch": epoch + 1}
        for arm in ARMS:
            row[f"{arm}_train_loss"] = sums[arm] / total
            row[f"{arm}_holdout_loss"] = None
            hx, hv, hy, hw = arrays["holdout"]
            if len(hx):
                numerator, denominator = 0., 0.
                with torch.inference_mode():
                    for start in range(0, len(hx), recipe.batch):
                        stop = start + recipe.batch
                        bx = torch.from_numpy(normalized_actions(hx[start:stop], mean, std)).to(gpu)
                        bv, by, bw = (torch.from_numpy(value[start:stop]).to(gpu) for value in (hv, hy, hw))
                        loss = ranking_loss(models[arm](bx), by, bv, bw, arm)
                        numerator += float(loss) * float(bw.sum())
                        denominator += float(bw.sum())
                row[f"{arm}_holdout_loss"] = numerator / denominator
        logs.append(row)
        save_checkpoint(path, {**plan, "epoch": epoch + 1, "mean": original["mean"], "std": original["std"],
            "models": {arm: {k: value.detach().cpu() for k, value in model.state_dict().items()}
                       for arm, model in models.items()},
            "optimizers": {arm: o.state_dict() for arm, o in optimizers.items()}, "logs": logs})
        write_csv(output / "losses.csv", logs)
        print(json.dumps(row), flush=True)
    check_sources(plan["source_hashes"])
    return {"output": str(output), "epoch": recipe.epochs, "models": list(ARMS),
            "full_training_data": plan["full_training_data"], "detector_training_ran": False}
