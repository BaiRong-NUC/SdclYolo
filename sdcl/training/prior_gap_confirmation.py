"""An existing IoU-gap ranking shape adapted as an ordinary action-task control."""

import json
import os

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from sdcl.config import project_path
from sdcl.evaluation.headroom_screen import verify_dependencies
from sdcl.evaluation.size_analysis import file_hash, write_csv
from sdcl.postprocess.localization import normalized_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.localization import DATA, read_action_cache
from sdcl.training.ranking_seed_confirmation import matched_models
from sdcl.training.regret_probe import checked_inputs
from sdcl.training.rescue import archive_source, save_checkpoint


CONFIG = "configs/experiments/prior_gap_confirmation_960.yaml"
REGISTRATION = "docs/960_ranking_nearest_prior_and_next_screen_20261013.md"
RUN = "output/runs/prior_gap_confirmation_960_20261013"
REPEAT = "output/analysis/ranking-seeds-fixed-20261013"
FIXED = {
    "version": "2026-10-13-prior-gap-confirmation-v1", "seeds": [0, 1, 2],
    "epochs": 20, "batch": 2048, "learning_rate": .002, "weight_decay": .0001,
    "alpha": 1., "fp_temperature": .05, "selection_margin": .01,
}


def recipe():
    if yaml.safe_load(project_path(CONFIG).read_text()) != FIXED:
        raise ValueError("Registered prior-gap control changed.")
    return dict(FIXED)


def prior_gap_loss(outputs, targets, valid, importance):
    delta = targets[:, :, None] - targets[:, None, :]
    gap = outputs[:, :, None] - outputs[:, None, :]
    n = targets.shape[1]
    pairs = (valid[:, :, None] & valid[:, None, :] & (delta.abs() > 1e-6)
             & torch.triu(torch.ones(n, n, dtype=torch.bool, device=outputs.device), diagonal=1))
    # Exclude invalid pairs before exp, so a masked score cannot overflow the loss.
    exponent = torch.where(pairs, -delta * gap, torch.zeros_like(gap))
    positive = (exponent.exp() * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
    fp_loss = F.softplus((outputs[:, 1:] - outputs[:, :1]) / .05)
    negative = (fp_loss * valid[:, 1:]).sum(1) / valid[:, 1:].sum(1).clamp_min(1)
    per_anchor = torch.where(targets[:, 0] > 0, positive, negative)
    return (per_anchor * importance).sum() / importance.sum()


def checked_repeat():
    root = project_path(REPEAT)
    decision = json.loads((root / "decision.json").read_text())
    if (decision["ranking_task"] != "PROMISING_HEAD_SEED_REPEAT"
            or not all(json.loads((root / "verification.json").read_text()).values())):
        raise ValueError("Only proceed after the complete registered ranking seed screen passes.")
    from sdcl.evaluation.ranking_seed_confirmation import seed_decision
    if seed_decision(json.loads((root / "official_metrics.json").read_text())) != decision:
        raise ValueError("Ranking seed decision/report mismatch.")
    for name, digest in json.loads((root / "input_manifest.json").read_text()).items():
        if file_hash(root / name) != digest:
            raise ValueError("Seed-repeat official input changed.")
    return root


def train(data=DATA, output=RUN, device="0"):
    fixed = recipe()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Use an available CUDA index.")
    repeat = checked_repeat()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, output = project_path(data), project_path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a registered prior-gap run.")
    manifest, original = checked_inputs(data)
    if not manifest["settings"]["full_training_data"]:
        raise ValueError("Use the complete train action cache.")
    arrays = read_action_cache(data, manifest)
    output.mkdir(parents=True)
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             project_path(CONFIG), project_path(REGISTRATION)]
    hashes = {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}
    archive_source(output, hashes)
    mean, std = original["mean"].numpy(), original["std"].numpy()
    gpu = torch.device(f"cuda:{device}")
    features, valid, targets, weights = arrays["train"]
    x = torch.from_numpy(normalized_actions(features, mean, std)).to(gpu)
    v, y, w = (torch.from_numpy(a).to(gpu) for a in (valid, targets, weights))
    plan = {
        "recipe": fixed, "source_hashes": hashes,
        "source_archive_sha256": file_hash(output / "source.zip"),
        "cache_path": str(data), "cache_sha256": file_hash(data / "manifest.json"),
        "registration_sha256": file_hash(project_path(REGISTRATION)),
        "repeat_inputs_sha256": {n: file_hash(repeat / n) for n in
                                 ("plan.json", "official_metrics.json", "decision.json",
                                  "verification.json", "input_manifest.json")},
        "full_training_data": True, "rows": {k: len(a[0]) for k, a in arrays.items()},
        "selection": "Only epoch20; train-scene holdout monitors only.",
        "scope": "Adapted positive loss shape only; not a RankDetNet reproduction or novel loss.",
    }
    save_json(output / "plan.json", plan)
    for seed in fixed["seeds"]:
        run = output / f"seed{seed}"
        run.mkdir()
        model = matched_models(seed, gpu)["uniform"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=fixed["learning_rate"],
                                      weight_decay=fixed["weight_decay"])
        logs = []
        for epoch in range(fixed["epochs"]):
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch)).to(gpu)
            numerator, denominator = 0., 0.
            for start in range(0, len(order), fixed["batch"]):
                chosen = order[start:start + fixed["batch"]]
                optimizer.zero_grad(set_to_none=True)
                loss = prior_gap_loss(model(x[chosen]), y[chosen], v[chosen], w[chosen])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite prior-gap loss; do not silently clip or retune.")
                loss.backward()
                optimizer.step()
                numerator += float(loss.detach()) * float(w[chosen].sum())
                denominator += float(w[chosen].sum())
            row = {"seed": seed, "epoch": epoch + 1, "train_loss": numerator / denominator}
            hx, hv, hy, hw = arrays["holdout"]
            numerator, denominator = 0., 0.
            with torch.inference_mode():
                for start in range(0, len(hx), fixed["batch"]):
                    stop = start + fixed["batch"]
                    bx = torch.from_numpy(normalized_actions(hx[start:stop], mean, std)).to(gpu)
                    bv, by, bw = (torch.from_numpy(a[start:stop]).to(gpu) for a in (hv, hy, hw))
                    loss = prior_gap_loss(model(bx), by, bv, bw)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite holdout prior-gap loss.")
                    numerator += float(loss) * float(bw.sum())
                    denominator += float(bw.sum())
            row["holdout_loss"] = numerator / denominator if denominator else None
            logs.append(row)
            save_checkpoint(run / "last.pt", {
                **plan, "seed": seed, "epoch": epoch + 1, "mean": original["mean"],
                "std": original["std"], "model": {k: p.detach().cpu() for k, p in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "logs": logs,
            })
            write_csv(run / "losses.csv", logs)
            print(json.dumps(row), flush=True)
    verify_dependencies(hashes, output / "source.zip")
    checked_inputs(data)
    if any(file_hash(repeat / n) != h for n, h in plan["repeat_inputs_sha256"].items()):
        raise ValueError("The completed repeat changed during training.")
    save_json(output / "verification.json", {
        "complete_train_cache_unchanged": True, "source_unchanged": True,
        "all_three_matched_seeds": True, "only_final_epoch": True,
        "no_detector_training_or_inference": True, "repeat_condition_verified": True,
    })
    return {"output": str(output), "seeds": fixed["seeds"], "epoch": fixed["epochs"]}
