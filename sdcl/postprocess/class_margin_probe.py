"""Frozen last-projection learnability screen; not a detection novelty claim."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.localization import frozen_control
from sdcl.postprocess.rescue import scene_holdout


CONFIG = "configs/experiments/class_margin_probe_960.yaml"
CONTROL = "output/analysis/visdrone-official-control-20261013/corrected_control_manifest.json"
CONTROL_SHA = "59e5657a3d3d62705015281d7c2c9ca17406fbcf1783944ced87bc80e79e005d"
CACHE = "output/class_margin/train_visual_seed0"
RUN = "output/runs/class_margin_probe_960_seed0"
ARMS = ("bce", "margin")
GATES = {
    "official_ap_vs_control_pp": (">=", .30),
    "official_ap_vs_bce_pp": (">=", .20),
    "tiny_ap_vs_control_pp": (">=", .50),
    "small_all_ap_vs_control_pp": (">=", .30),
    "small_all_recall_fppi5_pp": (">=", -.10),
    "small_all_recall_fppi10_pp": (">=", -.10),
    "official_ar500_vs_control_pp": (">=", -.10),
    "owned_tiny_correct_class_vs_control_pp": (">=", 1.0),
    "owned_tiny_correct_class_vs_bce_pp": (">=", .50),
    "added_processing_mean_ms": ("<=", 10.0),
}


@dataclass(frozen=True)
class Recipe:
    version: str = "2026-10-13-frozen-visual-class-margin-v1"
    seed: int = 0
    fit_images: int = 512
    monitor_images: int = 128
    epochs: int = 10
    batch: int = 4096
    learning_rate: float = .0002
    weight_decay: float = .0001
    positive_per_level: int = 24
    negative_per_level: int = 16
    margin: float = .2
    margin_weight: float = .25
    retention_weight: float = .01
    tiny_reference_side: float = 16.


def recipe():
    value = yaml.safe_load(project_path(CONFIG).read_text())
    if value != asdict(Recipe()):
        raise ValueError("Frozen classification probe changed; do not retune after results.")
    return Recipe()


def checked_control():
    frozen_control()
    path = project_path(CONTROL)
    if file_hash(path) != CONTROL_SHA:
        raise ValueError("Corrected strong control manifest changed.")
    value = json.loads(path.read_text())
    for item in (value["prediction_capture"], value["source_archive"]):
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Corrected control predictions/source changed.")
    evaluation = value["evaluation"]
    if file_hash(evaluation["official_report"]) != evaluation["sha256"]:
        raise ValueError("Corrected official control report changed.")
    return value


def sources():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             *project_path("sdcl/evaluation/native").glob("*.cc"), project_path(CONFIG)]
    return {p.relative_to(project_path(".")).as_posix(): file_hash(p) for p in sorted(paths)}


def check_sources(expected):
    if sources() != expected:
        raise ValueError("Probe code differs from its source snapshot.")


def archive_source(output, hashes):
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(project_path(name), name)


def verify_archive(output, hashes, digest):
    if file_hash(output / "source.zip") != digest:
        raise ValueError("Probe source archive changed.")
    with zipfile.ZipFile(output / "source.zip") as archive:
        if set(archive.namelist()) != set(hashes) or any(
            hashlib.sha256(archive.read(n)).hexdigest() != v for n, v in hashes.items()
        ):
            raise ValueError("Probe archived code does not match its plan.")


def select_records(records, limit=None):
    fixed = recipe()
    def key(r):
        return hashlib.sha256(f"class-probe:{fixed.seed}:{r['stem']}".encode()).hexdigest()
    fit = sorted((r for r in records if not scene_holdout(r["stem"])), key=key)
    monitor = sorted((r for r in records if scene_holdout(r["stem"])), key=key)
    n_fit, n_monitor = (fixed.fit_images, fixed.monitor_images) if limit is None else (limit, limit)
    if len(fit) < n_fit or len(monitor) < n_monitor:
        raise ValueError("Insufficient scene-separated training images.")
    chosen = [(r, False) for r in fit[:n_fit]] + [(r, True) for r in monitor[:n_monitor]]
    if {r['stem'].split('_')[0] for r, h in chosen if h} & {
        r['stem'].split('_')[0] for r, h in chosen if not h
    }:
        raise AssertionError("Scene overlap in fit/monitor split.")
    return chosen


class Projection(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight.float().clone())
        self.bias = nn.Parameter(bias.float().clone())

    def forward(self, features, levels):
        output = features.new_empty((len(features), 10))
        for level in range(3):
            mask = levels == level
            output[mask] = F.linear(features[mask], self.weight[level], self.bias[level])
        return output


def objective(logits, targets, tiny, original, arm):
    fixed = recipe()
    if arm not in ARMS:
        raise ValueError("Unknown classification objective.")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / targets.sum().clamp_min(1)
    loss = bce + fixed.retention_weight * (logits - original).square().mean()
    active = tiny & (targets.sum(-1) > 0)
    if arm == "margin" and active.any():
        labels = targets[active].argmax(-1)
        values = logits[active]
        true = values.gather(1, labels[:, None]).squeeze(1)
        wrong = values.masked_fill(F.one_hot(labels, 10).bool(), -torch.inf).amax(-1)
        quality = targets[active].sum(-1)
        term = F.softplus(fixed.margin + wrong - true)
        loss = loss + fixed.margin_weight * (term * quality).sum() / quality.sum().clamp_min(1e-6)
    return loss


def decide(values, complete):
    if set(values) != set(GATES) or not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Missing/non-finite fixed probe metrics.")
    checks = []
    for name, (operator, threshold) in GATES.items():
        value = float(values[name])
        passed = value >= threshold if operator == ">=" else value <= threshold
        checks.append(dict(metric=name, value=value, comparison=operator, threshold=threshold, passed=bool(passed)))
    return {
        "decision": ("CONTINUE" if all(c["passed"] for c in checks) else "STOP") if complete else "SMOKE_ONLY",
        "complete_fixed_probe": complete, "checks": checks,
        "stop_rule": "End this frozen-feature last-classification-layer recipe if STOP. Do not tune subset/seed/margin/epochs or choose another checkpoint.",
        "scope": "Only a bounded learnability/investment probe. A STOP does not falsify all small-object classification methods. A pass permits independent test-dev verification, not a novelty claim or automatic full training.",
        "prior_stops": "Localization/action-ranking, CPR-TAL and phase consistency remain stopped.",
    }
