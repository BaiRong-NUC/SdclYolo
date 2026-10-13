"""Anchor-referenced localization gain selection on frozen calibrated detections."""

from dataclasses import asdict, dataclass
import hashlib
import json
import zipfile

import numpy as np
import torch
from torch import nn
import yaml

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash, match_at_threshold
from sdcl.postprocess.rescue import (
    RescueScorer, candidate_features, clean_views, normalized_features, soft_nms_with_origins,
)


CONTROL = "output/analysis/calibration-control-supplement-20261013/control_manifest.json"
CONTROL_SHA256 = "43cb0fe9b451a22fc3bb49171b532560cc2b2d9f9b52b836e39e2159799cf5f8"
VARIANTS = ("absolute", "gain")
ACTIONS = 6  # Keep, one best peer from each of four other views, weighted coordinate vote.


@dataclass(frozen=True)
class LocalizationRecipe:
    version: str = "2026-10-13-anchor-gain-v1"
    seed: int = 0
    epochs: int = 20
    batch: int = 2048
    hidden: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    samples_per_tp_fp_per_image: int = 64
    peer_iou: float = 0.5
    peer_confidence: float = 0.05
    gain_margin: float = 0.01
    smooth_l1_beta: float = 0.1
    reference_size: int = 640


def load_recipe():
    saved = yaml.safe_load(project_path("configs/experiments/localization_gain_960.yaml").read_text())
    if saved != asdict(LocalizationRecipe()):
        raise ValueError("Localization recipe changed; do not tune this fixed screen.")
    return LocalizationRecipe()


def frozen_control():
    path = project_path(CONTROL)
    if file_hash(path) != CONTROL_SHA256:
        raise ValueError("Frozen calibration control manifest changed.")
    control = json.loads(path.read_text(encoding="utf-8"))
    if (control["scorer"]["model_branch"] != "calibration" or control["scorer"]["epoch"] != 20
            or file_hash(control["scorer"]["path"]) != control["scorer"]["sha256"]
            or file_hash(control["detector"]["path"]) != control["detector"]["sha256"]
            or file_hash(control["prediction_capture"]["path"]) != control["prediction_capture"]["sha256"]):
        raise ValueError("Frozen control checkpoint/predictions changed.")
    # New modules may be added, but none of the original replay dependencies may change.
    for name, digest in control["source_hashes"].items():
        if file_hash(project_path(name)) != digest:
            raise ValueError(f"Frozen replay dependency changed: {name}")
    with zipfile.ZipFile(control["source_archive"]) as archive:
        if (file_hash(control["source_archive"]) != control["source_archive_sha256"]
                or set(archive.namelist()) != set(control["source_hashes"])
                or any(hashlib.sha256(archive.read(name)).hexdigest() != digest
                       for name, digest in control["source_hashes"].items())):
            raise ValueError("Frozen replay source archive changed.")
    return control


def load_calibrator(control):
    saved = torch.load(control["scorer"]["path"], map_location="cpu", weights_only=False)
    if saved["epoch"] != 20 or saved["source_hashes"] != control["source_hashes"]:
        raise ValueError("Calibration checkpoint provenance mismatch.")
    torch.set_num_threads(1)
    model = RescueScorer().eval()
    model.load_state_dict(saved["models"]["calibration"])
    return model, saved["mean"].numpy(), saved["std"].numpy()


def calibrated_candidates(views, windows, size, ignores, calibrator):
    candidates, origins, raw_scores = soft_nms_with_origins(views, size, ignores)
    features = candidate_features(candidates, origins, raw_scores, views, windows, size, ignores)
    model, mean, std = calibrator
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(
            normalized_features(features, mean, std, "calibration"))).softmax(1).numpy()
    values = candidates.copy()
    confidence = np.clip(values[:, 4], 1e-6, 1 - 1e-6)
    logits = np.log(confidence / (1 - confidence))
    logits += 1.0 * (probabilities[:, 1] - probabilities[:, 0])
    values[:, 4] = 1 / (1 + np.exp(-logits))
    keep = np.flatnonzero(values[:, 4] >= 0.001)
    order = keep[np.argsort(-values[keep, 4], kind="stable")][:500]
    actual = values[order]
    return actual, origins[order], raw_scores[order], features[order]


def action_feature_names():
    from sdcl.postprocess.rescue import feature_layout
    return [*feature_layout()[0], "dx1", "dy1", "dx2", "dy2", "iou_anchor",
            "source_score", *[f"view_{i}" for i in range(5)], "is_vote",
            "peer_count", "calibrated_score"]


def action_menu(candidates, origins, raw_scores, base_features, views, size, ignores):
    """Prediction-only proposals; each peer slot follows ascending view id excluding own."""
    n = len(candidates)
    boxes = np.repeat(candidates[:, None, :4], ACTIONS, axis=1)
    valid = np.zeros((n, ACTIONS), dtype=bool)
    valid[:, 0] = True
    view_id = np.full((n, ACTIONS), -1, dtype=int)
    view_id[:, 0] = origins
    scores = np.zeros((n, ACTIONS))
    scores[:, 0] = raw_scores
    clean, source_ids = clean_views(views, size, ignores)
    recipe = load_recipe()
    for peer in range(5):
        for cls in range(10):
            query = np.flatnonzero((origins != peer) & (candidates[:, 5] == cls))
            other = clean[(source_ids == peer) & (clean[:, 5] == cls)
                          & (clean[:, 4] >= recipe.peer_confidence)]
            if not len(query) or not len(other):
                continue
            ious = box_iou_matrix(candidates[query, :4], other[:, :4])
            # Highest IoU wins; equal IoUs favor higher score then original source order.
            priority = np.argsort(-other[:, 4], kind="stable")
            chosen = priority[np.argmax(ious[:, priority], axis=1)]
            available = ious[np.arange(len(query)), chosen] >= recipe.peer_iou
            rows, members = query[available], other[chosen[available]]
            # Slot order is ascending source view, excluding the anchor's own view.
            actions = np.where(peer < origins[rows], peer + 1, peer)
            boxes[rows, actions] = members[:, :4]
            scores[rows, actions] = members[:, 4]
            view_id[rows, actions] = peer
            valid[rows, actions] = True
    weights = scores[:, :5] * valid[:, :5]
    vote = (boxes[:, :5] * weights[:, :, None]).sum(1) / np.maximum(weights.sum(1)[:, None], 1e-12)
    has_peer = valid[:, 1:5].any(1)
    boxes[has_peer, 5] = vote[has_peer]
    valid[:, 5] = has_peer
    scores[:, 5] = np.where(has_peer, weights.sum(1) / np.maximum(valid[:, :5].sum(1), 1), 0)
    width_height = candidates[:, 2:4] - candidates[:, :2]
    scale = np.column_stack([width_height, width_height])
    relative = (boxes - candidates[:, None, :4]) / np.maximum(scale[:, None], 1e-6)
    overlap_low = np.maximum(boxes[:, :, :2], candidates[:, None, :2])
    overlap_high = np.minimum(boxes[:, :, 2:], candidates[:, None, 2:4])
    intersection = np.maximum(overlap_high - overlap_low, 0).prod(2)
    area = np.maximum(boxes[:, :, 2:] - boxes[:, :, :2], 0).prod(2)
    anchor_area = width_height.prod(1)
    ious = intersection / np.maximum(area + anchor_area[:, None] - intersection, 1e-12)
    onehot = np.zeros((n, ACTIONS, 5))
    row, action = np.nonzero(view_id >= 0)
    onehot[row, action, view_id[row, action]] = 1
    features = np.concatenate([
        np.repeat(base_features[:, None], ACTIONS, axis=1), relative,
        ious[:, :, None], scores[:, :, None], onehot,
        np.broadcast_to(np.arange(ACTIONS) == 5, (n, ACTIONS))[:, :, None],
        np.repeat((valid[:, 1:5].sum(1) / 4)[:, None, None], ACTIONS, axis=1),
        np.repeat(candidates[:, 4, None, None], ACTIONS, axis=1),
    ], axis=2).astype(np.float32)
    if features.shape != (n, ACTIONS, len(action_feature_names())) or not np.isfinite(features).all():
        raise ValueError("Invalid localization action features.")
    return boxes, valid, features


def action_targets(record, candidates, boxes):
    """Train-only: compare each action to keep against the same baseline-owned GT."""
    objects = record["objects"]
    gt = np.asarray([obj["box_xyxy"] for obj in objects], dtype=float).reshape(-1, 4)
    classes = np.asarray([obj["class_id"] for obj in objects])
    matched, _ = match_at_threshold(box_iou_matrix(gt, candidates[:, :4]), classes, candidates, 0.001)
    targets = np.zeros((len(candidates), ACTIONS), dtype=np.float32)
    owned = np.zeros(len(candidates), dtype=bool)
    for gt_index in np.flatnonzero(matched >= 0):
        row = matched[gt_index]
        targets[row] = box_iou_matrix(gt[gt_index:gt_index+1], boxes[row])[0]
        owned[row] = True
    return targets, owned


def sample_anchors(owned, stem):
    import hashlib
    generator = np.random.default_rng(int.from_bytes(hashlib.sha256(stem.encode()).digest()[:8], "little"))
    indices, weights = [], []
    for state in (False, True):
        pool = np.flatnonzero(owned == state)
        count = min(len(pool), LocalizationRecipe().samples_per_tp_fp_per_image)
        if count:
            indices.extend(np.sort(generator.choice(pool, count, replace=False)))
            weights.extend([len(pool) / count] * count)
    return np.asarray(indices, dtype=np.int64), np.asarray(weights, dtype=np.float32)


class ActionScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(action_feature_names()), LocalizationRecipe().hidden), nn.SiLU(),
            nn.Linear(LocalizationRecipe().hidden, LocalizationRecipe().hidden), nn.SiLU(),
            nn.Linear(LocalizationRecipe().hidden, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def select_actions(outputs, valid, variant):
    outputs = np.asarray(outputs, dtype=float)
    if outputs.shape != valid.shape or not np.isfinite(outputs).all():
        raise ValueError("Invalid action outputs.")
    if variant == "vote":
        return np.where(valid[:, 5], 5, 0)
    if variant == "absolute":
        gains = outputs - outputs[:, :1]
    elif variant in ("gain", "gain_always"):
        gains = outputs.copy()
        gains[:, 0] = 0
    else:
        raise ValueError(f"Unknown variant: {variant}")
    proposal = np.where(valid[:, 1:], gains[:, 1:], -np.inf)
    best = np.argmax(proposal, axis=1) + 1
    available = valid[:, 1:].any(1)
    if variant != "gain_always":
        available &= gains[np.arange(len(best)), best] > LocalizationRecipe().gain_margin
    return np.where(available, best, 0)


def normalized_actions(features, mean, std):
    return np.clip((features - mean) / std, -6, 6).astype(np.float32)


def apply_actions(candidates, boxes, selected):
    values = candidates.copy()
    values[:, :4] = boxes[np.arange(len(values)), selected]
    if (not np.isfinite(values).all()
            or np.any(values[:, 2:4] <= values[:, :2])
            or not np.array_equal(values[:, 4:], candidates[:, 4:])):
        raise ValueError("Action changed scores/classes or produced an invalid box.")
    return values


def source_hashes():
    paths = [*project_path("sdcl").rglob("*.py"), *project_path("scripts").rglob("*.py"),
             project_path("configs/experiments/localization_gain_960.yaml"),
             project_path("configs/experiments/rescue_softnms_960.yaml")]
    return {path.relative_to(project_path(".")).as_posix(): file_hash(path) for path in sorted(paths)}


def check_sources(expected):
    if expected != source_hashes():
        raise ValueError("Localization source/recipe changed; use the saved source snapshot.")
