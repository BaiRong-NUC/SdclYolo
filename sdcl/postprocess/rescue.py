"""View-conditioned rescue rescoring on a frozen Soft-NMS candidate set."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch
from torch import nn
import yaml

from sdcl.config import project_path
from sdcl.evaluation.fusion_screen import FusionPolicy
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.size_analysis import box_iou_matrix, match_at_threshold
from sdcl.runtime.files import atomic_replace


VARIANTS = ("calibration", "no_observability", "rescue")
STRONG = "output/analysis/fusion-softnms-fixed-20261013"
RAW_VAL_SHA256 = "9bbdfc46ce419edeb8324ae54ba90d3e45a96460e00a1bfe23a382ab0235aae2"


@dataclass(frozen=True)
class RescueRecipe:
    version: str = "2026-10-13-vcr-v1"
    seed: int = 0
    epochs: int = 20
    batch: int = 2048
    hidden: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    samples_per_state_per_image: int = 32
    support_confidence: float = 0.25
    opportunity_coverage: float = 0.90
    opportunity_side: float = 8.0
    score_logit_gain: float = 1.0
    holdout_scene_modulus: int = 10
    reference_size: int = 640
    classes: int = 10


def load_recipe(path="configs/experiments/rescue_softnms_960.yaml"):
    saved = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
    if saved != asdict(RescueRecipe()):
        raise ValueError("Recipe differs from the frozen screen. Do not tune after seeing results.")
    return RescueRecipe()


def scene_holdout(stem):
    scene = stem.split("_", 1)[0]
    digest = hashlib.sha256(scene.encode()).digest()
    return int.from_bytes(digest[:8], "little") % RescueRecipe().holdout_scene_modulus == 0


def clean_views(views, size, ignores):
    """Keep source-view identity through the exact strong-control ignore filter."""
    arrays = [np.asarray(view, dtype=np.float32).reshape(-1, 6) for view in views]
    values = np.concatenate(arrays).copy()
    origins = np.concatenate([np.full(len(view), index, dtype=np.int64)
                              for index, view in enumerate(arrays)])
    finite = np.isfinite(values).all(axis=1)
    values, origins = values[finite], origins[finite]
    width, height = size
    values[:, [0, 2]] = values[:, [0, 2]].clip(0, width)
    values[:, [1, 3]] = values[:, [1, 3]].clip(0, height)
    valid = ((values[:, 2] > values[:, 0]) & (values[:, 3] > values[:, 1])
             & (values[:, 4] >= 0.001) & (values[:, 4] <= 1)
             & (values[:, 5] >= 0) & (values[:, 5] == np.floor(values[:, 5])))
    values, origins = values[valid], origins[valid]
    tensor = torch.from_numpy(values.copy())
    ignored = prediction_ignore_mask(
        tensor[:, :4], torch.as_tensor(np.asarray(ignores), dtype=torch.float32).reshape(-1, 4), 0.5)
    keep = ~ignored.numpy()
    return values[keep], origins[keep]


def soft_nms_with_origins(views, size, ignores):
    """Same Gaussian algorithm as the strong control, with a parallel provenance array."""
    values, origins = clean_views(views, size, ignores)
    pending = values.astype(float)
    selected, source_ids, original_scores = [], [], []
    policy = FusionPolicy()
    raw_scores = pending[:, 4].copy()
    while len(pending) and len(selected) < policy.max_det:
        index = int(np.argmax(pending[:, 4]))
        best = pending[index].copy()
        selected.append(best)
        source_ids.append(origins[index])
        original_scores.append(raw_scores[index])
        pending = np.delete(pending, index, axis=0)
        origins = np.delete(origins, index)
        raw_scores = np.delete(raw_scores, index)
        same = pending[:, 5] == best[5]
        other = pending[same]
        low = np.maximum(best[:2], other[:, :2])
        high = np.minimum(best[2:4], other[:, 2:4])
        intersection = np.maximum(high - low, 0).prod(axis=1)
        area = (best[2] - best[0]) * (best[3] - best[1])
        other_area = (other[:, 2] - other[:, 0]) * (other[:, 3] - other[:, 1])
        iou = intersection / np.maximum(area + other_area - intersection, 1e-12)
        pending[same, 4] *= np.exp(-(iou * iou) / policy.sigma)
        keep = pending[:, 4] >= policy.score_floor
        pending, origins, raw_scores = pending[keep], origins[keep], raw_scores[keep]
    return (np.asarray(selected, dtype=float).reshape(-1, 6),
            np.asarray(source_ids, dtype=np.int64), np.asarray(original_scores, dtype=float))


def opportunity(boxes, window):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1, y1, x2, y2 = window
    area = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0).prod(axis=1)
    low = np.maximum(boxes[:, :2], [x1, y1])
    high = np.minimum(boxes[:, 2:4], [x2, y2])
    fraction = np.maximum(high - low, 0).prod(axis=1) / np.maximum(area, 1e-12)
    side = np.sqrt(area) * 960 / max(x2 - x1, y2 - y1)
    recipe = RescueRecipe()
    visible = (fraction >= recipe.opportunity_coverage) & (side >= recipe.opportunity_side)
    return visible, fraction, side


def feature_layout():
    names = [
        "soft_score", "soft_logit", "raw_score", "soft_raw_ratio",
        "log_reference_side", "log_aspect", "center_x", "center_y",
        "is_global", "log_view_zoom", "own_border_x", "own_border_y",
        *[f"class_{index}" for index in range(10)],
    ]
    censored = []
    for view in range(5):
        censored.extend(range(len(names), len(names) + 3))
        names.extend(f"peer_{view}_{key}" for key in (
            "opportunity", "coverage", "log_input_side", "max_same_iou",
            "same_score_iou05", "same_score_iou01"))
    censored.extend([len(names), len(names) + 1])
    names.extend(["peer_opportunities", "unsupported_opportunities"])
    return names, censored


def candidate_features(candidates, origins, raw_scores, views, windows, size, ignores):
    """Features depend on predictions and view geometry only, never annotations."""
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 6)
    n = len(candidates)
    if len(views) != 5 or len(windows) != 5 or len(origins) != n:
        raise ValueError("Exactly five views and aligned candidate origins are required.")
    if n == 0:
        return np.empty((0, len(feature_layout()[0])), dtype=np.float32)
    if np.any(candidates[:, 5] >= 10):
        raise ValueError("Frozen ten-class recipe mismatch.")
    boxes = candidates[:, :4]
    widths, heights = (boxes[:, 2:4] - boxes[:, :2]).T
    side = np.sqrt(widths * heights)
    score = np.clip(candidates[:, 4], 1e-6, 1 - 1e-6)
    windows_array = np.asarray(windows, dtype=float)
    own = windows_array[origins]
    own_width, own_height = own[:, 2] - own[:, 0], own[:, 3] - own[:, 1]
    center = (boxes[:, :2] + boxes[:, 2:4]) / 2
    base = np.column_stack([
        score, np.log(score / (1 - score)), raw_scores,
        score / np.maximum(raw_scores, 1e-6),
        np.log1p(side * 640 / max(size) / 16),
        np.log(np.maximum(widths / heights, 1e-6)),
        center[:, 0] / size[0], center[:, 1] / size[1],
        origins == 0, np.log(max(size) / np.maximum(own_width, own_height)),
        np.minimum(boxes[:, 0] - own[:, 0], own[:, 2] - boxes[:, 2]) / own_width,
        np.minimum(boxes[:, 1] - own[:, 1], own[:, 3] - boxes[:, 3]) / own_height,
        np.eye(10)[candidates[:, 5].astype(int)],
    ])
    clean, view_ids = clean_views(views, size, ignores)
    blocks, available_total, unsupported_total = [], np.zeros(n), np.zeros(n)
    for view, window in enumerate(windows):
        available, fraction, input_side = opportunity(boxes, window)
        max_iou, strong, weak = np.zeros(n), np.zeros(n), np.zeros(n)
        for class_id in range(10):
            query = np.flatnonzero(candidates[:, 5] == class_id)
            other = clean[(view_ids == view) & (clean[:, 5] == class_id)]
            if not len(query) or not len(other):
                continue
            ious = box_iou_matrix(boxes[query], other[:, :4])
            max_iou[query] = ious.max(axis=1)
            strong[query] = np.where(ious >= 0.5, other[None, :, 4], 0).max(axis=1)
            weak[query] = np.where(ious >= 0.1, other[None, :, 4], 0).max(axis=1)
        is_peer = origins != view
        block = np.column_stack([available, fraction, np.log1p(input_side / 8),
                                 max_iou, strong, weak])
        block[~is_peer] = 0
        blocks.append(block)
        available_total += available & is_peer
        unsupported_total += available & is_peer & (strong < RescueRecipe().support_confidence)
    result = np.column_stack([base, *blocks, available_total / 4, unsupported_total / 4])
    if not np.isfinite(result).all():
        raise ValueError("Non-finite rescoring features.")
    return result.astype(np.float32)


def rescue_labels(record, candidates, origins, views, windows, ignores):
    """Train-only labels: final FP, supported TP, observable rescue, sole observer."""
    objects = record["objects"]
    gt = np.asarray([obj["box_xyxy"] for obj in objects], dtype=float).reshape(-1, 4)
    classes = np.asarray([obj["class_id"] for obj in objects])
    ious = box_iou_matrix(gt, candidates[:, :4])
    matched, _ = match_at_threshold(ious, classes, candidates, 0.001)
    labels = np.zeros(len(candidates), dtype=np.int64)
    clean, source_ids = clean_views(views, record["size"], ignores)
    recipe = RescueRecipe()
    for gt_index in np.flatnonzero(matched >= 0):
        row = matched[gt_index]
        has_opportunity, has_support = False, False
        for view, window in enumerate(windows):
            if view == origins[row]:
                continue
            visible = opportunity(gt[gt_index:gt_index + 1], window)[0][0]
            has_opportunity |= bool(visible)
            other = clean[(source_ids == view) & (clean[:, 5] == classes[gt_index])
                          & (clean[:, 4] >= recipe.support_confidence)]
            if len(other):
                has_support |= bool((box_iou_matrix(gt[gt_index:gt_index + 1],
                                                   other[:, :4]) >= 0.5).any())
        labels[row] = 1 if has_support else 2 if has_opportunity else 3
    return labels


def sample_candidates(labels, stem):
    seed = int.from_bytes(hashlib.sha256(stem.encode()).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    indices, weights = [], []
    for state in range(4):
        pool = np.flatnonzero(labels == state)
        if not len(pool):
            continue
        count = min(len(pool), RescueRecipe().samples_per_state_per_image)
        chosen = np.sort(generator.choice(pool, count, replace=False))
        indices.extend(chosen)
        weights.extend([len(pool) / count] * count)
    return np.asarray(indices, dtype=np.int64), np.asarray(weights, dtype=np.float32)


class RescueScorer(nn.Module):
    def __init__(self):
        super().__init__()
        features = len(feature_layout()[0])
        hidden = RescueRecipe().hidden
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4),
        )

    def forward(self, features):
        return self.network(features)


def normalized_features(features, mean, std, variant):
    result = ((features - mean) / std).clip(-6, 6)
    if variant == "no_observability":
        result[:, feature_layout()[1]] = 0
    return result


def variant_labels(labels, variant):
    result = labels.copy()
    if variant == "calibration":
        result[result > 0] = 1
    elif variant == "no_observability":
        result[result == 3] = 2
    return result


def score_correction(candidates, probabilities, variant):
    values = np.asarray(candidates, dtype=float).copy()
    if probabilities.shape != (len(values), 4):
        raise ValueError("Prediction/probability shape mismatch.")
    if variant == "calibration":
        residual = probabilities[:, 1] - probabilities[:, 0]
    else:
        residual = probabilities[:, 2] + probabilities[:, 3] - probabilities[:, 0]
    confidence = np.clip(values[:, 4], 1e-6, 1 - 1e-6)
    logits = np.log(confidence / (1 - confidence))
    logits += RescueRecipe().score_logit_gain * residual
    values[:, 4] = 1 / (1 + np.exp(-logits))
    values = values[values[:, 4] >= 0.001]
    return values[np.argsort(-values[:, 4], kind="stable")][:500]


def source_hashes():
    from sdcl.evaluation.size_analysis import file_hash
    paths = sorted(project_path("sdcl").rglob("*.py"))
    paths += sorted(project_path("scripts").rglob("*.py"))
    paths += [project_path("configs/experiments/rescue_softnms_960.yaml")]
    return {path.relative_to(project_path(".")).as_posix(): file_hash(path) for path in paths}


def check_sources(expected):
    if source_hashes() != expected:
        raise ValueError("Experiment source/recipe changed. Restore the saved version before continuing.")


def save_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    atomic_replace(temporary, path)
