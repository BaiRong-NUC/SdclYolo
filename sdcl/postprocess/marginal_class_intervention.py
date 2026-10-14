"""Prediction-only action features and exact training-only marginal TP labels."""

import numpy as np
import torch
from torch import nn

from sdcl.evaluation.size_analysis import box_iou_matrix, match_at_threshold


FEATURE_DIM = 61
LOCAL_DIM = 4


class InterventionScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def features(baseline, hard_score, soft_score, size):
    n = len(baseline)
    wh = baseline[:, 2:4] - baseline[:, :2]
    score = np.clip(baseline[:, 4], 1e-6, 1 - 1e-6)
    original = baseline[:, 5].astype(int)
    basic = np.column_stack((
        score, np.log(score / (1 - score)),
        np.log1p(np.sqrt(wh.prod(1)) * 640 / max(size) / 16),
        np.log(wh[:, 0] / wh[:, 1]),
        (baseline[:, :2] + baseline[:, 2:4]) / 2 / size,
        np.eye(10)[original],
    ))
    local = np.zeros((n, 10, LOCAL_DIM), np.float32)
    overlap = box_iou_matrix(baseline[:, :4], baseline[:, :4])
    order = np.argsort(-baseline[:, 4], kind="stable")
    position = np.empty(n, int)
    position[order] = np.arange(n)
    stronger = (position[None, :] < position[:, None]) & (baseline[None, :, 4] >= .25)
    near = (overlap >= .5) & stronger
    support = np.zeros((n, 10), np.float32)
    count = np.zeros_like(support)
    for cls in range(10):
        available = near & (original[None, :] == cls)
        support[:, cls] = np.max(np.where(available, overlap * baseline[None, :, 4], 0),
                                 axis=1, initial=0)
        count[:, cls] = np.log1p(available.sum(1))
    local[:, :, 0], local[:, :, 1] = support, count
    local[:, :, 2] = support[np.arange(n), original, None]
    local[:, :, 3] = count[np.arange(n), original, None]
    evidence = [
        np.stack((s, np.log(np.clip(s, 1e-6, 1)), s >= .05), 3).reshape(n, 10, 15)
        for s in (hard_score, soft_score)
    ]
    result = np.concatenate((
        np.repeat(basic[:, None], 10, axis=1),
        np.broadcast_to(np.eye(10), (n, 10, 10)),
        (np.arange(10)[None, :] == original[:, None])[:, :, None],
        *evidence, local,
    ), 2).astype(np.float32)
    if result.shape != (n, 10, FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid class intervention features.")
    return result


def marginal_targets(ious, gt_classes, baseline, valid):
    """Only the two affected classes need rematching for a single label change."""
    original = baseline[:, 5].astype(int)
    result = np.zeros(valid.shape, np.float32)
    classes = np.asarray(gt_classes)
    matched, _ = match_at_threshold(ious, classes, baseline, .25)
    base_count = np.bincount(classes[matched >= 0], minlength=10)
    for row in np.flatnonzero((baseline[:, 4] >= .25) & (valid.sum(1) > 1)):
        old = original[row]
        for new in np.flatnonzero(valid[row]):
            if new == old:
                continue
            gt_mask = (classes == old) | (classes == new)
            dt_mask = (original == old) | (original == new)
            subset = baseline[dt_mask].copy()
            subset[np.flatnonzero(np.flatnonzero(dt_mask) == row)[0], 5] = new
            rematched, _ = match_at_threshold(ious[gt_mask][:, dt_mask], classes[gt_mask], subset, .25)
            result[row, new] = int((rematched >= 0).sum()) - base_count[old] - base_count[new]
    return result


def choose(outputs, baseline, valid, arm):
    utility = np.where(valid, outputs, -np.inf)
    original = baseline[:, 5].astype(int)
    rows = np.arange(len(baseline))
    if arm == "class_ce":
        probability = np.exp(utility - utility.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        best = probability.argmax(1)
        gain = probability[rows, best] - probability[rows, original]
    else:
        gain_by_class = utility - utility[rows, original, None]
        best = gain_by_class.argmax(1)
        gain = gain_by_class[rows, best]
    return np.where((baseline[:, 4] >= .25) & (gain > .1), best, original)
