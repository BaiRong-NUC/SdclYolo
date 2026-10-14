"""Prediction-only cross-class view evidence; scores/coordinates remain fixed."""

import numpy as np

from sdcl.evaluation.size_analysis import box_iou_matrix
from sdcl.postprocess.rescue import clean_views


def class_evidence(candidates, views, windows, size):
    if len(views) != 5 or len(windows) != 5:
        raise ValueError("Expected the fixed five-view observation policy.")
    candidates = np.asarray(candidates, float).reshape(-1, 6)
    n = len(candidates)
    score = np.zeros((n, 10, 5), np.float32)
    raw_score = np.zeros_like(score)
    clean, origins = clean_views(views, size, [])
    for view in range(5):
        others = clean[origins == view]
        overlaps = box_iou_matrix(candidates[:, :4], others[:, :4])
        for cls in range(10):
            members = np.flatnonzero((others[:, 5] == cls) & (others[:, 4] >= .05))
            if not len(members):
                continue
            eligible = overlaps[:, members] >= .7
            raw = np.where(eligible, others[None, members, 4], 0)
            raw_score[:, cls, view] = raw.max(1)
            score[:, cls, view] = (raw * overlaps[:, members]).max(1)
    valid = (score > 0).any(2)
    original = candidates[:, 5].astype(int)
    valid[np.arange(n), original] = True
    width_height = candidates[:, 2:4] - candidates[:, :2]
    area = width_height.prod(1)
    context = np.zeros((n, 5, 4), np.float32)
    center = (candidates[:, :2] + candidates[:, 2:4]) / 2
    expanded_low, expanded_high = center - width_height, center + width_height
    for view, window in enumerate(windows):
        low, high = np.asarray(window[:2]), np.asarray(window[2:])
        visible = np.maximum(np.minimum(candidates[:, 2:4], high) -
                             np.maximum(candidates[:, :2], low), 0).prod(1)
        expanded = np.maximum(np.minimum(expanded_high, high) -
                              np.maximum(expanded_low, low), 0).prod(1)
        context[:, view, 0] = visible / np.maximum(area, 1e-12)
        context[:, view, 1] = np.log1p(np.sqrt(area) * 960 / max(high - low) / 8)
        context[:, view, 2] = expanded / np.maximum(area * 4, 1e-12)
        context[:, view, 3] = view == 0
    return valid, score, raw_score, context


def vote_labels(candidates, valid, score):
    original = candidates[:, 5].astype(int)
    utility = np.where(valid, score.sum(2), -np.inf)
    best = utility.argmax(1)
    peak = utility[np.arange(len(best)), best]
    keep = utility[np.arange(len(best)), original]
    return np.where(keep >= peak, original, best)


def apply_labels(candidates, labels):
    labels = np.asarray(labels)
    if (labels.shape != (len(candidates),) or not np.isfinite(labels).all() or
            np.any((labels < 0) | (labels >= 10) | (labels != np.floor(labels)))):
        raise ValueError("Invalid class actions.")
    values = candidates.copy()
    values[:, 5] = labels
    if not np.array_equal(values[:, :5], candidates[:, :5]):
        raise ValueError("Class choice changed original boxes/scores.")
    return values
