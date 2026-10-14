"""Deterministic convex whole-box fusion under fitted error second moments."""

from itertools import combinations

import numpy as np


def source_order(features, mean, std, columns, boxes, valid):
    templates = np.clip((np.eye(5)-mean[columns])/std[columns], -6, 6)
    observed = features[:, :5, columns]
    distance = ((observed[:, :, None]-templates[None, None])**2).sum(-1)
    source = distance.argmin(-1)
    available = valid[:, :5]
    if np.any(distance.min(-1)[available] > 1e-8):
        raise ValueError("Native source identity does not exactly match the frozen one-hot encoding.")
    ordered = np.zeros((len(boxes), 5, 4))
    supported = np.zeros((len(boxes), 5), bool)
    for action in range(5):
        rows = np.flatnonzero(available[:, action])
        view = source[rows, action]
        if supported[rows, view].any():
            raise ValueError("A native source appears more than once.")
        ordered[rows, view] = boxes[rows, action]
        supported[rows, view] = True
    return ordered, supported


def simplex_minimum(matrix):
    """Enumerate active supports of a five-variable convex quadratic."""
    matrix = np.asarray(matrix, float)
    size = len(matrix)
    best_value, best_weights = np.inf, None
    for count in range(1, size+1):
        for support in combinations(range(size), count):
            support = np.asarray(support, int)
            sub = matrix[np.ix_(support, support)]
            inverse_one = np.linalg.solve(sub, np.ones(len(support)))
            weights = inverse_one / inverse_one.sum()
            if np.any(weights < -1e-10):
                continue
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            full = np.zeros(size)
            full[support] = weights
            value = float(full @ matrix @ full)
            if value < best_value:
                best_value, best_weights = value, full
    if best_weights is None:
        raise ValueError("No feasible convex fusion weights.")
    return best_weights


def weight_table(matrices, diagonal=False):
    table = np.zeros((3, 32, 5))
    for group, matrix in enumerate(matrices):
        if diagonal:
            matrix = np.diag(np.diag(matrix))
        for mask in range(1, 32):
            views = np.flatnonzero(mask & (1 << np.arange(5)))
            table[group, mask, views] = simplex_minimum(matrix[np.ix_(views, views)])
    return table


def fuse(candidates, native, available, groups, table, baseline):
    result = baseline.copy()
    active = (candidates[:, 4] >= .05) & (available.sum(1) >= 2)
    masks = (available * (1 << np.arange(5))).sum(1)
    weights = table[groups[active], masks[active]]
    coordinates = (weights[:, :, None] * native[active]).sum(1)
    if not np.isfinite(coordinates).all() or not (coordinates[:, 2:] > coordinates[:, :2]).all():
        raise ValueError("Convex whole-box fusion produced invalid coordinates.")
    result[active, :4] = coordinates
    if not np.array_equal(result[:, 4:], baseline[:, 4:]):
        raise ValueError("Fusion changed scores or classes.")
    return result
