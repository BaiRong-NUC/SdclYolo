"""Global-prediction-only features for selecting an uncomputed local view."""

import numpy as np
from torch import nn

from sdcl.evaluation.size_analysis import box_areas


def global_window_features(predictions, windows, size):
    values = np.asarray(predictions, float).reshape(-1, 6)
    values = values[np.isfinite(values).all(1)]
    if len(windows) != 5:
        raise ValueError("Exactly one global and four local windows required.")
    center = (values[:, :2] + values[:, 2:4]) / 2
    area = box_areas(values[:, :4]) * (640 / max(size)) ** 2
    masks = (np.ones(len(values), bool), area < 16 ** 2, area < 32 ** 2)
    thresholds = (.001, .05, .25, .5)
    inside = [((center >= w[:2]) & (center < w[2:])).all(1) for w in windows]

    def counts(mask):
        return [np.log1p((mask & group & (values[:, 4] >= score)).sum())
                for group in masks for score in thresholds]

    context = counts(inside[0])
    local_counts = [np.log1p((mask & (values[:, 4] >= score)).sum())
                    for mask in inside[1:] for score in (.05, .25)]
    rows = []
    for view, window in enumerate(windows[1:], 1):
        w = np.asarray(window, float)
        geometry = [*(w / np.tile(size, 2)),
                    np.log(max(size) / max(w[2:] - w[:2])),
                    *np.eye(4)[view - 1]]
        active = inside[view] & (values[:, 4] >= .05)
        moments = []
        for group in masks:
            scores = values[active & group, 4]
            moments.extend([float(scores.mean()) if len(scores) else 0,
                            float(scores.max()) if len(scores) else 0])
        classes = [np.log1p((active & (values[:, 5] == cls)).sum()) for cls in range(10)]
        contact = [
            np.log1p((active & (values[:, axis] < w[axis])).sum()) for axis in (0, 1)]
        contact += [
            np.log1p((active & (values[:, axis] > w[axis])).sum()) for axis in (2, 3)]
        rows.append([*geometry, *counts(inside[view]), *moments, *classes,
                     *contact, *context, *local_counts])
    features = np.asarray(rows, np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite view-value features.")
    return features


class ViewValueRegressor(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimensions, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)
