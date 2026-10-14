"""Ordinary bounded coordinate residual control for native feature screening."""

import numpy as np
import torch
from torch import nn


class NativeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(132, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return .1*torch.tanh(self.network(features))


def correct_coordinates(reference, residual, scale, active, size):
    if residual.shape != (len(reference), 4) or scale.shape != residual.shape:
        raise ValueError("Coordinate residual and original box scale must align.")
    if not np.isfinite(residual).all() or np.any(np.abs(residual) > .100001):
        raise ValueError("Coordinate residual is not finite and bounded.")
    result = reference.copy()
    rows = np.flatnonzero(active & (residual != 0).any(1))
    changed = reference[rows, :4].astype(float)+residual[rows]*scale[rows]
    width, height = size
    changed[:, [0, 2]] = np.clip(changed[:, [0, 2]], 0, width)
    changed[:, [1, 3]] = np.clip(changed[:, [1, 3]], 0, height)
    valid = (changed[:, 2:] > changed[:, :2]).all(1)
    result[rows[valid], :4] = changed[valid]
    return result
