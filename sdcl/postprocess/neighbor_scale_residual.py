"""Scalar regret residual with matched local/global scale-context channels."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ScaleResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(72, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def context_channels(context, reference, global_context=False):
    score = context["global_scale" if global_context else "local_scale"]
    difference = score-score[np.arange(len(reference)), reference, None]
    count = context["global_peer_count"] if global_context else (context["neighbors"] >= 0).sum(1)
    result = np.stack((score, difference, np.broadcast_to(np.log1p(count)[:, None], score.shape),
                       np.broadcast_to(context["supported"][:, None], score.shape)), -1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Invalid frozen scale-context channels.")
    return result


def derange_blocks(values, seed):
    n = len(values)
    if n < 2:
        raise ValueError("At least2 anchors needed for a context derangement.")
    order = np.random.default_rng(seed).permutation(n)
    partner = np.empty(n, int)
    partner[order] = np.roll(order, 1)
    return values[partner], partner


def objective(utility, ious, valid, owned, reference, pairs):
    delta = ious[:, pairs[:, 0]]-ious[:, pairs[:, 1]]
    comparisons = valid[:, pairs[:, 0]] & valid[:, pairs[:, 1]] & (delta.abs() > .01)
    signed = delta.sign()*(utility[:, pairs[:, 0]]-utility[:, pairs[:, 1]])
    positive = (F.softplus(-signed/.05)*comparisons).sum(1)/comparisons.sum(1).clamp_min(1)
    row = torch.arange(len(utility), device=utility.device)
    other = valid.clone()
    other.scatter_(1, reference[:, None], False)
    negative = (F.softplus((utility-utility[row, reference, None])/.05)*other).sum(1)
    negative /= other.sum(1).clamp_min(1)
    active = torch.where(owned, comparisons.any(1), other.any(1))
    losses = torch.where(owned, positive, negative)
    return (losses*active).sum()/active.sum().clamp_min(1)
