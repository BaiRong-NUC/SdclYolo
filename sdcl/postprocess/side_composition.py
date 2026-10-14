"""Side-wise coordinate choice with a matched whole-box readout."""

import numpy as np
import torch
from torch import nn

from sdcl.postprocess.localization import apply_actions, select_actions


class SideResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(68, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features)


def choose_coordinates(candidates, boxes, valid, uniform, utility, composed):
    if utility.shape != (*valid.shape, 4):
        raise ValueError("Four utilities per coordinate action required.")
    if not composed:
        mean = utility[:, :, 0] + (utility - utility[:, :, :1]).mean(-1)
        selected = select_actions(mean, valid, "absolute")
        return apply_actions(candidates, boxes, selected), {
            "invalid_fallback": 0, "new_compositions_at_score025": 0,
        }
    sides = np.stack([select_actions(utility[:, :, side], valid, "absolute") for side in range(4)], 1)
    row = np.arange(len(candidates))[:, None]
    coordinates = boxes[row, sides, np.arange(4)[None]]
    invalid = (~np.isfinite(coordinates).all(1) |
               (coordinates[:, 2] <= coordinates[:, 0]) | (coordinates[:, 3] <= coordinates[:, 1]))
    reference = apply_actions(candidates, boxes, uniform)
    coordinates[invalid] = reference[invalid, :4]
    result = candidates.copy()
    result[:, :4] = coordinates
    identical = (coordinates[:, None] == boxes).all(-1) & valid
    novel = ~identical.any(1) & (candidates[:, 4] >= .25)
    if not np.array_equal(result[:, 4:], candidates[:, 4:]):
        raise ValueError("Side composition changed confidence or class.")
    return result, {"invalid_fallback": int(invalid.sum()),
                    "new_compositions_at_score025": int(novel.sum())}
