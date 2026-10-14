"""Ordinary quality estimation followed by a bounded score-only residual."""

import numpy as np
from torch import nn

from sdcl.postprocess.coordinate_score_debt import correct_scores


class QualityControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(68, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def quality_scores(predictions, residual, active):
    residual = np.asarray(residual, float)
    active = np.asarray(active, bool)
    if residual.shape != (len(predictions),) or active.shape != residual.shape:
        raise ValueError("Quality residual and active mask must align with frozen predictions.")
    correction = np.zeros(len(predictions))
    changed = active & (residual != 0)
    score = predictions[changed, 4]
    bounded = np.clip(score, 1e-6, 1-1e-6)
    z = np.log(bounded/(1-bounded))+residual[changed]
    # Stable sigmoid for potentially extreme learned logit residuals.
    quality = np.exp(-np.logaddexp(0., -z))
    correction[changed] = quality-score
    return correct_scores(predictions, correction), correction
