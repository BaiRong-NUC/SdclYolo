"""Per-label evidence features and a conservative prediction-only class choice."""

import numpy as np
import torch
from torch import nn


FEATURE_DIM = 62
CONTEXT_DIM = 20


class ClassChoiceScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def make_features(candidates, score, raw_score, context, size):
    n = len(candidates)
    dimensions = candidates[:, 2:4] - candidates[:, :2]
    confidence = np.clip(candidates[:, 4], 1e-6, 1 - 1e-6)
    center = (candidates[:, :2] + candidates[:, 2:4]) / 2 / size
    basic = np.column_stack((
        confidence, np.log(confidence / (1 - confidence)),
        np.log1p(np.sqrt(dimensions.prod(1)) * 640 / max(size) / 16),
        np.log(dimensions[:, 0] / dimensions[:, 1]), center,
        np.eye(10)[candidates[:, 5].astype(int)],
    ))
    basic = np.repeat(basic[:, None], 10, axis=1)
    classes = np.broadcast_to(np.eye(10), (n, 10, 10))
    keep = (np.arange(10)[None, :] == candidates[:, 5:6])[:, :, None]
    evidence = np.stack((score, raw_score, raw_score > 0), axis=3).reshape(n, 10, 15)
    observation = (score[:, :, :, None] * context[:, None, :, :]).reshape(n, 10, CONTEXT_DIM)
    result = np.concatenate((basic, classes, keep, evidence, observation), axis=2).astype(np.float32)
    if result.shape != (n, 10, FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid class-choice features.")
    return result


def choose_labels(outputs, candidates, valid):
    utilities = np.where(valid, outputs, -np.inf)
    maximum = utilities.max(1, keepdims=True)
    probability = np.exp(utilities - maximum)
    probability /= probability.sum(1, keepdims=True)
    original = candidates[:, 5].astype(int)
    best = probability.argmax(1)
    rows = np.arange(len(best))
    replace = (candidates[:, 4] >= .05) & (
        probability[rows, best] - probability[rows, original] > .10)
    return np.where(replace, best, original)
