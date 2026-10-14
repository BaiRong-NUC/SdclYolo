"""Separate the integer interpolation floor from native DFL variance."""

import numpy as np


def distribution_reliability(probability):
    probability = np.asarray(probability, dtype=float)
    if probability.shape[-2:] != (4, 16) or not np.isfinite(probability).all():
        raise ValueError("Expected finite four-side sixteen-bin distributions.")
    if np.any(probability < 0):
        raise ValueError("Distribution masses must be nonnegative.")
    mass = probability.sum(-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("Distribution masses must sum positive.")
    probability = probability / mass
    bins = np.arange(16)
    mean = (probability * bins).sum(-1)
    variance = (probability * (bins - mean[..., None]) ** 2).sum(-1)
    fraction = mean - np.floor(mean)
    floor = fraction * (1 - fraction)
    if np.any(variance < floor - 1e-10):
        raise ValueError("Integer-support variance bound violated.")
    excess = np.maximum(variance - floor, 0)
    dimension = np.stack((mean[..., 0] + mean[..., 2],
                          mean[..., 1] + mean[..., 3]), -1)
    scale = np.maximum(np.tile(dimension, (1,) * (dimension.ndim - 1) + (2,)), 1e-6)
    entropy = -(probability * np.log(np.maximum(probability, 1e-300))).sum(-1)
    return {
        "mean": mean, "variance": variance, "floor": floor, "excess": excess,
        "raw_variance": -(np.sqrt(variance) / scale).mean(-1),
        "excess_variance": -(np.sqrt(excess) / scale).mean(-1),
        "entropy": -entropy.mean(-1),
    }


def original_coordinates(mean, anchors, strides, window):
    x1, y1, x2, y2 = window
    width, height = x2-x1, y2-y1
    gain = min(960/height, 960/width)
    pad = np.array([round((960-round(width*gain))/2-.1),
                    round((960-round(height*gain))/2-.1)])
    local = np.concatenate((anchors-mean[..., :2], anchors+mean[..., 2:]), -1)
    local = (local*strides[..., None]-np.tile(pad, 2))/gain
    local[..., [0, 2]] = np.clip(local[..., [0, 2]], 0, width)+x1
    local[..., [1, 3]] = np.clip(local[..., [1, 3]], 0, height)+y1
    return local
