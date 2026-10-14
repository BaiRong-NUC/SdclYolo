"""Action-conditioned native DFL evidence, without labels or fitted parameters."""

import numpy as np


def action_distances(boxes, anchors, strides, window, input_shape):
    x1, y1, x2, y2 = window
    width, height = x2 - x1, y2 - y1
    gain = min(input_shape[0] / height, input_shape[1] / width)
    pad = np.array([round((input_shape[1] - round(width * gain)) / 2 - .1),
                    round((input_shape[0] - round(height * gain)) / 2 - .1)])
    local = (boxes - [x1, y1, x1, y1]) * gain + np.tile(pad, 2)
    local = local / strides[:, None, None]
    return np.concatenate((anchors[:, None] - local[:, :, :2],
                           local[:, :, 2:] - anchors[:, None]), axis=2)


def evidence(probability, distances):
    if probability.shape != (len(distances), 4, 16) or distances.shape[-1] != 4:
        raise ValueError("Native four-side16-bin shapes differ.")
    bins = np.arange(16)
    # FP16 native softmax can sum slightly away from1; use the same normalized shape
    # for all three comparisons, while native replay is verified separately.
    probability = probability.astype(float)
    probability /= np.maximum(probability.sum(-1, keepdims=True), 1e-12)
    mean = (probability * bins).sum(-1)
    variance = (probability * (bins - mean[:, :, None]) ** 2).sum(-1)
    lo = np.floor(distances).astype(int)
    fraction = distances - lo
    n, a, _ = distances.shape
    rows = np.arange(n)[:, None, None]
    sides = np.arange(4)[None, None, :]
    lower = probability[rows, sides, np.clip(lo, 0, 15)]
    upper = probability[rows, sides, np.clip(lo + 1, 0, 15)]
    mass = lower * (1 - fraction) + upper * fraction
    mass[(distances < 0) | (distances > 15)] = 0
    delta = distances - mean[:, None]
    return {
        "distribution": np.log(np.maximum(mass, 1e-9)).mean(-1),
        "mean_distance": -np.abs(delta).mean(-1),
        "gaussian": (-.5 * (delta ** 2 / np.maximum(variance[:, None], 1e-6) +
                            np.log(2 * np.pi * np.maximum(variance[:, None], 1e-6)))).mean(-1),
    }
