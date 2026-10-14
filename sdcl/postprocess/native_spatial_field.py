"""Prediction-only adjacent-grid coordinate fields with order-matched controls."""

import hashlib

import numpy as np


OFFSETS = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)
                    if (x, y) != (0, 0)])


def neighboring_indices(indices, shapes=(120, 60, 30)):
    indices = np.asarray(indices, dtype=int)
    total = sum(h*h for h in shapes)
    if np.any((indices < 0) | (indices >= total)):
        raise ValueError("Native index outside frozen grid.")
    neighbors = np.full((len(indices), 8), -1, int)
    base = 0
    for h in shapes:
        rows = np.flatnonzero((indices >= base) & (indices < base+h*h))
        local = indices[rows]-base
        x = local[:, None] % h+OFFSETS[None, :, 0]
        y = local[:, None] // h+OFFSETS[None, :, 1]
        valid = (x >= 0) & (x < h) & (y >= 0) & (y < h)
        neighbors[rows] = np.where(valid, base+y*h+x, -1)
        base += h*h
    return neighbors


def field_features(coordinates, probabilities, indices, classes, scale):
    if coordinates.shape != (18900, 4) or probabilities.shape != (18900, 10):
        raise ValueError("Expected frozen square960 decoded native grid.")
    neighbors = neighboring_indices(indices)
    gathered = coordinates[np.maximum(neighbors, 0)]
    central = coordinates[indices, None]
    valid = ((neighbors >= 0) & np.isfinite(gathered).all(-1) &
             (gathered[:, :, 2:] > gathered[:, :, :2]).all(-1))
    selected_prob = probabilities[np.maximum(neighbors, 0)]
    valid &= np.isfinite(selected_prob).all(-1)
    delta = np.clip((gathered-central)/np.maximum(scale[:, None], 1e-6), -1, 1)
    query_prob = np.take_along_axis(selected_prob, classes[:, None, None], axis=2)[:, :, 0]
    same = selected_prob.argmax(-1) == classes[:, None]
    lo, hi = np.maximum(gathered[:, :, :2], central[:, :, :2]), np.minimum(gathered[:, :, 2:], central[:, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(-1)
    union = (np.maximum(gathered[:, :, 2:]-gathered[:, :, :2], 0).prod(-1) +
             np.maximum(central[:, :, 2:]-central[:, :, :2], 0).prod(-1)-intersection)
    overlap = intersection/np.maximum(union, 1e-12)
    result = np.concatenate((delta, query_prob[:, :, None], same[:, :, None],
                             overlap[:, :, None], valid[:, :, None]), axis=2)
    result[~valid] = 0
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite adjacent-grid features.")
    return result.astype(np.float32), neighbors


def control_features(slots, stem):
    if slots.shape[1:] != (8, 8):
        raise ValueError("Eight ordered slots with eight channels required.")
    valid = slots[:, :, 7] > 0
    mean = slots.sum(1)/np.maximum(valid.sum(1, keepdims=True), 1)
    permutations = np.empty((len(slots), 8), int)
    for row in range(len(slots)):
        seed = int.from_bytes(hashlib.sha256(f"{stem}:{row}:spatial".encode()).digest()[:8], "little")
        permutations[row] = np.random.default_rng(seed).permutation(8)
    shuffled = np.take_along_axis(slots, permutations[:, :, None], axis=1)
    return {"ordered": slots.reshape(-1, 64), "mean": np.tile(mean, (1, 8)).astype(np.float32),
            "shuffled": shuffled.reshape(-1, 64)}, permutations
