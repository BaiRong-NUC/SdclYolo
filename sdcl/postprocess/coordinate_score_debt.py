"""Frozen box coordinates with explicitly bounded logit score residuals."""

import numpy as np


def correct_scores(predictions, residual):
    predictions = np.asarray(predictions, float)
    residual = np.asarray(residual, float)
    if residual.shape != (len(predictions),) or not np.isfinite(residual).all():
        raise ValueError("One finite score residual per frozen detection required.")
    if np.any(np.abs(residual) > 1+1e-6):
        raise ValueError("This fixed score screen only allows residuals in[-1,1].")
    result = predictions.copy()
    changed = residual != 0
    score = np.clip(predictions[changed, 4], 1e-6, 1-1e-6)
    result[changed, 4] = 1/(1+np.exp(-(np.log(score/(1-score))+residual[changed])))
    if not np.array_equal(result[:, :4], predictions[:, :4]) or not np.array_equal(result[:, 5:], predictions[:, 5:]):
        raise ValueError("Score correction changed coordinates or class.")
    return result


def threshold_quality(ious, owned):
    ious = np.asarray(ious, float)
    result = (ious[:, :, None] >= np.linspace(.5, .95, 10)).mean(-1)
    result[~np.asarray(owned, bool)] = 0
    return result.astype(np.float32)
