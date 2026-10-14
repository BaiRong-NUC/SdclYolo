"""Dense class-map summaries before native thresholding and NMS."""

import numpy as np


def map_anchor_centers(centers, size, input_shape=(960, 960)):
    width, height = size
    gain = min(input_shape[0] / height, input_shape[1] / width)
    pad = np.array([round((input_shape[1] - round(width * gain)) / 2 - .1),
                    round((input_shape[0] - round(height * gain)) / 2 - .1)])
    return (centers - pad) / gain


def dense_features(probability, centers, levels, windows, size):
    values = []
    valid_image = ((centers >= [0, 0]) & (centers < size)).all(1)
    for window in windows[1:]:
        inside = valid_image & ((centers >= window[:2]) & (centers < window[2:])).all(1)
        row = []
        for level in range(3):
            p = probability[inside & (levels == level)]
            if not len(p):
                row.extend(np.zeros(40))
                continue
            top = np.sort(p, axis=0)[-5:]
            stats = np.stack((p.max(0), top.mean(0), p.mean(0), np.log1p((p >= .05).sum(0))), 1)
            row.extend(stats.reshape(-1))
        values.append(row)
    result = np.asarray(values, np.float32)
    if result.shape != (4, 120) or not np.isfinite(result).all():
        raise ValueError("Invalid frozen dense summaries.")
    return result
