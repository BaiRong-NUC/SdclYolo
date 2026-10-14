"""Prediction-only boundary/context features for cached coordinate actions."""

import cv2
import numpy as np

from sdcl.postprocess.localization import action_feature_names


VISUAL_DIM = 24


def recover_cached_boxes(features, size):
    """Invert unnormalized cached geometry; float32 rounding is checked below."""
    names = {name: i for i, name in enumerate(action_feature_names())}
    base = features[:, 0]
    side = np.expm1(base[:, names["log_reference_side"]].astype(np.float64))
    side *= 16 * max(size) / 640.
    aspect = np.exp(base[:, names["log_aspect"]].astype(np.float64))
    widths, heights = side * np.sqrt(aspect), side / np.sqrt(aspect)
    centers = base[:, [names["center_x"], names["center_y"]]].astype(np.float64) * size
    keep = np.column_stack((centers - np.column_stack((widths, heights)) / 2,
                            centers + np.column_stack((widths, heights)) / 2))
    relative = features[:, :, [names[n] for n in ("dx1", "dy1", "dx2", "dy2")]]
    scale = np.column_stack((widths, heights, widths, heights))
    boxes = keep[:, None, :] + relative * scale[:, None, :]
    low, high = np.maximum(boxes[:, :, :2], keep[:, None, :2]), np.minimum(
        boxes[:, :, 2:], keep[:, None, 2:])
    intersection = np.maximum(high - low, 0).prod(2)
    area = np.maximum(boxes[:, :, 2:] - boxes[:, :, :2], 0).prod(2)
    overlap = intersection / np.maximum(area + (widths * heights)[:, None] - intersection, 1e-12)
    if (not np.isfinite(boxes).all() or np.any(boxes[:, :, 2:] <= boxes[:, :, :2])
            or not np.allclose(overlap, features[:, :, names["iou_anchor"]], atol=2e-5, rtol=0)):
        raise ValueError("Cached coordinate inversion failed its anchor-IoU check.")
    return boxes


def visual_features(image, boxes):
    """Four boundary descriptors per side, plus eight local grayscale statistics."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an original BGR image.")
    original_shape = boxes.shape[:-1]
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(b):
        return np.empty((*original_shape, VISUAL_DIM), dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.
    magnitude = np.sqrt(gx * gx + gy * gy)
    fractions = np.linspace(.1, .9, 16, dtype=np.float32)[None, :]
    widths, heights = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    offsets = np.maximum(np.minimum(widths, heights) * .05, 1.)[:, None]

    def sample(array, x, y):
        return cv2.remap(array, x.astype(np.float32), y.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    borders, inner, outer = [], [], []
    for side in range(4):
        vertical = side in (0, 2)
        if vertical:
            x = np.broadcast_to(b[:, side:side + 1], (len(b), 16))
            y = b[:, 1:2] + heights[:, None] * fractions
        else:
            x = b[:, :1] + widths[:, None] * fractions
            y = np.broadcast_to(b[:, side:side + 1], (len(b), 16))
        sign = 1 if side < 2 else -1
        dx, dy = (sign * offsets, 0) if vertical else (0, sign * offsets)
        inside, outside = sample(gray, x + dx, y + dy), sample(gray, x - dx, y - dy)
        edge = np.abs(sample(gx if vertical else gy, x, y))
        contrast = inside - outside
        borders.extend((edge.mean(1), np.quantile(edge, .9, axis=1),
                        contrast.mean(1), np.abs(contrast).mean(1)))
        inner.append(inside)
        outer.append(outside)
    inside, outside = np.concatenate(inner, axis=1), np.concatenate(outer, axis=1)
    grid = np.array([.2, .4, .6, .8], np.float32)
    grid_x, grid_y = np.meshgrid(grid, grid)
    xs = b[:, :1] + widths[:, None] * grid_x.ravel()[None, :]
    ys = b[:, 1:2] + heights[:, None] * grid_y.ravel()[None, :]
    interior = sample(gray, xs, ys)
    interior_edge = sample(magnitude, xs, ys)
    context = (interior.mean(1), interior.std(1), outside.mean(1), outside.std(1),
               inside.mean(1) - outside.mean(1), np.abs(
                   inside.mean(1) - outside.mean(1)), interior_edge.mean(1),
               interior_edge.std(1))
    result = np.column_stack((*borders, *context)).astype(np.float32)
    if result.shape != (len(b), VISUAL_DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid boundary/context features.")
    return result.reshape(*original_shape, VISUAL_DIM)
