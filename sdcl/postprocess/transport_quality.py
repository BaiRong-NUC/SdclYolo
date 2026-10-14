"""Compare native boundary distributions in image coordinates before NMS."""

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from sdcl.postprocess.native_spatial_field import neighboring_indices


ARMS = ("geometry", "dgqp", "centered", "transport")
WIDTH = 164


def wasserstein1(x, p, y, q):
    """Batched exact 1D W1 for two equally sized discrete distributions."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    points = np.concatenate((x, y), axis=-1)
    masses = np.concatenate((p, -q), axis=-1)
    order = np.argsort(points, axis=-1, kind="stable")
    points = np.take_along_axis(points, order, axis=-1)
    cumulative = np.cumsum(np.take_along_axis(masses, order, axis=-1), axis=-1)
    return (np.abs(cumulative[..., :-1]) * np.diff(points, axis=-1)).sum(-1)


def boundary_support(anchors, strides, window):
    x1, y1, x2, y2 = window
    width, height = x2-x1, y2-y1
    gain = 960/max(width, height)
    pad = np.array([round((960-round(width*gain))/2-.1),
                    round((960-round(height*gain))/2-.1)])
    centers = anchors * strides[:, None]
    origins = centers[:, [0, 1, 0, 1]]
    support = origins[:, :, None] + (np.array([-1, -1, 1, 1])[None, :, None] *
                                   strides[:, None, None] * np.arange(16))
    support = (support-pad[[0, 1, 0, 1]][None, :, None])/gain
    support = np.maximum(0, np.minimum(support, np.array([width, height, width, height])[None, :, None]))
    return support+np.array([x1, y1, x1, y1])[None, :, None]


def active_indices(decoded):
    scores = np.asarray(decoded)[4:14].max(0)
    rows = np.flatnonzero(scores >= .001)
    return rows[np.argsort(-scores[rows].astype(float), kind="stable")[:4096]]


def quality_features(decoded, raw_dfl, raw_class, boxes, anchors, strides, window, size, indices):
    """Inputs use predictions/image dimensions only; no labels or GT selection."""
    if decoded.shape != (14, 18900) or raw_dfl.shape != (18900, 4, 16):
        raise ValueError("Expected frozen square960 native tensors.")
    indices = np.asarray(indices, int)
    probabilities = softmax(raw_dfl.astype(np.float32), axis=-1).astype(float)
    support = boundary_support(anchors, strides, window)
    neighbors = neighboring_indices(indices)
    safe = np.maximum(neighbors, 0)
    n = len(indices)
    classes = decoded[4:14, indices].argmax(0)
    own = boxes[indices]
    scale = np.maximum(own[:, 2:]-own[:, :2], 1.)
    scale4 = scale[:, [0, 1, 0, 1]]
    other = boxes[safe]
    valid = ((neighbors >= 0) & np.isfinite(other).all(-1) &
             (other[:, :, 2:] > other[:, :, :2]).all(-1))
    delta = np.clip((other-own[:, None])/scale4[:, None], -4, 4)
    class_prob = decoded[4:14].T[safe]
    query_prob = np.take_along_axis(class_prob, classes[:, None, None], axis=-1)[:, :, 0]
    same = class_prob.argmax(-1) == classes[:, None]
    low, high = np.maximum(other[:, :, :2], own[:, None, :2]), np.minimum(other[:, :, 2:], own[:, None, 2:])
    intersection = np.maximum(high-low, 0).prod(-1)
    union = (np.maximum(other[:, :, 2:]-other[:, :, :2], 0).prod(-1) +
             np.maximum(own[:, 2:]-own[:, :2], 0).prod(-1)[:, None]-intersection)
    overlap = intersection/np.maximum(union, 1e-12)
    slots = np.concatenate((delta, query_prob[:, :, None], same[:, :, None],
                            overlap[:, :, None], valid[:, :, None]), -1)
    slots[~valid] = 0
    levels = np.where(indices < 14400, 0, np.where(indices < 18000, 1, 2))
    base = np.concatenate((np.clip(raw_class[indices], -20, 20), np.eye(10)[classes],
                           np.eye(3)[levels], own/np.array([*size, *size]),
                           decoded[4:14, indices].max(0)[:, None], slots.reshape(n, 64)), 1)
    # GFLV2-inspired ordinary control, not a reproduction of its trained detector.
    top = np.sort(probabilities, axis=-1)[..., -4:][..., ::-1]
    stats = np.concatenate((top, top.mean(-1, keepdims=True)), -1).reshape(18900, 20)
    nearby = (stats[safe]*valid[:, :, None]).sum(1)/np.maximum(valid.sum(1)[:, None], 1)
    quality = np.concatenate((stats[indices], nearby), 1)
    p, q = probabilities[indices, None], probabilities[safe]
    x, y = support[indices, None], support[safe]
    transported = wasserstein1(np.broadcast_to(x, y.shape), np.broadcast_to(p, q.shape), y, q)
    mean_x = (x*p).sum(-1, keepdims=True)
    mean_y = (y*q).sum(-1, keepdims=True)
    centered = wasserstein1(np.broadcast_to(x-mean_x, y.shape), np.broadcast_to(p, q.shape), y-mean_y, q)
    result = {}
    for arm in ARMS:
        extra = np.zeros((n, 8, 4))
        if arm in ("centered", "transport"):
            distance = centered if arm == "centered" else transported
            extra = np.minimum(distance/scale4[:, None], 4.)
            extra[~valid] = 0
        result[arm] = np.concatenate((base, quality if arm != "geometry" else np.zeros_like(quality),
                                      extra.reshape(n, 32)), 1).astype(np.float32)
        if result[arm].shape != (n, WIDTH) or not np.isfinite(result[arm]).all():
            raise ValueError("Invalid fixed quality inputs.")
    return result


class QualityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WIDTH, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        return 2*torch.tanh(self.net(inputs).squeeze(-1))


def normalized(inputs, mean, std):
    return np.clip((inputs-mean)/std, -8, 8).astype(np.float32)


def adjusted_decoded(decoded, indices, shifts):
    """Preserve the original argmax class by shifting all class logits equally."""
    result = decoded.clone()
    rows = torch.as_tensor(indices, device=decoded.device)
    shift = torch.as_tensor(shifts, dtype=torch.float32, device=decoded.device)
    if not torch.isfinite(shift).all() or (shift.abs() > 2.000001).any():
        raise ValueError("Invalid bounded native score shift.")
    if torch.count_nonzero(shift) == 0:
        return result
    values = decoded[:, 4:14, rows].float().clamp(1e-7, 1-1e-7)
    result[:, 4:14, rows] = torch.sigmoid(torch.logit(values)+shift[None, None]).to(decoded.dtype)
    if not torch.equal(result[:, :4], decoded[:, :4]):
        raise AssertionError("Quality scoring modified coordinates.")
    return result
