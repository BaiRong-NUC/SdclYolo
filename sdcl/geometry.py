"""Geometry for ignored regions in model-input coordinates."""

import torch


def anchor_ignore_mask(points, boxes_xywh, batch_ids, batch_size, image_size):
    result = torch.zeros((batch_size, points.shape[0]), dtype=torch.bool, device=points.device)
    if boxes_xywh.numel() == 0:
        return result
    height, width = image_size
    boxes = boxes_xywh.to(points.device, dtype=torch.float32)
    centers = boxes[:, :2] * boxes.new_tensor([width, height])
    half = boxes[:, 2:] * boxes.new_tensor([width, height]) / 2
    low, high = centers - half, centers + half
    for batch_id in range(batch_size):
        chosen = batch_ids.to(points.device).long() == batch_id
        if not chosen.any():
            continue
        inside = (points[:, None, :] >= low[chosen]).all(-1) & (points[:, None, :] <= high[chosen]).all(-1)
        result[batch_id] = inside.any(-1)
    return result


def prediction_ignore_mask(boxes, ignores, threshold=0.5):
    """Union-free IoF filter for diagnostics; not the official VisDrone metric."""
    if boxes.numel() == 0 or ignores.numel() == 0:
        return torch.zeros(len(boxes), device=boxes.device, dtype=torch.bool)
    lt = torch.maximum(boxes[:, None, :2], ignores[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], ignores[None, :, 2:])
    intersection = (rb - lt).clamp_min(0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1).clamp_min(1e-12)
    return (intersection / area[:, None]).amax(-1) > threshold

