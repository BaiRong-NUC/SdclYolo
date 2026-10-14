"""Frozen stride8 feature sampling at action boundaries and expanded context."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F


SIDE = 960
CHANNELS = 128
PROJECTED = 16
VISUAL_DIM = 2 * 3 * 3 * PROJECTED


def letterbox_geometry(size, side=SIDE):
    ratio = side / max(size)
    width, height = (round(s * ratio) for s in size)
    left = round((side - width) / 2 - .1)
    top = round((side - height) / 2 - .1)
    return ratio, left, top, width, height


def image_tensor(image, device):
    size = [image.shape[1], image.shape[0]]
    ratio, left, top, width, height = letterbox_geometry(size)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, SIDE - height - top,
                                left, SIDE - width - left, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).to(device=device, dtype=torch.float16).unsqueeze(0) / 255.


def fixed_projection(device):
    value = torch.randn(PROJECTED, CHANNELS, 1, 1,
                        generator=torch.Generator().manual_seed(20261014))
    return (value / np.sqrt(CHANNELS)).to(device)


def sample_context(projected, boxes, size):
    if projected.shape[0] != 1 or projected.shape[1] != PROJECTED:
        raise ValueError("Wrong projected local feature map.")
    shape = boxes.shape[:-1]
    b = torch.as_tensor(boxes.reshape(-1, 4), dtype=torch.float32, device=projected.device)
    if not len(b):
        return np.empty((*shape, VISUAL_DIM), np.float32)
    ratio, left, top, _, _ = letterbox_geometry(size)
    center = (b[:, :2] + b[:, 2:]) / 2
    dimensions = b[:, 2:] - b[:, :2]
    fractions = torch.tensor([0., .5, 1.], device=projected.device)
    ys, xs = torch.meshgrid(fractions, fractions, indexing="ij")
    points = torch.stack((xs.ravel(), ys.ravel()), dim=1)
    samples = []
    for expansion in (1., 1.25):
        lower = center - dimensions * expansion / 2
        xy = lower[:, None, :] + dimensions[:, None, :] * expansion * points[None, :, :]
        xy = xy * ratio + torch.tensor([left, top], device=projected.device)
        grid = (xy * 2 / SIDE - 1).unsqueeze(0)
        value = F.grid_sample(projected.float(), grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
        samples.append(value[0].permute(1, 2, 0).reshape(len(b), -1))
    result = torch.cat(samples, dim=1).cpu().numpy().astype(np.float32)
    if result.shape != (len(b), VISUAL_DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid frozen local features.")
    return result.reshape(*shape, VISUAL_DIM)
