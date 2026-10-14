"""Native RGB coordinate-action differences with aligned and shuffled controls."""

import hashlib

import cv2
import numpy as np
import torch
from torch import nn


PIXEL_DIM = 480


def sample_actions(image, boxes, mode):
    shape = boxes.shape[:2]
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    if not len(b):
        return np.empty((*shape, PIXEL_DIM), np.float32)
    width, height = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    if mode == "boundary":
        fraction = np.linspace(.1, .9, 8, dtype=np.float32)[None, :, None]
        normal = np.arange(-2, 3, dtype=np.float32)[None, None, :]
        xs, ys = [], []
        for side in range(4):
            inward = 1 if side < 2 else -1
            if side in (0, 2):
                x = b[:, side, None, None] + inward * normal
                y = b[:, 1, None, None] + height[:, None, None] * fraction
            else:
                x = b[:, 0, None, None] + width[:, None, None] * fraction
                y = b[:, side, None, None] + inward * normal
            xs.append(np.broadcast_to(x, (len(b), 8, 5)).reshape(len(b), 40))
            ys.append(np.broadcast_to(y, (len(b), 8, 5)).reshape(len(b), 40))
        x, y = np.concatenate(xs, 1), np.concatenate(ys, 1)
    elif mode == "interior":
        xx, yy = np.meshgrid(np.linspace(.1, .9, 10), np.linspace(.1, .9, 16))
        x = b[:, 0, None] + width[:, None] * xx.reshape(1, -1)
        y = b[:, 1, None] + height[:, None] * yy.reshape(1, -1)
    else:
        raise ValueError("Unknown fixed pixel sampling mode.")
    # OpenCV remap is limited to SHRT_MAX output dimensions; chunk along boxes.
    values = [cv2.remap(image, x[i:i+4096].astype(np.float32), y[i:i+4096].astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
              for i in range(0, len(b), 4096)]
    return np.concatenate(values).reshape(*shape, PIXEL_DIM).astype(np.float32)


def action_differences(image, boxes, uniform, mode):
    sampled = sample_actions(image, boxes, mode)
    reference = sampled[np.arange(len(boxes)), uniform]
    return ((sampled - reference[:, None]) / 255).astype(np.float32)


def shuffle_differences(differences, stem, anchor_rows):
    result = differences.reshape(len(differences), 6, 160, 3).copy()
    for index, row in enumerate(anchor_rows):
        seed = int.from_bytes(hashlib.sha256(f"native-boundary:{stem}:{int(row)}".encode()).digest()[:8],
                              "little")
        permutation = np.random.default_rng(seed).permutation(160)
        result[index] = result[index][:, permutation]
    return result.reshape(differences.shape)


class PixelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(68 + PIXEL_DIM, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, geometry, pixels):
        return self.network(torch.cat((geometry, pixels), -1)).squeeze(-1)
