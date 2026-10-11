"""Training-only, instance-masked sub-stride translation consistency."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.utils.ops import xywh2xyxy


SHIFTS = ((1, 3), (-3, 1), (5, -2), (-2, -5), (7, 4), (-4, 7), (3, -7), (-7, -3))


@dataclass(frozen=True)
class PhaseConfig:
    enabled: bool = False
    weight: float = 1.0
    warmup_epochs: int = 5
    ramp_epochs: int = 10
    interval: int = 2
    reference_size: int = 640
    max_side: float = 32.0
    max_objects_per_image: int = 256
    border_margin: int = 32

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("phase.enabled must be boolean.")
        if not math.isfinite(self.weight) or not 0 < self.weight <= 1:
            raise ValueError("phase.weight must be in (0, 1].")
        for key in ("warmup_epochs", "ramp_epochs", "interval", "reference_size",
                    "max_objects_per_image", "border_margin"):
            value = getattr(self, key)
            if type(value) is not int or value < (1 if key in {
                    "interval", "reference_size", "max_objects_per_image"} else 0):
                raise ValueError(f"Invalid phase.{key}.")
        if not math.isfinite(self.max_side) or self.max_side <= 0:
            raise ValueError("phase.max_side must be positive.")

    @classmethod
    def from_dict(cls, values=None):
        values = values or {}
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown or invalid phase settings.")
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    def strength(self, epoch):
        if not self.enabled or epoch < self.warmup_epochs:
            return 0.0
        progress = 1.0 if self.ramp_epochs == 0 else (
            epoch - self.warmup_epochs) / self.ramp_epochs
        return self.weight * min(1.0, max(0.0, progress))


def translate_images(images, shift):
    """Non-circular integer translation; exposed pixels use YOLO padding."""
    dx, dy = shift
    height, width = images.shape[-2:]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("Shift exceeds image dimensions.")
    result = torch.full_like(images, 114 / 255)
    sx, sy = max(0, -dx), max(0, -dy)
    tx, ty = max(0, dx), max(0, dy)
    w, h = width - abs(dx), height - abs(dy)
    result[..., ty:ty+h, tx:tx+w] = images[..., sy:sy+h, sx:sx+w]
    return result


@contextmanager
def auxiliary_batch_stats(model):
    """Use training batch statistics without a second update of BN buffers."""
    layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
              and m.training]
    previous = [m.track_running_stats for m in layers]
    try:
        for layer in layers:
            layer.track_running_stats = False
        yield
    finally:
        for layer, tracking in zip(layers, previous):
            layer.track_running_stats = tracking


def object_samples(batch, shift, config, small_only=True):
    """Return ROI points and ownership masks; no augmentation RNG is consumed."""
    height, width = batch["img"].shape[-2:]
    boxes = xywh2xyxy(batch["bboxes"].float()) * batch["bboxes"].new_tensor(
        [width, height, width, height])
    indices = batch["batch_idx"].flatten().long()
    ignore_boxes = batch.get("ignore_bboxes", boxes.new_empty((0, 4)))
    ignore_boxes = xywh2xyxy(ignore_boxes.float()) * boxes.new_tensor([width, height] * 2)
    ignore_indices = batch.get("ignore_batch_idx", indices.new_empty(0)).flatten().long()
    fractions = boxes.new_tensor([0.25, 0.5, 0.75])
    yy, xx = torch.meshgrid(fractions, fractions, indexing="ij")
    fractions = torch.stack((xx.flatten(), yy.flatten()), -1)
    offset = boxes.new_tensor(shift)
    groups = []
    selected_count = valid_count = valid_points = 0
    for image_index in range(len(batch["img"])):
        local = boxes[indices == image_index]
        ignored = ignore_boxes[ignore_indices == image_index]
        sides = (local[:, 2:] - local[:, :2]).clamp_min(0)
        side = sides.prod(-1).sqrt() * config.reference_size / max(height, width)
        eligible = sides.min(-1).values > 0
        if small_only:
            eligible &= side < config.max_side
        margin = config.border_margin
        eligible &= (local[:, :2] >= margin).all(-1)
        eligible &= (local[:, 2:] <= local.new_tensor([width, height]) - margin).all(-1)
        eligible &= (local[:, :2] + offset >= margin).all(-1)
        eligible &= (local[:, 2:] + offset <= local.new_tensor([width, height]) - margin).all(-1)
        selected = eligible.nonzero().flatten()
        if len(selected) > config.max_objects_per_image:
            positions = torch.linspace(0, len(selected)-1, config.max_objects_per_image,
                                       device=selected.device).long()
            selected = selected[positions]
        if not len(selected):
            continue
        points = local[selected, None, :2] + sides[selected, None] * fractions
        inside = ((points[:, :, None] >= local[None, None, :, :2])
                  & (points[:, :, None] <= local[None, None, :, 2:])).all(-1)
        own = torch.arange(len(selected), device=selected.device)
        inside[own, :, selected] = False
        valid = ~inside.any(-1)
        if len(ignored):
            inside_ignore = ((points[:, :, None] >= ignored[None, None, :, :2])
                             & (points[:, :, None] <= ignored[None, None, :, 2:])).all(-1)
            valid &= ~inside_ignore.any(-1)
        selected_count += len(selected)
        valid_count += int(valid.any(-1).sum())
        valid_points += int(valid.sum())
        groups.append((image_index, points, valid, side[selected]))
    return groups, {"selected_objects": selected_count, "valid_objects": valid_count,
                    "valid_points": valid_points}


def sample_features(feature, image_index, points, image_hw):
    height, width = image_hw
    fh, fw = feature.shape[-2:]
    coordinates = points.float() * points.new_tensor([fw / width, fh / height]) - 0.5
    lower = coordinates.floor().long()
    fraction = coordinates - lower
    flattened = feature[image_index].float().flatten(1)
    result = feature.new_zeros((*points.shape[:2], feature.shape[1]), dtype=torch.float32)
    # index_select supports deterministic CUDA backward; grid_sample does not.
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = lower[..., 0] + dx, lower[..., 1] + dy
        inside = (x >= 0) & (x < fw) & (y >= 0) & (y < fh)
        indices = (y.clamp(0, fh-1) * fw + x.clamp(0, fw-1)).flatten()
        sampled = flattened.index_select(1, indices).T.reshape(result.shape)
        wx = fraction[..., 0] if dx else 1 - fraction[..., 0]
        wy = fraction[..., 1] if dy else 1 - fraction[..., 1]
        result = result + sampled * (wx * wy * inside)[..., None]
    return result


def phase_distance(original, translated, groups, shift, image_hw):
    values, sides = [], []
    for image_index, points, valid, side in groups:
        first = sample_features(original, image_index, points, image_hw)
        second = sample_features(translated.detach(), image_index,
                                 points + points.new_tensor(shift), image_hw)
        distances = 1 - F.cosine_similarity(first, second, dim=-1, eps=1e-6)
        objects = valid.any(-1)
        per_object = (distances * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        values.append(per_object[objects])
        sides.append(side[objects])
    if not values:
        return original.sum() * 0, original.new_empty(0), original.new_empty(0)
    values, sides = torch.cat(values), torch.cat(sides)
    return (values.mean() if len(values) else original.sum() * 0), values, sides


def phase_model_loss(model, batch, preds=None):
    config = PhaseConfig.from_dict(getattr(model, "phase_settings", {}))
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    if preds is None:
        preds = model.predict(batch["img"])
    loss, items = model.criterion(preds, batch)
    if not config.enabled:
        return loss, items
    auxiliary = loss.new_zeros(())
    strength = config.strength(getattr(model, "sdcl_epoch", 0))
    step = getattr(model, "phase_step", 0)
    if model.training:
        model.phase_step = step + 1
    if model.training and strength > 0 and step % config.interval == 0:
        shift = SHIFTS[(step // config.interval) % len(SHIFTS)]
        groups, stats = object_samples(batch, shift, config)
        if stats["valid_objects"]:
            with torch.no_grad(), auxiliary_batch_stats(model):
                target = model.predict(translate_images(batch["img"], shift))
            original = model.criterion.parse_output(preds)["feats"][0]
            translated = model.criterion.parse_output(target)["feats"][0]
            distance, _, _ = phase_distance(
                original, translated, groups, shift, batch["img"].shape[-2:])
            auxiliary = distance * strength * config.interval
            model.phase_diagnostics = {
                **stats, "distance": float(distance.detach()),
                "strength": strength, "shift": list(shift), "step": step,
            }
            running = getattr(model, "phase_statistics", {})
            running["active_steps"] = running.get("active_steps", 0) + 1
            running["distance_sum"] = running.get("distance_sum", 0.0) + float(distance.detach())
            running["valid_objects_sum"] = running.get("valid_objects_sum", 0) + stats["valid_objects"]
            running["selected_objects_sum"] = running.get("selected_objects_sum", 0) + stats["selected_objects"]
            model.phase_statistics = running
    if model.training:
        running = getattr(model, "phase_statistics", {})
        running["batches"] = running.get("batches", 0) + 1
        running["weighted_loss_sum"] = running.get("weighted_loss_sum", 0.0) + float(auxiliary.detach())
        model.phase_statistics = running
    items = {**items, "phase_loss": auxiliary.detach()}
    return torch.cat((loss, (auxiliary * len(batch["img"])).reshape(1))), items
