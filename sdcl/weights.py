"""Object-level, bounded, quality-mass-conserving training weights."""

from dataclasses import dataclass
import math

import torch

from .config import SDCLConfig


@dataclass
class WeightResult:
    weights: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@torch.no_grad()
def object_weights(
    config: SDCLConfig,
    epoch: int,
    target_boxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_gt_idx: torch.Tensor,
    foreground: torch.Tensor,
    probabilities: torch.Tensor,
    ious: torch.Tensor,
    image_size: tuple[int, int],
) -> WeightResult:
    """Use matched GT indices; padded/background indices never form objects."""
    quality = target_scores.float().sum(-1)
    weights = torch.ones_like(quality)
    valid = foreground & (quality > 0)
    zero = quality.new_zeros(())
    stats = {
        "quality_mass": quality.sum(),
        "weight_min": quality.new_ones(()),
        "weight_max": quality.new_ones(()),
        "mass_relative_error": zero,
        "matched_objects": zero,
        "eligible_objects": zero,
        "eligible_quality_mass": zero,
        "eligible_mass_relative_error": zero,
        "ineligible_weight_max_deviation": zero,
        "mean_scale": zero,
        "mean_difficulty": zero,
        "strength": quality.new_tensor(config.strength(epoch)),
    }
    if not valid.any():
        return WeightResult(weights, stats)

    batch_ids = torch.arange(quality.shape[0], device=quality.device)[:, None].expand_as(quality)
    pairs = torch.stack((batch_ids[valid], target_gt_idx[valid]), dim=1)
    objects, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    count = objects.shape[0]
    counts = torch.bincount(inverse, minlength=count).float()

    def average(values):
        totals = quality.new_zeros(count)
        totals.index_add_(0, inverse, values.float())
        return totals / counts

    boxes = target_boxes[valid].float()
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
    side = wh.prod(-1).sqrt() * config.scale_reference / math.sqrt(math.prod(image_size))
    scales = average(1.0 / (1.0 + (side / config.scale_s0).square()))
    true_classes = target_scores[valid].argmax(-1)
    correct_probability = probabilities[valid].float().gather(1, true_classes[:, None]).squeeze(1)
    classification = average((correct_probability - quality[valid]).abs().clamp(0, 1))
    localization = average(1.0 - ious[valid].float().clamp(0, 1))
    difficulty = config.difficulty_alpha * classification + (1 - config.difficulty_alpha) * localization
    signals = {
        "joint": scales * difficulty,
        "scale": scales,
        "difficulty": difficulty,
        "additive": (scales + difficulty) / 2,
    }
    signal = signals[config.signal]
    mass = quality.new_zeros(count)
    mass.index_add_(0, inverse, quality[valid])
    if config.object_scope == "all":
        eligible = torch.ones(count, dtype=torch.bool, device=quality.device)
        center = (mass * signal).sum() / mass.sum()
        per_object = 1.0 + config.strength(epoch) * (signal - center)
    else:
        eligible = average(side) < config.small_side_threshold
        per_object = torch.ones_like(mass)
        if eligible.any():
            # Center within the small-object group; all other objects keep unit weight.
            center = (mass[eligible] * signal[eligible]).sum() / mass[eligible].sum()
            per_object[eligible] = 1.0 + config.strength(epoch) * (signal[eligible] - center)
    weights[valid] = per_object[inverse]
    eligible_mass = mass[eligible].sum()
    stats.update(
        weight_min=per_object.min(),
        weight_max=per_object.max(),
        mass_relative_error=((quality * weights).sum() - quality.sum()).abs() / quality.sum().clamp_min(1e-12),
        matched_objects=quality.new_tensor(count),
        eligible_objects=eligible.sum().float(),
        eligible_quality_mass=eligible_mass,
        eligible_mass_relative_error=(
            ((mass[eligible] * per_object[eligible]).sum() - eligible_mass).abs()
            / eligible_mass.clamp_min(1e-12)
        ),
        ineligible_weight_max_deviation=(
            (per_object[~eligible] - 1).abs().max() if (~eligible).any() else zero
        ),
        mean_scale=scales.mean(),
        mean_difficulty=difficulty.mean(),
    )
    return WeightResult(weights, stats)

