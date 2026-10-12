"""Training-only repair of uncovered small GTs after TAL conflict resolution."""

from dataclasses import asdict, dataclass, fields
import math

import torch
from ultralytics.utils.tal import TaskAlignedAssigner


@dataclass(frozen=True)
class CoverageConfig:
    enabled: bool = False
    warmup_epochs: int = 5
    reference_size: int = 640
    max_side: float = 32.0
    minimum_ciou: float = 0.1
    minimum_relative_alignment: float = 0.25
    candidates: int = 13

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("coverage.enabled must be boolean.")
        for key in ("warmup_epochs", "reference_size", "candidates"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "warmup_epochs" else 1):
                raise ValueError(f"Invalid coverage.{key}.")
        if not math.isfinite(self.max_side) or self.max_side <= 0:
            raise ValueError("coverage.max_side must be finite and positive.")
        for key in ("minimum_ciou", "minimum_relative_alignment"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"coverage.{key} must be in (0, 1].")

    @classmethod
    def from_dict(cls, values=None):
        values = values or {}
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown or invalid coverage settings.")
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def repair_ownership(owners, counts, rows):
    """Repair a sparse candidate graph, never taking a donor's last positive.

    Each row is (GT id, [(anchor id, recipient relative quality, donor quality)]).
    Candidate edges have already passed true-box and localization checks.
    """
    changes = []
    for gt, edges in sorted(rows, key=lambda row: (len(row[1]), row[0])):
        if counts[gt]:
            continue
        options = []
        for anchor, quality, donor_quality in edges:
            donor = owners[anchor]
            if donor >= 0 and counts[donor] <= 1:
                continue
            cost = donor_quality / counts[donor] if donor >= 0 else 0.0
            options.append((donor < 0, quality - cost, quality, -anchor, anchor, donor))
        if not options:
            continue
        _, _, _, _, anchor, donor = max(options)
        if donor >= 0:
            counts[donor] -= 1
        owners[anchor] = gt
        counts[gt] += 1
        changes.append((gt, anchor, donor))
    return changes


class CoverageAssigner(TaskAlignedAssigner):
    def __init__(self, *args, coverage=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage = coverage or CoverageConfig()
        self.epoch = 0
        self.image_hw = (960, 960)
        self.statistics = {}

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if not self.coverage.enabled or self.epoch < self.coverage.warmup_epochs:
            return super()._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        _, _, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric)
        statistics = self._repair(mask_pos, align_metric, overlaps, anc_points, gt_bboxes, mask_gt)
        fg_mask, target_gt_idx = mask_pos.max(-2)
        labels, boxes, scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # Preserve TAL's quality labels and normalization, including its epsilon.
        align_metric *= mask_pos
        maximum_alignment = align_metric.amax(-1, keepdim=True)
        overlaps *= mask_pos
        maximum_overlap = overlaps.amax(-1, keepdim=True)
        align_metric.mul_(maximum_overlap).div_(maximum_alignment + self.eps)
        scores = scores * align_metric.amax(-2).unsqueeze(-1)
        # Commit after normalization succeeds, so an OOM retry cannot double-count.
        for key, value in statistics.items():
            self.statistics[key] = self.statistics.get(key, 0) + value
        return labels, boxes, scores, fg_mask.bool(), target_gt_idx

    def _repair(self, mask_pos, alignment, overlaps, anchors, gt, valid):
        productive = mask_pos.bool() & (alignment > 0)
        counts = productive.sum(-1)
        areas = (gt[..., 2:] - gt[..., :2]).prod(-1)
        scale = self.coverage.reference_size / max(self.image_hw)
        small = valid[..., 0].bool() & (areas * scale**2 < self.coverage.max_side**2)
        orphans = small & (counts == 0)
        keys = {"batches": 1, "small_gt": int(small.sum()),
                "small_uncovered_before": int(orphans.sum()), "rescued_gt": 0,
                "free_repairs": 0, "donor_transfers": 0,
                "eligible_orphans": 0, "donor_starvation": 0}
        for bi in range(self.bs):
            gt_ids = orphans[bi].nonzero().flatten()
            if not len(gt_ids):
                continue
            selected_gt = gt[bi, gt_ids]
            inside = ((anchors[None] > selected_gt[:, None, :2] + self.eps).all(-1)
                      & (anchors[None] < selected_gt[:, None, 2:] - self.eps).all(-1))
            eligible = inside & (overlaps[bi, gt_ids] >= self.coverage.minimum_ciou)
            values = alignment[bi, gt_ids].masked_fill(~eligible, 0)
            best = values.amax(-1, keepdim=True)
            values.masked_fill_(
                (values < best * self.coverage.minimum_relative_alignment) | (values <= 0), -1)
            quality, indices = values.topk(min(self.coverage.candidates, values.shape[-1]), dim=-1)
            relative = quality / best.clamp_min(torch.finfo(best.dtype).tiny)
            occupied, owners = productive[bi].max(0)
            owners = owners.masked_fill(~occupied, -1)
            edge_donors = owners[indices]
            donor_best = (alignment[bi] * productive[bi]).amax(-1)
            donor_quality = alignment[bi, edge_donors.clamp_min(0), indices]
            donor_quality = donor_quality / donor_best[edge_donors.clamp_min(0)].clamp_min(
                torch.finfo(donor_best.dtype).tiny)
            donor_quality.masked_fill_(edge_donors < 0, 0)
            owner_list = owners.cpu().tolist()
            count_list = counts[bi].cpu().tolist()
            rows = [
                (gt_id, [(anchor, q, dq) for anchor, q, dq in zip(ids, qs, dqs) if q > 0])
                for gt_id, ids, qs, dqs in zip(
                    gt_ids.cpu().tolist(), indices.cpu().tolist(),
                    relative.cpu().tolist(), donor_quality.cpu().tolist())
            ]
            keys["eligible_orphans"] += sum(bool(edges) for _, edges in rows)
            changes = repair_ownership(owner_list, count_list, rows)
            if changes:
                recipient_ids, anchor_ids, donors = zip(*changes)
                locations = torch.tensor(anchor_ids, device=mask_pos.device)
                mask_pos[bi, :, locations] = 0
                mask_pos[bi, torch.tensor(recipient_ids, device=mask_pos.device), locations] = 1
                keys["rescued_gt"] += len(changes)
                keys["free_repairs"] += sum(donor < 0 for donor in donors)
                keys["donor_transfers"] += sum(donor >= 0 for donor in donors)
        after = (mask_pos.bool() & (alignment > 0)).sum(-1)
        keys["donor_starvation"] = int(((counts > 0) & (after == 0)).sum())
        keys["small_uncovered_after"] = int((small & (after == 0)).sum())
        if (keys["donor_starvation"]
                or keys["small_uncovered_before"] - keys["small_uncovered_after"] != keys["rescued_gt"]):
            raise RuntimeError("Coverage repair violated the productive-ownership invariant.")
        return keys
