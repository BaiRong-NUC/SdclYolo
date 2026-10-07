"""Ultralytics 8.4.172 adapter; assignment and soft targets remain unchanged."""

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors

from .config import SDCLConfig
from .geometry import anchor_ignore_mask
from .weights import object_weights


def weighted_classification(
    logits, targets, weights, ignored=None, foreground=None, *, mode="positive_term",
):
    bce = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    if mode == "positive_term":
        bce = bce + (weights[..., None] - 1) * targets * F.softplus(-logits)
    elif mode == "full_bce":
        # Positive soft targets identify the matched object's true-class entries.
        # Scale both BCE terms there; other classes and background keep unit weight.
        bce = bce * torch.where(targets > 0, weights[..., None], 1.0)
    else:
        raise ValueError(f"Unknown classification weighting mode: {mode}")
    if ignored is not None:
        if foreground is None:
            raise ValueError("Foreground is required when ignoring negative locations.")
        bce = bce.masked_fill((ignored & ~foreground)[..., None], 0)
    return bce


class SDCLLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        if self.class_weights is not None:
            raise ValueError("Additional class weights are not supported.")
        self.settings = SDCLConfig.from_dict(model.sdcl_settings)
        self.epoch = getattr(model, "sdcl_epoch", 0)
        self.use_ignore = getattr(model, "sdcl_ignore_regions", True)
        self.diagnostics = {}
        self.steps = 0

    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        distances = preds["boxes"].permute(0, 2, 1).contiguous()
        logits = preds["scores"].permute(0, 2, 1).contiguous()
        anchors, stride = make_anchors(preds["feats"], self.stride, 0.5)
        image_size = tuple(batch["img"].shape[-2:])
        size_tensor = logits.new_tensor(image_size)
        targets = torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1)
        targets = self.preprocess(
            targets.to(self.device), logits.shape[0], size_tensor[[1, 0, 1, 0]]
        )
        gt_classes, gt_boxes = targets.split((1, 4), 2)
        valid_gt = gt_boxes.sum(2, keepdim=True) > 0
        decoded = self.bbox_decode(anchors, distances)
        _, boxes, scores, foreground, gt_indices = self.assigner(
            logits.detach().sigmoid(),
            (decoded.detach() * stride).to(gt_boxes.dtype),
            anchors * stride,
            gt_classes,
            gt_boxes,
            valid_gt,
        )
        denominator = scores.sum().clamp_min(1)
        ious = torch.zeros_like(scores[..., 0])
        if foreground.any():
            ious[foreground] = bbox_iou(
                decoded.detach()[foreground] * stride.expand(logits.shape[0], -1, -1)[foreground],
                boxes[foreground],
                xywh=False,
            ).squeeze(-1)
        result = object_weights(
            self.settings, self.epoch, boxes, scores, gt_indices, foreground,
            logits.detach().sigmoid(), ious, image_size,
        )
        ignored = None
        if self.use_ignore:
            if "ignore_bboxes" not in batch:
                raise ValueError("Ignore-aware loss requires the SDCL dataset adapter.")
            ignored = anchor_ignore_mask(
                anchors * stride, batch["ignore_bboxes"], batch["ignore_batch_idx"],
                logits.shape[0], image_size,
            )
        classification_weights = (
            result.weights if self.settings.apply_to in {"both", "classification"}
            else torch.ones_like(result.weights)
        )
        loss[1] = weighted_classification(
            logits, scores, classification_weights, ignored, foreground,
            mode=self.settings.classification_weighting,
        ).sum() / denominator
        # BboxLoss consumes these scores only as regression weights, not soft labels.
        regression_weights = (
            scores * result.weights[..., None]
            if self.settings.apply_to in {"both", "regression"} else scores
        )
        loss[0], loss[2] = self.bbox_loss(
            distances, decoded, anchors, boxes / stride, regression_weights,
            denominator, foreground, size_tensor, stride,
        )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        self.steps += 1
        if self.steps % self.settings.log_interval == 0:
            self.diagnostics = {key: float(value) for key, value in result.diagnostics.items()}
            self.diagnostics["total_gt"] = int(valid_gt.sum())
            self.diagnostics["unmatched_gt"] = int(valid_gt.sum()) - int(result.diagnostics["matched_objects"])
        return (
            (foreground, gt_indices, boxes, anchors, stride),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )

